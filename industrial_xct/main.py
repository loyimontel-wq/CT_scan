"""main.py - command-line entry point: runs the whole pipeline in the order of the project spec.

    python -m industrial_xct.main --config my.yaml                  # full run
    python -m industrial_xct.main --config my.yaml --dry-run        # validate + plan, no heavy work
    python -m industrial_xct.main --config my.yaml --estimate-cor-only
    python -m industrial_xct.main --config my.yaml --probe-astra    # what does THIS ASTRA support?
    python -m industrial_xct.main --config my.yaml --verify-row-crop
    python -m industrial_xct.main --config my.yaml --cor-sweep

Any config value can be overridden:  --set preprocessing.batch_size=16 --set geometry.cor_source=manual

Stages (spec section 4, module 6):
 1 config  2 metadata  3 angles  4 discover images  5 match ids<->angles  6 validate geometry
 7 memory estimate  8 calibration images  9 preprocess  10 CoR + corrections  11 geometry vectors
 12 reconstruct  13 export  14 diagnostic plots  15 report (Markdown + JSON + log)
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from industrial_xct import artifacts as art
from industrial_xct import config as cfgmod
from industrial_xct import io_xtek as io
from industrial_xct import preprocessing as pp
from industrial_xct import reconstruct as rec

logger = logging.getLogger("industrial_xct")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 5000 else f"<array shape={x.shape} dtype={x.dtype}>"
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (Path, enum.Enum)):
        return str(x.value if isinstance(x, enum.Enum) else x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    return x


def _parse_overrides(items: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise cfgmod.ConfigError(f"--set expects section.key=value, got {it!r}")
        key, val = it.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def setup_logging(out_dir: Path, level: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out_dir / "reconstruction.log", "w")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logging.getLogger("industrial_xct").propagate = False
    for name in ("industrial_xct.config", "industrial_xct.io_xtek", "industrial_xct.preprocessing",
                 "industrial_xct.artifacts", "industrial_xct.reconstruct"):
        lg = logging.getLogger(name)
        lg.handlers, lg.propagate = [], True
        lg.setLevel(logger.level)


def _tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm
        bar = tqdm(total=total, desc=desc, unit="proj")
        last = [0]

        def cb(done: int, _total: int) -> None:
            bar.update(done - last[0])
            last[0] = done
            if done >= total:
                bar.close()
        return cb
    except ImportError:
        return None


# --------------------------------------------------------------------------- #
# diagnostics (matplotlib is imported lazily so the pipeline runs without it)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_projection_pair(raw: np.ndarray, corrected: np.ndarray, idx: int, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    im0 = ax[0].imshow(raw, cmap="gray"); ax[0].set_title(f"raw projection #{idx} ({raw.dtype}, counts)")
    im1 = ax[1].imshow(corrected, cmap="gray"); ax[1].set_title("corrected: line integral P = -ln(T)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046); fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_sinograms(before: np.ndarray, after: Optional[np.ndarray], row: int, path: Path) -> None:
    plt = _plt()
    n = 2 if after is not None else 1
    fig, ax = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    ax[0][0].imshow(before, cmap="gray", aspect="auto"); ax[0][0].set_title(f"sinogram, row {row} (before stripe correction)")
    if after is not None:
        ax[0][1].imshow(after, cmap="gray", aspect="auto"); ax[0][1].set_title("after stripe correction")
    for a in ax[0]:
        a.set_xlabel("detector column"); a.set_ylabel("projection (angle order)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_slices(vol: np.ndarray, path: Path) -> None:
    plt = _plt()
    nz, ny, nx = vol.shape
    xy, xz, yz = vol[nz // 2], vol[:, ny // 2, :], vol[:, :, nx // 2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, im, t in zip(ax, (xy, xz, yz), ("XY (z mid)", "XZ (y mid)", "YZ (x mid)")):
        lo, hi = np.percentile(im, [0.5, 99.5])
        a.imshow(im, cmap="gray", vmin=lo, vmax=hi, aspect="equal"); a.set_title(t)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_angles(unwrapped: np.ndarray, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(unwrapped, ".", ms=1); ax[0].set_title("unwrapped angle vs projection"); ax[0].set_xlabel("index"); ax[0].set_ylabel("deg")
    ax[1].hist(np.diff(unwrapped), bins=60); ax[1].set_title("angular step histogram (deg)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_cor(est: art.CoREstimate, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(est.pair_estimates_px, "o-"); ax[0].axhline(est.delta_px, color="r", ls="--")
    ax[0].set_title("delta per opposing pair (px)"); ax[0].set_xlabel("pair #")
    ax[1].plot(est.band_centres_px, est.band_estimates_px, "o-"); ax[1].axhline(est.delta_px, color="r", ls="--")
    ax[1].set_title(f"delta per row band (slope*H = {est.row_variation_px} px)"); ax[1].set_xlabel("detector row")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_cor_sweep(sweep: Dict[str, Any], est_delta: Optional[float], path: Path) -> None:
    plt = _plt()
    sl, d = sweep["slices"], sweep["deltas_px"]
    n = len(sl)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows + 1, cols, figsize=(3.2 * cols, 3.2 * (rows + 1)), squeeze=False)
    for k in range(rows * cols):
        a = ax[k // cols][k % cols]
        a.axis("off")
        if k < n:
            lo, hi = np.percentile(sl[k], [0.5, 99.5])
            a.imshow(sl[k], cmap="gray", vmin=lo, vmax=hi); a.set_title(f"delta={d[k]:+.2f} px", fontsize=9)
    for a in ax[rows]:
        a.axis("off")
    axm = fig.add_subplot(rows + 1, 1, rows + 1)
    for key in ("gradient_energy", "total_variation"):
        v = np.array([m[key] for m in sweep["metrics"]], float)
        axm.plot(d, (v - v.min()) / (np.ptp(v) + 1e-30), "o-", label=key + " (min-max scaled)")
    if est_delta is not None:
        axm.axvline(est_delta, color="r", ls="--", label="opposing-pair estimate")
    axm.set_xlabel("trial CoR offset (px)"); axm.legend(fontsize=8)
    axm.set_title("descriptive indicators only: sharpest != correct; judge with the geometric tests")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
VALIDATION_TEXT = """## What this run does and does not establish
* Establishes: internal consistency of files, angle/projection pairing, geometry self-consistency,
  numerical safeguards, CoR agreement between opposing pairs (if estimated).
* Does NOT establish: that the reconstruction matches Nikon CT Pro 3D in accuracy or image quality;
  correct orientation/handedness (rotation_sign, detector_u_sign, detector_v_sign are ASSUMED);
  absolute attenuation scaling; correct physical scale in VGSTUDIO MAX.
* Needed for that: a reference object of known dimensions or a Nikon CT Pro 3D reconstruction of the
  same scan to compare slices and measurements, a known asymmetric (chiral) object for orientation,
  forward-projection (reprojection) residuals, and a VGSTUDIO MAX import check of the exported file.
"""


def write_report(path: Path, S: Dict[str, Any]) -> None:
    L: List[str] = ["# Reconstruction report", "",
                    f"Generated: {S['generated']}  |  package version {S['version']}", ""]
    L += ["## Geometry (value | provenance | unit | note)", "", "| parameter | value | provenance | unit | note |", "|---|---|---|---|---|"]
    for k, p in S["geometry"].items():
        L.append(f"| {k} | {p['value']} | {p['provenance']} | {p['unit'] or ''} | {p['note']} |")
    for title, key in (("Geometry findings", "geometry_issues"), ("Angle findings", "angle_issues"),
                       ("Matching findings", "match_issues"), ("Config warnings", "config_issues")):
        if S.get(key):
            L += ["", f"## {title}", ""] + [f"* {i}" for i in S[key]]
    if "coverage" in S:
        c = S["coverage"]
        L += ["", "## Angular coverage", "",
              f"* kind: **{c['kind']}**, span {c['span_deg']:.3f} deg, closure gap {c['closure_gap_deg']:.3f} deg"] + \
             [f"* {n}" for n in c["notes"]]
    if "intensity" in S:
        L += ["", "## Detector intensity (sampled projections, raw values unchanged)", ""] + \
             [f"* {v}" for v in S["intensity"]["verdict"]]
    if "preprocessing" in S:
        L += ["", "## Preprocessing", "", "```", json.dumps(S["preprocessing"], indent=1), "```"]
    if "cor" in S:
        c = S["cor"]
        L += ["", "## Centre of rotation", "",
              f"* source used for reconstruction: **{S['cor_used']['source']}** = {S['cor_used']['delta_px']} px "
              f"({S['cor_used']['provenance']}; {S['cor_used']['note']})",
              f"* image-based estimate: delta = {c['delta_px']} px = {c['delta_detector_mm']} mm on the detector = "
              f"{c['delta_object_mm']} mm at the rotation axis; accepted = **{c['accepted']}**",
              f"* method: {c['method']}; pairs used {c['n_pairs_used']}; pair std {c['pair_std_px']} px; "
              f"CoR change across rows {c['row_variation_px']} px; mean vertical shift {c['vertical_shift_px']} px",
              "* pairs (index i, j, separation deg, residual deg): " +
              "; ".join(f"({p['i']},{p['j']},{p['separation_deg']:.3f},{p['residual_deg']:.3f})" for p in c["pairs"])]
        L += [f"* {i}" for i in c["issues"]] + [f"* {n}" for n in c["notes"]]
    elif "cor_used" in S:
        L += ["", "## Centre of rotation", "", f"* {S['cor_used']}"]
    if S.get("corrections"):
        L += ["", "## Artifact corrections", "", "```", json.dumps(S["corrections"], indent=1), "```"]
    if "reconstruction" in S:
        L += ["", "## Reconstruction", "", "```", json.dumps(S["reconstruction"], indent=1), "```"]
    if "export" in S:
        L += ["", "## Export", "", "```", json.dumps(S["export"], indent=1), "```"]
    L += ["", VALIDATION_TEXT]
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def _issue_strs(issues) -> List[str]:
    return [str(i) for i in issues]


def run_pipeline(cfg: cfgmod.PipelineConfig, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    out_dir = Path(cfg.export.output_dir)
    setup_logging(out_dir, cfg.log_level)
    from industrial_xct import __version__
    S: Dict[str, Any] = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "version": __version__,
                         "config": cfgmod.config_to_dict(cfg)}
    summary_path, report_path = out_dir / "summary.json", out_dir / "report.md"

    def finish(reason: str) -> Dict[str, Any]:
        S["finished"] = reason
        S["seconds"] = round(time.time() - t0, 1)
        summary_path.write_text(json.dumps(_jsonable(S), indent=2), encoding="utf-8")
        write_report(report_path, _jsonable(S))
        logger.info("Finished (%s). Report: %s", reason, report_path)
        return S

    # 1. configuration ------------------------------------------------------
    S["config_issues"] = _issue_strs(cfgmod.validate_or_raise(cfg))
    if args.probe_astra:
        caps = rec.probe_astra(cfg.reconstruction.gpu_index)
        S["astra"] = _jsonable(caps)
        logger.info("ASTRA probe: %s", json.dumps(_jsonable(caps), indent=1))
        return finish("astra probe")

    # 2. metadata -------------------------------------------------------------
    if not cfg.input.metadata_path:
        raise cfgmod.ConfigError("input.metadata_path is required (scanner geometry must come from a file or overrides)")
    meta = io.parse_xtekct(cfg.input.metadata_path)
    geometry = io.build_scanner_geometry(meta, cfg.geometry)
    S["geometry"] = geometry.to_dict()
    S["geometry_issues"] = _issue_strs(io.validate_geometry(geometry))
    S["unconsumed_metadata"] = meta.unconsumed_keys()
    cfgmod.raise_on_errors(io.validate_geometry(geometry), "Scanner geometry")

    # 3. angles -----------------------------------------------------------------
    table = io.read_angle_file(cfg.input.angle_path)
    rep = io.validate_angle_table(table, expected_projections=meta.get_int("Projections"))
    S["angle_issues"] = _issue_strs(rep.issues + io.compare_metadata_to_angles(meta, table, rep))
    if cfg.geometry.acquisition_direction not in ("auto", rep.direction):
        raise cfgmod.ConfigError(f"geometry.acquisition_direction={cfg.geometry.acquisition_direction} but the "
                                 f"angle table is {rep.direction}")
    cfgmod.raise_on_errors(rep.issues, "Angle table")

    # 4-5. images and matching ------------------------------------------------------
    source, match = io.build_source(cfg.input, table)
    S["match_issues"] = _issue_strs(match.issues)
    S["match"] = {"n_matched": len(match.matched), "images_without_angle": match.images_without_angle,
                  "angles_without_image": match.angles_without_image}
    idx = np.array([m.table_index for m in match.matched])
    unwrapped = rep.unwrapped_deg[idx]
    n_proj = len(source)
    H, W = source.image_shape

    # 6. geometry validation ---------------------------------------------------------
    exp = io.expected_image_shape(geometry)
    if exp is not None and tuple(exp) != (H, W):
        raise cfgmod.ConfigError(f"projection images are {H}x{W} but metadata expects {exp[0]}x{exp[1]}")
    for i in source.verify_headers()[:10]:
        S["match_issues"].append(str(i))
    cfgmod.raise_on_errors(source.verify_headers(), "Projection files")
    cov = io.classify_angular_coverage(
        unwrapped, fan_half_angle_deg=geometry.value("fan_half_angle_deg"), n_detector_cols=W,
        full_scan_tolerance_deg=cfg.geometry.full_scan_tolerance_deg)
    S["coverage"] = _jsonable(cov)
    logger.info("coverage: %s (span %.3f deg, gap %.3f deg)", cov.kind, cov.span_deg, cov.closure_gap_deg)
    short_scan, cov_issues = rec.check_algorithm_vs_coverage(cfg.reconstruction.algorithm, cov, None)
    S["config_issues"] += _issue_strs(cov_issues)
    wl = geometry.params["white_level_raw"].value
    intens = io.inspect_projection_intensity(source, cfg.preprocessing.inspect_sample_projections, wl)
    S["intensity"] = _jsonable(intens)
    for v in intens.verdict:
        logger.info("intensity: %s", v)

    # 7. memory / plan -----------------------------------------------------------------
    spec = rec.resolve_volume_spec(geometry, cfg.geometry, cfg.reconstruction)
    delta_for_plan = cfg.geometry.cor_manual_px or 0.0
    cal_plan = rec.make_calibration(geometry, cfg.geometry, H, W, delta_for_plan)
    mem = pp.estimate_memory(n_proj, H, W, cfg.preprocessing.batch_size)
    S["reconstruction"] = {"volume_shape_zyx": spec.shape_zyx, "voxel_mm": spec.voxel_mm,
                           "volume_notes": spec.notes, "preprocessed_stack_gb": mem["output_bytes"] / 1e9,
                           "orientation_assumptions": {"rotation_sign": cfg.geometry.rotation_sign,
                                                       "detector_u_sign": cfg.geometry.detector_u_sign,
                                                       "detector_v_sign": cfg.geometry.detector_v_sign,
                                                       "status": "ASSUMED, not validated"}}
    try:
        plan = rec.plan_reconstruction(spec, art.build_cone_vectors(unwrapped, cal_plan), n_proj, H, W,
                                       cfg.reconstruction, rec.detect_gpu_free_bytes(cfg.reconstruction.gpu_index),
                                       allow_row_crop=args.allow_row_crop)
        S["reconstruction"].update({"plan_mode": plan.mode, "n_slabs": len(plan.slabs),
                                    "slabs_k0_k1_r0_r1": plan.slabs, "host_peak_gb": plan.host_bytes_peak / 1e9,
                                    "plan_notes": plan.notes})
    except rec.ReconstructionError as exc:
        if not args.dry_run:
            raise
        S["reconstruction"]["plan_error"] = str(exc)
        logger.warning("plan: %s", exc)
    plot_angles(unwrapped, out_dir / "diag_angles.png")
    if args.dry_run:
        return finish("dry run")

    # 8-9. calibration + preprocessing -------------------------------------------------------
    cal = pp.load_calibration(cfg.input, cfg.preprocessing, (H, W))
    S["geometry_issues"] += _issue_strs(cal.issues)
    pre_cfg = cfg.preprocessing
    if pre_cfg.work_dir is None:
        pre_cfg.work_dir = str(out_dir / "work")
    stack, pstats, pinfo = pp.preprocess_stack(source, cal, pre_cfg)
    S["preprocessing"] = {"stats": _jsonable(pstats), "memory": pinfo, "calibration_info": cal.info}
    mid = n_proj // 2
    plot_projection_pair(source.read(mid), np.asarray(stack[mid]), mid, out_dir / "diag_projection.png")
    srow = H // 2
    sino_before = np.array(stack[:, srow, :])

    # 10. CoR + corrections --------------------------------------------------------------------
    estimate: Optional[art.CoREstimate] = None
    if cfg.artifacts.cor.enabled:
        estimate = art.estimate_cor(stack, unwrapped, cfg.artifacts.cor,
                                    pixel_u_mm=geometry.value("pixel_size_x_mm"),
                                    magnification=geometry.value("magnification"))
        raw_fields = {k: geometry.value(k) for k in ("cor_top_raw", "cor_bottom_raw") if geometry.params[k].known}
        for nme in ("ObjectOffsetX",):
            v = meta.get_float(nme)
            if v is not None:
                raw_fields[nme] = v
        estimate.issues += art.compare_cor_to_metadata(estimate.delta_px, raw_fields,
                                                       geometry.value("pixel_size_x_mm"), geometry.value("magnification"))
        S["cor"] = estimate.to_dict()
        plot_cor(estimate, out_dir / "diag_cor.png")
        logger.info("CoR estimate: %.3f px (accepted=%s)", estimate.delta_px, estimate.accepted)
    if args.estimate_cor_only:
        S["cor_used"] = {"source": cfg.geometry.cor_source, "delta_px": None, "provenance": "n/a",
                         "note": "estimate-only run: nothing applied, nothing reconstructed"}
        return finish("CoR estimation only")
    delta, prov, note = rec.resolve_cor_offset_px(cfg.geometry, geometry, estimate)   # the ONE place CoR is chosen
    S["cor_used"] = {"source": cfg.geometry.cor_source, "delta_px": delta, "provenance": prov, "note": note}

    corr: Dict[str, Any] = {}
    a = cfg.artifacts
    if a.beam_hardening.enabled:
        for s in range(0, n_proj, pre_cfg.batch_size):
            blk = np.asarray(stack[s:s + pre_cfg.batch_size])
            stack[s:s + blk.shape[0]] = art.apply_beam_hardening(blk, a.beam_hardening)
        corr["beam_hardening"] = {"coefficients": a.beam_hardening.coefficients,
                                  "calibrated": a.beam_hardening.coefficients_are_calibrated}
    if a.metal.enabled:
        corr["metal"] = art.apply_metal_trace(stack, a.metal)
    sino_after = None
    if a.ring.enabled:
        corr["ring"] = art.suppress_rings(stack, a.ring)
        sino_after = np.array(stack[:, srow, :])
    S["corrections"] = corr
    plot_sinograms(sino_before, sino_after, srow, out_dir / "diag_sinogram.png")

    # 11-12. geometry vectors + reconstruction ------------------------------------------------------
    cal_geo = rec.make_calibration(geometry, cfg.geometry, H, W, delta)
    caps = rec.probe_astra(cfg.reconstruction.gpu_index, [cfg.reconstruction.filter_name.lower()])
    S["astra"] = _jsonable(caps)
    if not caps.fdk_cone_vec:
        raise rec.ReconstructionError("ASTRA FDK_CUDA with cone_vec is not available: " + "; ".join(caps.errors))
    if cfg.reconstruction.filter_name.lower() not in caps.filters_ok and cfg.reconstruction.filter_name.lower() != "ram-lak":
        raise rec.ReconstructionError(f"filter {cfg.reconstruction.filter_name!r} failed the ASTRA probe: {caps.filters_failed}")
    if cfg.reconstruction.non_negativity and not caps.sirt_min_constraint:
        raise rec.ReconstructionError("this ASTRA did not accept MinConstraint for SIRT3D_CUDA")
    short_scan, _ = rec.check_algorithm_vs_coverage(cfg.reconstruction.algorithm, cov, caps)
    plan = rec.plan_reconstruction(spec, art.build_cone_vectors(unwrapped, cal_geo), n_proj, H, W, cfg.reconstruction,
                                   rec.detect_gpu_free_bytes(cfg.reconstruction.gpu_index), allow_row_crop=args.allow_row_crop)
    if args.verify_row_crop or plan.needs_row_crop_verification:
        v = rec.verify_row_crop(stack, unwrapped, cal_geo, spec, cfg.reconstruction, short_scan)
        S["reconstruction"]["row_crop_verification"] = v
        logger.info("row-crop verification: %s", v)
        if not v["passed"]:
            raise rec.ReconstructionError(f"row-crop verification FAILED ({v}); ASTRA's FDK does not treat the shifted "
                                          f"cropped detector like the full one. Use more host RAM or crop/bin differently.")
        if args.verify_row_crop:
            return finish("row-crop verification only")
    if args.cor_sweep or cfg.artifacts.cor.sweep_values_px:
        c0 = delta
        deltas = cfg.artifacts.cor.sweep_values_px or list(np.arange(
            c0 - cfg.artifacts.cor.sweep_half_range_px,
            c0 + cfg.artifacts.cor.sweep_half_range_px + 1e-9, cfg.artifacts.cor.sweep_step_px))
        sw = rec.cor_sweep(stack, unwrapped, cal_geo, spec, cfg.reconstruction, short_scan, deltas,
                           cfg.reconstruction.sweep_slab_slices, args.allow_row_crop,
                           rec.detect_gpu_free_bytes(cfg.reconstruction.gpu_index))
        S["cor_sweep"] = {"deltas_px": sw["deltas_px"], "metrics": sw["metrics"], "slab": sw["slab"]}
        plot_cor_sweep(sw, delta, out_dir / "diag_cor_sweep.png")
        if args.cor_sweep:
            return finish("CoR sweep")
    volume = rec.reconstruct_volume(stack, unwrapped, cal_geo, spec, cfg.reconstruction, plan, short_scan,
                                    work_dir=out_dir / "work", progress=_tqdm_slabs())
    S["reconstruction"].update({"algorithm": cfg.reconstruction.algorithm, "filter": cfg.reconstruction.filter_name,
                                "short_scan_weighting": short_scan,
                                "value_range": [float(np.min(volume)), float(np.max(volume))]})

    # 13-14. export + plots ------------------------------------------------------------------------------
    sp = (spec.voxel_mm,) * 3
    x0, _, y0, _, z0, _ = spec.slab_bounds(0, spec.shape_zyx[0])
    exp_info: Dict[str, Any] = {}
    base = out_dir / cfg.export.basename
    if "tiff32" in cfg.export.formats:
        exp_info["tiff32"] = io.write_tiff_float32(volume, str(base) + "_f32.tif", sp).to_dict()
    vmin = vmax = None
    if "tiff16" in cfg.export.formats:
        vmin, vmax = io.choose_window(volume, cfg.export.tiff16_window_mode, cfg.export.window_min,
                                      cfg.export.window_max, cfg.export.percentile_low, cfg.export.percentile_high)
        exp_info["tiff16"] = io.write_tiff_uint16(volume, str(base) + "_u16.tif", sp, vmin, vmax).to_dict()
    if "vol" in cfg.export.formats:
        exp_info["vol"] = io.write_vol_vgi(volume, str(base) + "_f32.vol", sp, origin_mm=(x0, y0, z0)).to_dict()
        exp_info["vol"]["note"] = ".vgi layout unverified against VGSTUDIO MAX; see the .vol.json sidecar for raw-import parameters"
    S["export"] = exp_info
    plot_slices(volume, out_dir / "diag_slices.png")
    return finish("complete")


def _tqdm_slabs():
    try:
        from tqdm import tqdm
        bar = {"b": None}

        def cb(done: int, total: int) -> None:
            if bar["b"] is None:
                bar["b"] = tqdm(total=total, desc="slabs", unit="slab")
            bar["b"].update(1)
            if done >= total:
                bar["b"].close()
        return cb
    except ImportError:
        return None


def probe_air(cfg: cfgmod.PipelineConfig, region: str) -> int:
    """Print raw-count statistics of a candidate air region in sample projections (no files written)."""
    x0, x1, y0, y1 = (int(v) for v in region.split(","))
    table = io.read_angle_file(cfg.input.angle_path)
    source, _ = io.build_source(cfg.input, table)
    idx = np.unique(np.linspace(0, len(source) - 1, 8).astype(int))
    print("projection  region_median  region_CV   image_p99")
    for i in idx:
        im = source.read(int(i)).astype(np.float64)
        r = im[y0:y1, x0:x1]
        print(f"{i:10d}  {np.median(r):13.1f}  {r.std() / max(np.median(r), 1e-9):9.4f}  {np.percentile(im, 99):10.1f}")
    print("A valid air region: median close to the image p99 in EVERY projection and a small CV.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="industrial_xct", description="Cone-beam XCT reconstruction pipeline")
    ap.add_argument("--config", help="YAML configuration (see config/default.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value")
    ap.add_argument("--dry-run", action="store_true", help="validate files, matching, geometry and memory plan only")
    ap.add_argument("--probe-astra", action="store_true", help="run tiny ASTRA jobs to list what this install supports")
    ap.add_argument("--probe-air", metavar="X0,X1,Y0,Y1", help="print statistics of a candidate air region and exit")
    ap.add_argument("--estimate-cor-only", action="store_true", help="stop after the CoR estimate")
    ap.add_argument("--cor-sweep", action="store_true", help="reconstruct a thin slab at several CoR values and stop")
    ap.add_argument("--verify-row-crop", action="store_true", help="test row-cropped vs full-detector FDK and stop")
    ap.add_argument("--allow-row-crop", action="store_true", help="permit row-cropped slabs (needs verification)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = cfgmod.load_config(args.config, _parse_overrides(args.set))
        if args.probe_air:
            return probe_air(cfg, args.probe_air)
        run_pipeline(cfg, args)
        return 0
    except (cfgmod.ConfigError, io.MetadataError, io.AngleFileError, io.ProjectionLoadError,
            io.ProjectionMatchError, pp.PreprocessingError, art.ArtifactError, rec.ReconstructionError,
            io.ExportError) as exc:
        logging.getLogger("industrial_xct").error("%s: %s", type(exc).__name__, exc)
        print(f"\nERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
