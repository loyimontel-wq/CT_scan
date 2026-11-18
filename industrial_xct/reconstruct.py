"""reconstruct.py - 3-D cone-beam reconstruction (CPU OpenMP FDK + optional ASTRA CUDA)."""
from __future__ import annotations

import ctypes
import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from industrial_xct.artifacts import (CoREstimate, GeometryCalibration,
                                      build_cone_vectors, project_point,
                                      slice_metrics)
from industrial_xct.config import (GeometryConfig, Issue, ReconstructionConfig,
                                   SUPPORTED_FILTERS)
from industrial_xct.io_xtek import CoverageReport, ScannerGeometry

logger = logging.getLogger(__name__)


class ReconstructionError(RuntimeError):
    """Reconstruction cannot be performed safely."""


# =========================================================================== #
# CoR resolution
# =========================================================================== #
def resolve_cor_offset_px(geo_cfg: GeometryConfig, geometry: ScannerGeometry,
                          estimate: Optional[CoREstimate]) -> Tuple[float, str, str]:
    src = geo_cfg.cor_source
    if src == "estimate":
        if estimate is None or estimate.delta_px is None:
            raise ReconstructionError("cor_source='estimate' but no CoR estimate was produced")
        if not estimate.accepted:
            raise ReconstructionError("CoR estimate was rejected: " +
                                      "; ".join(str(i) for i in estimate.issues))
        return float(estimate.delta_px), "estimated", estimate.method
    if src == "manual":
        return float(geo_cfg.cor_manual_px), "user_override", "geometry.cor_manual_px"
    if src == "none":
        return 0.0, "assumed", "no CoR correction: axis assumed at the detector centre (likely wrong)"
    raise ReconstructionError(f"cor_source={src!r} is not supported")


def make_calibration(geometry: ScannerGeometry, geo_cfg: GeometryConfig, n_rows: int,
                     n_cols: int, cor_offset_px: float) -> GeometryCalibration:
    g = geometry.require("sod_mm", "sdd_mm", "pixel_size_x_mm", "pixel_size_y_mm")
    roll = float(geometry.value("detector_roll_deg")) + float(geometry.value("rotation_axis_tilt_deg"))
    return GeometryCalibration(
        sod_mm=g["sod_mm"], sdd_mm=g["sdd_mm"], pixel_u_mm=g["pixel_size_x_mm"],
        pixel_v_mm=g["pixel_size_y_mm"], n_rows=n_rows, n_cols=n_cols,
        cor_offset_px=cor_offset_px,
        detector_offset_v_px=float(getattr(geo_cfg, "detector_offset_v_px", 0.0) or 0.0),
        detector_roll_deg=roll, detector_tilt_deg=float(geometry.value("detector_tilt_deg")),
        rotation_sign=int(getattr(geo_cfg, "rotation_sign", 1) or 1),
        angle_offset_deg=float(getattr(geo_cfg, "angle_offset_deg", 0.0) or 0.0),
        detector_u_sign=int(getattr(geo_cfg, "detector_u_sign", 1) or 1),
        detector_v_sign=int(getattr(geo_cfg, "detector_v_sign", -1) or -1))


# =========================================================================== #
# Volume specification and slab planning
# =========================================================================== #
@dataclass
class VolumeSpec:
    shape_zyx: Tuple[int, int, int]
    voxel_mm: float
    centre_mm: Tuple[float, float, float]
    notes: List[str] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        nz, ny, nx = self.shape_zyx
        return nz * ny * nx * 4

    def slab_bounds(self, k0: int, k1: int) -> Tuple[float, float, float, float, float, float]:
        nz, ny, nx = self.shape_zyx
        s = self.voxel_mm
        cx, cy, cz = self.centre_mm
        x0, y0, zb = cx - nx * s / 2, cy - ny * s / 2, cz - nz * s / 2
        return x0, x0 + nx * s, y0, y0 + ny * s, zb + k0 * s, zb + k1 * s


def resolve_volume_spec(geometry: ScannerGeometry, geo_cfg: GeometryConfig,
                        rec_cfg: ReconstructionConfig) -> VolumeSpec:
    notes: List[str] = []
    vox = rec_cfg.voxel_size_mm or geo_cfg.voxel_size_mm
    if vox is None:
        p = geometry.params.get("voxel_size_from_geometry_mm")
        if p is None or not p.known:
            raise ReconstructionError("voxel size unknown: set reconstruction.voxel_size_mm")
        vox = p.value
        notes.append("voxel size = pitch*SOD/SDD (calculated)")
    dims = rec_cfg.volume_voxels or geo_cfg.volume_voxels
    if dims is None:
        vals = [geometry.params[f"volume_voxels_{a}"] for a in "xyz"]
        if not all(v.known for v in vals):
            raise ReconstructionError("volume dimensions unknown: set reconstruction.volume_voxels")
        dims = tuple(int(v.value) for v in vals)
        notes.append("volume dimensions taken from metadata Voxels{X,Y,Z}")
    nx, ny, nz = dims
    cen = tuple(float(c) for c in (geo_cfg.volume_offset_mm or (0.0, 0.0, 0.0)))
    if cen == (0.0, 0.0, 0.0):
        notes.append("volume centred on the rotation axis / mid-plane (ASSUMED); Nikon's own "
                     "OffsetX/Y/Z are unverified and NOT applied")
    return VolumeSpec((nz, ny, nx), float(vox), cen, notes)  # type: ignore[arg-type]


def required_rows(vectors: np.ndarray, spec: VolumeSpec, k0: int, k1: int,
                  n_rows: int, n_cols: int, margin: int = 2) -> Tuple[int, int]:
    x0, x1, y0, y1, z0, z1 = spec.slab_bounds(k0, k1)
    rmin, rmax = np.inf, -np.inf
    for x in (x0, x1):
        for y in (y0, y1):
            for z in (z0, z1):
                r, _ = project_point(vectors, (x, y, z), n_rows, n_cols)
                rmin, rmax = min(rmin, float(r.min())), max(rmax, float(r.max()))
    return max(0, int(math.floor(rmin)) - margin), min(n_rows, int(math.ceil(rmax)) + 1 + margin)


@dataclass
class SlabPlan:
    mode: str
    slabs: List[Tuple[int, int, int, int]]
    host_bytes_peak: float
    gpu_heuristic_bytes: float
    needs_row_crop_verification: bool
    notes: List[str]


def detect_gpu_free_bytes(gpu_index: int) -> Optional[float]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                              "-i", str(gpu_index)], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0]) * 1024 ** 2
    except Exception:
        return None


def plan_reconstruction(spec: VolumeSpec, vectors: np.ndarray, n_proj: int, n_rows: int,
                        n_cols: int, rec_cfg: ReconstructionConfig,
                        gpu_free_bytes: Optional[float], allow_row_crop: bool = False) -> SlabPlan:
    nz, ny, nx = spec.shape_zyx
    host_lim = rec_cfg.host_memory_limit_gb * 1e9
    proj_full = float(n_proj) * n_rows * n_cols * 4
    notes: List[str] = []
    is_cpu = rec_cfg.algorithm == "FDK_CPU"
    gpu_lim = None if is_cpu else ((rec_cfg.gpu_memory_limit_gb * 1e9) if rec_cfg.gpu_memory_limit_gb else gpu_free_bytes)

    if rec_cfg.algorithm not in ("FDK_CUDA", "FDK_CPU"):
        need = proj_full + 3.0 * spec.nbytes + proj_full
        if need > host_lim:
            raise ReconstructionError(f"{rec_cfg.algorithm}: needs ~{need / 1e9:.1f} GB host memory (> {rec_cfg.host_memory_limit_gb} GB).")
        return SlabPlan("single", [(0, nz, 0, n_rows)], need, spec.nbytes + proj_full, False, notes)

    def fits_gpu(nzs: int) -> bool:
        return gpu_lim is None or nzs * ny * nx * 4 <= 0.35 * gpu_lim

    nz_slab = rec_cfg.max_slab_voxels_z or nz
    while nz_slab > 1 and not fits_gpu(nz_slab):
        nz_slab = int(math.ceil(nz_slab / 2))

    def build(nzs: int, crop: bool) -> Tuple[List[Tuple[int, int, int, int]], float]:
        slabs, peak = [], 0.0
        for k0 in range(0, nz, nzs):
            k1 = min(nz, k0 + nzs)
            r0, r1 = required_rows(vectors, spec, k0, k1, n_rows, n_cols) if crop else (0, n_rows)
            slabs.append((k0, k1, r0, r1))
            peak = max(peak, float(n_proj) * (r1 - r0) * n_cols * 4 + (k1 - k0) * ny * nx * 4)
        return slabs, peak

    slabs, peak = build(nz_slab, False)
    if peak <= host_lim:
        if nz_slab < nz:
            notes.append(f"z-slabs of {nz_slab} slices")
        return SlabPlan("volume_only", slabs, peak, nz_slab * ny * nx * 4.0, False, notes)
    if not allow_row_crop:
        raise ReconstructionError(
            f"Full projection stack needs {peak / 1e9:.1f} GB host RAM > host_memory_limit_gb="
            f"{rec_cfg.host_memory_limit_gb}. Pass --allow-row-crop to reconstruct in row-cropped z-slabs.")
    while True:
        slabs, peak = build(nz_slab, True)
        if peak <= host_lim or nz_slab == 1:
            break
        nz_slab = int(math.ceil(nz_slab / 2))
    if peak > host_lim:
        raise ReconstructionError(f"even 1-slice slabs need {peak / 1e9:.1f} GB host RAM > limit")
    notes.append(f"row-cropped z-slabs of {nz_slab} slices (rows cropped per slab; detector centre shifted)")
    return SlabPlan("row_crop", slabs, peak, nz_slab * ny * nx * 4.0, not is_cpu, notes)


# =========================================================================== #
# ASTRA & CPU FDK environment
# =========================================================================== #
@dataclass
class AstraCapabilities:
    installed: bool
    version: str = ""
    cuda: bool = False
    gpu_info: str = ""
    fdk_cone_vec: bool = False
    filters_ok: List[str] = field(default_factory=list)
    filters_failed: Dict[str, str] = field(default_factory=dict)
    short_scan: bool = False
    sirt_min_constraint: bool = False
    errors: List[str] = field(default_factory=list)


def _import_astra() -> Any:
    try:
        import astra  # noqa: WPS433
        return astra
    except Exception as exc:
        raise ReconstructionError(f"ASTRA toolbox is not importable: {exc}") from exc


def _tiny_vectors(n: int = 6, size: int = 8) -> Tuple[np.ndarray, int]:
    cal = GeometryCalibration(50.0, 100.0, 1.0, 1.0, size, size)


def check_algorithm_vs_coverage(algorithm: str, coverage: CoverageReport,
                                caps: Optional[AstraCapabilities] = None) -> Tuple[bool, List[Issue]]:
    """(use ShortScan weighting?, issues)."""
    I: List[Issue] = []
    short = False
    if algorithm in ("FDK_CUDA", "FDK_CPU"):
        if coverage.kind == "full_scan":
            pass
        elif coverage.kind == "short_scan":
            if algorithm == "FDK_CUDA" and caps is not None and not caps.short_scan:
                raise ReconstructionError("short-scan data need Parker weighting, which ASTRA probe rejected")
            short = True
            I.append(Issue("warning", "short_scan", "Short-scan: Parker weighting is enabled."))
        else:
            raise ReconstructionError(f"angular coverage is '{coverage.kind}': FDK is not valid.")
    elif coverage.kind in ("incomplete", "indeterminate"):
        I.append(Issue("warning", "iterative_incomplete", f"coverage '{coverage.kind}': limited-angle artifacts expected."))
    if coverage.sparse:
        I.append(Issue("warning", "sparse", "Sparse-angle acquisition: expect streaks with FDK."))
    return short, I
