"""Practical Pass 1 check on REAL data: read-only, writes nothing.

    python -m industrial_xct.examples.inspect_dataset --xtekct scan.xtekct --angles _ctdata.txt \
        [--projections /path/to/tiff_folder] [--id-regex "(\\d+)\\.tiff?$"]

Without --projections only metadata and angles are examined. Everything printed is a
finding about the FILES; none of it proves the reconstruction will be correct.
"""
import argparse
import logging

from industrial_xct import config as C
from industrial_xct import io_xtek as io


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xtekct", required=True)
    ap.add_argument("--angles", required=True)
    ap.add_argument("--projections")
    ap.add_argument("--id-regex", default=C.InputConfig().projection_id_regex)
    ap.add_argument("--id-offset", type=int, default=0)
    ap.add_argument("--tolerance-deg", type=float, default=1.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    meta = io.parse_xtekct(a.xtekct)
    geo = io.build_scanner_geometry(meta, C.GeometryConfig())
    print("== GEOMETRY (value | provenance | unit | note) ==")
    for k, p in geo.params.items():
        print(f"{k:28s} {str(p.value):>22s} | {p.provenance.value:12s} | {p.unit or '-':4s} | {p.note}")
    for i in io.validate_geometry(geo):
        print(i)
    print("\n== METADATA FIELDS NOT INTERPRETED (preserved) ==")
    for sec, kv in meta.unconsumed_keys().items():
        print(f"[{sec}] " + ", ".join(f"{k}={v}" for k, v in kv.items()))

    table = io.read_angle_file(a.angles)
    rep = io.validate_angle_table(table, expected_projections=meta.get_int("Projections"))
    print(f"\n== ANGLES == {table.source.name}: {rep.n_entries} entries, ids {rep.id_min}..{rep.id_max}, "
          f"direction {rep.direction}, step median {rep.step_median_deg:.5f} "
          f"(min {rep.step_min_deg:.3f}, max {rep.step_max_deg:.3f}) deg")
    fan = geo.params["fan_half_angle_deg"].value
    cov = io.classify_angular_coverage(rep.unwrapped_deg, fan_half_angle_deg=fan,
                                       n_detector_cols=geo.params["detector_pixels_x"].value,
                                       full_scan_tolerance_deg=a.tolerance_deg)
    print(f"coverage: {cov.kind}; span {cov.span_deg:.3f} deg; closure gap {cov.closure_gap_deg:.3f} deg "
          f"({cov.closure_gap_steps:.2f} steps)")
    for n in cov.notes:
        print("  -", n)
    for i in rep.issues + io.compare_metadata_to_angles(meta, table, rep):
        print(i)

    if a.projections:
        disc = io.discover_tiff_projections(a.projections, id_regex=a.id_regex)
        m = io.match_projections(disc.files_by_id, table, id_offset=a.id_offset,
                                 duplicate_image_ids=list(disc.duplicate_ids))
        print(f"\n== MATCHING == {len(disc.files_by_id)} image ids, {len(m.matched)} matched")
        for i in m.issues:
            print(i)
        if m.matched:
            src = io.TiffSequenceSource([x.path for x in m.matched], [x.file_id for x in m.matched])
            exp = io.expected_image_shape(geo)
            print(f"image shape {src.image_shape}, dtype {src.dtype}; expected from metadata {exp}")
            for i in src.verify_headers()[:10]:
                print(i)
            wl = geo.params["white_level_raw"].value
            r = io.inspect_projection_intensity(src, 8, wl)
            print("\n== INTENSITY (8 evenly spaced projections) ==")
            print({k: v for k, v in r.percentiles.items()})
            for line in r.verdict:
                print(" -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
