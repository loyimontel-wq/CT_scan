"""Pass 1 tests. Synthetic data only: they prove the CODE does what it claims, not that your
scanner data are reconstructed correctly (that needs real-data validation, see PASS1_NOTES.md).

Run from the directory that contains the industrial_xct folder:
    python -m pytest industrial_xct/tests -v
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile

from industrial_xct import config as C
from industrial_xct import io_xtek as io
from industrial_xct import preprocessing as pp

XTEKCT = """[XTekCT]\r
Name=test\r
SrcToObject=200.0\r
SrcToDetector=1000.0\r
DetectorPixelsX=64\r
DetectorPixelsY=48\r
DetectorPixelSizeX=0.2\r
DetectorPixelSizeY=0.2\r
DetectorOffsetX=0.0\r
DetectorOffsetY=0.0\r
CentreOfRotationTop=0.0\r
CentreOfRotationBottom=0.0\r
WhiteLevel=60000\r
VoxelSizeX=0.04\r
Projections=10\r
InitialAngle=12.0\r
AngularStep=36.0\r
Units=mm\r
MysteryKey=42\r
[Xrays]\r
XraykV=130\r
"""


def make_ctdata(n_rows=11, step=36.0, start=0.0, declared=10, skip=()):
    lines = ["Projections\tFrames\tExposure(ms)", f"{declared}\t1\t500.00", "Projection\tAngle(deg)\tTime(s)"]
    for i in range(n_rows):
        if i + 1 in skip:
            continue
        lines.append(f"{i + 1}\t{(start + i * step) % 360:07.3f}\t{i * 1.0:07.3f}")
    return "\r\n".join(lines) + "\r\n"


# ------------------------------- config ----------------------------------- #
def test_config_defaults_are_safe():
    cfg = C.load_config()
    a = cfg.artifacts
    assert not any([a.cor.enabled, a.ring.enabled, a.beam_hardening.enabled, a.scatter.enabled, a.metal.enabled])
    assert cfg.geometry.cor_source == "none"


def test_config_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("geometry:\n  sod_mmm: 100\n")
    with pytest.raises(C.ConfigError, match="sod_mmm"):
        C.load_config(p)


def test_config_type_error_rejected():
    with pytest.raises(C.ConfigError):
        C.load_config(overrides={"preprocessing": {"batch_size": "many"}})


def test_cor_applied_exactly_once_is_enforced():
    cfg = C.load_config(overrides={"artifacts": {"cor": {"enabled": True}}})
    codes = [i.code for i in C.validate_config(cfg, require_inputs=False)]
    assert "cor_double" in codes


def test_filter_cutoff_refused_until_verified():
    cfg = C.load_config(overrides={"reconstruction": {"filter_cutoff": 0.8}})
    assert any(i.code == "filter_cutoff" for i in C.validate_config(cfg, require_inputs=False))


def test_yaml_roundtrip(tmp_path):
    cfg = C.load_config(overrides={"geometry": {"volume_voxels": [4, 5, 6]}})
    C.dump_yaml(cfg, tmp_path / "o.yaml")
    assert C.load_config(tmp_path / "o.yaml").geometry.volume_voxels == (4, 5, 6)


# ------------------------------- metadata --------------------------------- #
def test_xtekct_parse_and_geometry(tmp_path):
    p = tmp_path / "a.xtekct"
    p.write_text(XTEKCT)
    meta = io.parse_xtekct(p)
    assert meta.get_int("Projections") == 10
    assert meta.unconsumed_keys()["XTekCT"]["MysteryKey"] == "42"          # preserved, not dropped
    g = io.build_scanner_geometry(meta, C.GeometryConfig())
    assert g.params["sod_mm"].provenance is C.Provenance.METADATA
    assert g.value("magnification") == pytest.approx(5.0)
    assert g.value("voxel_size_from_geometry_mm") == pytest.approx(0.04)   # 0.2*200/1000
    assert g.params["detector_tilt_deg"].provenance is C.Provenance.ASSUMED
    assert any(i.code == "zero_uncalibrated" for i in io.validate_geometry(g))
    # 64 px * 0.2 = 12.8 mm width; fan half-angle = atan(6.4/1000)
    assert g.value("fan_half_angle_deg") == pytest.approx(math.degrees(math.atan(6.4 / 1000)))


def test_missing_geometry_is_reported_not_defaulted(tmp_path):
    p = tmp_path / "a.xtekct"
    p.write_text("[XTekCT]\nName=x\n")
    g = io.build_scanner_geometry(io.parse_xtekct(p), C.GeometryConfig())
    assert not g.params["sod_mm"].known
    with pytest.raises(ValueError, match="sod_mm"):
        g.require("sod_mm")
    assert any(i.code == "geometry_missing" for i in io.validate_geometry(g))


def test_user_override_beats_metadata(tmp_path):
    p = tmp_path / "a.xtekct"
    p.write_text(XTEKCT)
    g = io.build_scanner_geometry(io.parse_xtekct(p), C.GeometryConfig(sod_mm=150.0))
    assert g.value("sod_mm") == 150.0 and g.params["sod_mm"].provenance is C.Provenance.USER


# --------------------------------- angles --------------------------------- #
def test_angle_table_3001_vs_3000_is_reported_not_trimmed(tmp_path):
    p = tmp_path / "_ctdata.txt"
    p.write_text(make_ctdata(n_rows=11, declared=10))
    t = io.read_angle_file(p)
    assert len(t.ids) == 11 and t.declared_projections == 10
    rep = io.validate_angle_table(t, expected_projections=10)
    assert len(t.ids) == 11                                                 # nothing removed
    assert sum(i.code in ("declared_count", "metadata_count") for i in rep.issues) == 2


def test_angle_duplicates_missing_invalid(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("Projection\tAngle(deg)\n1\t0.0\n2\t1.0\n2\t1.1\n4\t3.0\n5\tabc\n6\t5.0\n")
    t = io.read_angle_file(p)
    rep = io.validate_angle_table(t)
    assert rep.duplicate_ids == {2: 2}
    assert rep.missing_ids == [3, 5]
    assert len(t.invalid_rows) == 1
    codes = {i.code for i in rep.issues}
    assert {"duplicate_ids", "missing_ids", "invalid_rows"} <= codes


def test_wrap_and_decreasing_direction(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("Projection\tAngle(deg)\n1\t20.0\n2\t10.0\n3\t0.0\n4\t350.0\n5\t340.0\n")
    rep = io.validate_angle_table(io.read_angle_file(p))
    assert rep.direction == "decreasing" and rep.n_wrap_events == 1
    assert rep.unwrapped_deg[-1] == pytest.approx(-20.0)


def test_non_uniform_steps_flagged(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("Projection\tAngle(deg)\n1\t0\n2\t1\n3\t2\n4\t5\n5\t6\n")
    assert io.validate_angle_table(io.read_angle_file(p)).nonuniform_step_indices == [2]


def test_coverage_classification():
    full = np.arange(0, 360, 0.12)
    assert io.classify_angular_coverage(full, fan_half_angle_deg=10, n_detector_cols=2048).kind == "full_scan"
    short = np.arange(0, 200, 0.12)
    assert io.classify_angular_coverage(short, fan_half_angle_deg=10).kind == "short_scan"      # 200 >= 180+20
    assert io.classify_angular_coverage(np.arange(0, 150, 0.12), fan_half_angle_deg=10).kind == "incomplete"
    assert io.classify_angular_coverage(short).kind == "indeterminate"
    assert io.classify_angular_coverage(np.arange(0, 360, 5.0), n_detector_cols=2048).sparse


# -------------------------------- matching -------------------------------- #
def _write_tiffs(d: Path, ids, shape=(8, 10), dtype=np.uint16, fill=1000):
    d.mkdir(exist_ok=True)
    for i in ids:
        tifffile.imwrite(d / f"scan_{i:04d}.tif", np.full(shape, fill + i, dtype))


def test_matching_by_id_not_position(tmp_path):
    ct = tmp_path / "_ctdata.txt"
    ct.write_text(make_ctdata(n_rows=6, declared=6))
    _write_tiffs(tmp_path / "p", [1, 2, 4, 5, 6])                           # id 3 missing
    t = io.read_angle_file(ct)
    disc = io.discover_tiff_projections(tmp_path / "p")
    m = io.match_projections(disc.files_by_id, t)
    assert m.angles_without_image == [3] and not m.is_clean
    pairs = {x.angle_id: x.angle_deg for x in m.matched}
    assert pairs[4] == pytest.approx(108.0)                                 # NOT the angle of 3rd file
    with pytest.raises(io.ProjectionMatchError):
        io.require_match(m)


def test_offset_hint_is_not_applied(tmp_path):
    ct = tmp_path / "_ctdata.txt"
    ct.write_text(make_ctdata(n_rows=5, declared=5))
    _write_tiffs(tmp_path / "p", [0, 1, 2, 3, 4])                           # zero-based files
    m = io.match_projections(io.discover_tiff_projections(tmp_path / "p").files_by_id, io.read_angle_file(ct))
    assert m.offset_hint == 1 and len(m.matched) == 4                       # hint only; 4 of 5 matched at offset 0


def test_extra_images_reported(tmp_path):
    ct = tmp_path / "_ctdata.txt"
    ct.write_text(make_ctdata(n_rows=3, declared=3))
    _write_tiffs(tmp_path / "p", [1, 2, 3, 4])
    m = io.match_projections(io.discover_tiff_projections(tmp_path / "p").files_by_id, io.read_angle_file(ct))
    assert m.images_without_angle == [4]


def test_tiff_source_preserves_dtype_and_values(tmp_path):
    _write_tiffs(tmp_path / "p", [1, 2, 3], dtype=np.uint16, fill=65000)
    disc = io.discover_tiff_projections(tmp_path / "p")
    src = io.TiffSequenceSource([disc.files_by_id[i] for i in (1, 2, 3)], [1, 2, 3])
    assert src.dtype == np.uint16
    b = src.read_batch(0, 3)
    assert b.dtype == np.uint16 and b[2, 0, 0] == 65003 and src.verify_headers() == []


# ---------------------------------- DAT ----------------------------------- #
def test_dat_size_validation_and_read(tmp_path):
    arr = (np.arange(3 * 4 * 5) % 60000).astype("<u2").reshape(3, 4, 5)
    f = tmp_path / "s.dat"
    f.write_bytes(b"\0" * 16 + arr.tobytes())
    cfg = C.InputConfig(dat_path=str(f), dat_width=5, dat_height=4, dat_n_projections=3, dat_header_bytes=16)
    mm = io.open_dat_stack(cfg)
    assert mm.shape == (3, 4, 5) and np.array_equal(mm, arr)
    cfg.dat_header_bytes = 15
    with pytest.raises(io.ProjectionLoadError, match="size"):
        io.open_dat_stack(cfg)


def test_dat_big_endian(tmp_path):
    arr = np.arange(2 * 2 * 2, dtype=">u2").reshape(2, 2, 2)
    f = tmp_path / "s.dat"
    f.write_bytes(arr.tobytes())
    cfg = C.InputConfig(dat_path=str(f), dat_width=2, dat_height=2, dat_n_projections=2, dat_byte_order="big")
    assert np.array_equal(io.open_dat_stack(cfg), arr)


# ----------------------------- intensity / 16-bit -------------------------- #
def test_intensity_detects_12bit_in_16bit_container():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4096, (4, 64, 64)).astype(np.uint16)
    r = io.inspect_intensity(a)
    assert r.bits_needed_for_max == 12 and r.stored_bits == 16


def test_intensity_detects_stretched_data():
    rng = np.random.default_rng(0)
    a = (rng.integers(0, 4096, (4, 64, 64)) * 16).astype(np.uint16)
    assert io.inspect_intensity(a).level_spacing_gcd == 16


def test_intensity_saturation_and_whitelevel():
    a = np.full((2, 10, 10), 30000, np.uint16)
    a[0, :5] = 65535
    r = io.inspect_intensity(a, white_level=60000)
    assert r.fraction_at_dtype_max == pytest.approx(0.25)
    assert r.fraction_above_white_level == pytest.approx(0.25)


# ------------------------------- export ----------------------------------- #
def test_uint16_window_roundtrip_and_clipping():
    x = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0, np.nan]], np.float32)
    codes, lo, hi = io.to_uint16(x, 0.0, 1.0)
    assert codes.dtype == np.uint16 and codes.tolist() == [[0, 0, 32768, 65535, 65535, 0]]
    assert (lo, hi) == (2, 1)                                               # -1 and NaN low; 2 high
    back = 0.0 + codes[0, 2] * (1.0 - 0.0) / 65535
    assert abs(back - 0.5) <= 0.5 / 65535 + 1e-12                           # error <= half a level


def test_write_tiffs_and_vol(tmp_path):
    vol = np.random.default_rng(1).normal(0.5, 0.2, (5, 6, 7)).astype(np.float32)
    sp = (0.04, 0.04, 0.04)
    io.write_tiff_float32(vol, tmp_path / "f.tif", sp)
    assert np.array_equal(tifffile.imread(tmp_path / "f.tif"), vol)
    sc = io.write_tiff_uint16(vol, tmp_path / "u.tif", sp, 0.0, 1.0)
    u = tifffile.imread(tmp_path / "u.tif")
    assert u.dtype == np.uint16 and u.shape == vol.shape and sc.clipped_low_fraction > 0
    assert json.loads((tmp_path / "u.tif.json").read_text())["intensity"]["vmax"] == 1.0
    io.write_vol_vgi(vol, tmp_path / "v.vol", sp)
    raw = np.fromfile(tmp_path / "v.vol", dtype="<f4").reshape(5, 6, 7)
    assert np.array_equal(raw, vol)
    assert "Size = 7 6 5" in (tmp_path / "v.vgi").read_text()               # x y z


# ---------------------------- preprocessing ------------------------------- #
def _cfg(**kw):
    return C.PreprocessingConfig(**kw)


def test_flat_field_and_log_math():
    dark = np.full((4, 4), 100, np.float32)
    flat = np.full((4, 4), 1100, np.float32)
    raw = np.full((1, 4, 4), 100 + 1000 * math.exp(-0.5), np.float32)
    cal = pp.Calibration(dark=dark, flat=flat, denominator=flat - dark,
                         invalid_denominator=np.zeros((4, 4), bool))
    t = pp.normalise_batch(raw.astype(np.float32), cal, _cfg())
    assert t[0, 0, 0] == pytest.approx(math.exp(-0.5), rel=1e-5)
    P, lo, hi = pp.beer_lambert(t, 1e-6, None)
    assert P[0, 0, 0] == pytest.approx(0.5, rel=1e-4) and lo == 0


def test_raw_uint16_not_modified_and_exact_conversion():
    raw = np.full((1, 3, 3), 65535, np.uint16)
    cal = pp.Calibration(dark=np.zeros((3, 3), np.float32), flat=np.full((3, 3), 65535, np.float32),
                         denominator=np.full((3, 3), 65535, np.float32), invalid_denominator=np.zeros((3, 3), bool))
    t = pp.normalise_batch(raw, cal, _cfg())
    assert raw.dtype == np.uint16 and raw.max() == 65535 and t.dtype == np.float32 and t[0, 0, 0] == 1.0


def test_log_clamp_counts_and_no_nan():
    t = np.array([[[0.0, -0.1, 0.5, 2.0]]], np.float32)
    P, lo, hi = pp.beer_lambert(t.copy(), 1e-6, 1.0)
    assert lo == 2 and hi == 1 and np.all(np.isfinite(P)) and P[0, 0, 3] == pytest.approx(0.0)


def test_invalid_denominator_replaced_from_neighbours():
    dark = np.zeros((9, 9), np.float32)
    flat = np.full((9, 9), 1000.0, np.float32)
    flat[4, 4] = 0.0                                                         # dead pixel in flat
    cal = pp.Calibration(dark=dark, flat=flat, denominator=flat - dark,
                         invalid_denominator=(flat - dark) < 1.0)
    raw = np.full((2, 9, 9), 500, np.uint16)
    P, stats = pp.preprocess_batch(raw, cal, _cfg())
    assert np.allclose(P, -math.log(0.5), atol=1e-5) and stats.n_bad_pixels_replaced == 1


def test_replace_only_touches_masked_pixels():
    rng = np.random.default_rng(2)
    s = rng.random((2, 12, 12)).astype(np.float32)
    ref = s.copy()
    m = np.zeros((12, 12), bool)
    m[5, 5] = m[5, 6] = True
    assert pp.replace_masked_pixels(s, m) == 0
    assert np.array_equal(s[:, ~m], ref[:, ~m]) and np.all(s[:, m] != ref[:, m])


def test_detect_bad_pixels_finds_dead_and_hot():
    rng = np.random.default_rng(3)
    dark = 100 + rng.normal(0, 1, (40, 40)).astype(np.float32)
    flat = 1000 + rng.normal(0, 5, (40, 40)).astype(np.float32)
    flat[10, 10] = 100.0                                                     # dead (no response)
    dark[20, 20] += 500.0                                                    # hot
    mask, st = pp.detect_bad_pixels(dark, flat, _cfg())
    assert mask[10, 10] and mask[20, 20] and st["total"] < 10


def test_air_region_accepts_air_rejects_sample():
    t = np.ones((3, 20, 20), np.float32) * np.array([1.0, 0.99, 1.01], np.float32)[:, None, None]
    ok = pp.verify_air_region(t, (0, 5, 0, 5), 0.9, 0.02)
    assert ok.ok
    t[1, :6, :6] = 0.5                                                       # sample enters the region
    bad = pp.verify_air_region(t, (0, 5, 0, 5), 0.9, 0.02)
    assert not bad.ok
    cfg = _cfg(air_normalisation=True, air_region=(0, 5, 0, 5))
    cal = pp.Calibration(i0=1.0)
    cfg.normalisation_mode = "scalar_i0"
    with pytest.raises(pp.PreprocessingError, match="rejected"):
        pp.preprocess_batch((t * 1000).astype(np.float32), pp.Calibration(i0=1000.0), cfg)


def test_air_normalisation_removes_flux_drift():
    base = np.full((3, 20, 20), 0.8, np.float32)
    base[:, :, :5] = 1.0                                                     # air strip
    drift = np.array([1.0, 0.97, 1.03], np.float32)[:, None, None]
    cfg = _cfg(normalisation_mode="scalar_i0", i0_value=1.0, air_normalisation=True, air_region=(0, 5, 0, 20))
    P, st = pp.preprocess_batch(base * drift, pp.Calibration(i0=1.0), cfg)
    assert np.allclose(P[:, 5, 10], -math.log(0.8), atol=1e-5)


def test_stack_memory_spill_requires_work_dir(tmp_path):
    _write_tiffs(tmp_path / "p", [1, 2], shape=(16, 16))
    d = io.discover_tiff_projections(tmp_path / "p")
    src = io.TiffSequenceSource([d.files_by_id[1], d.files_by_id[2]], [1, 2])
    cal = pp.Calibration(i0=2000.0)
    cfg = _cfg(normalisation_mode="scalar_i0", i0_value=2000.0, memory_limit_gb=1e-9)
    with pytest.raises(pp.PreprocessingError, match="work_dir"):
        pp.preprocess_stack(src, cal, cfg)
    cfg.work_dir = str(tmp_path / "w")
    out, stats, info = pp.preprocess_stack(src, cal, cfg)
    assert not info["in_ram"] and out.shape == (2, 16, 16) and stats.n_projections == 2
