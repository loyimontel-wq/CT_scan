"""io_xtek.py - reading Nikon/X-Tek data and writing volumes.

PROBLEM THIS MODULE SOLVES
    Turn files on disk into *trustworthy, labelled* in-memory objects:
      * .xtekct metadata            -> XtekMetadata  -> ScannerGeometry (with provenance)
      * _ctdata.txt / .ang          -> AngleTable    -> AngleReport / CoverageReport
      * TIFF folder or raw DAT      -> ProjectionSource (one interface for both)
      * image IDs + angle IDs       -> MatchResult   (explicit, never positional)
      * pixel statistics            -> IntensityReport (is the 16-bit data really 16-bit?)
      * reconstructed volume        -> float32 TIFF, 16-bit TIFF (documented window), .vol/.vgi

PHYSICAL PICTURE (needed for the geometry maths below)
    Source S, rotation axis through the object, flat detector D. SOD = |S-axis|,
    SDD = |S-D|. A detector pixel of pitch p images an object-plane length
    p/M where M = SDD/SOD (magnification). Hence the *geometric voxel size at the
    rotation axis* is  voxel = p * SOD / SDD.  The metadata contains an independent
    VoxelSize, so we can check the geometry for self-consistency.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

from industrial_xct.config import (GeometryConfig, InputConfig, Issue,
                                   ParameterValue, Provenance)

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


class MetadataError(ValueError):
    """Malformed or unusable scanner metadata."""


class AngleFileError(ValueError):
    """Malformed or unusable angle file."""


class ProjectionLoadError(RuntimeError):
    """A projection image could not be read or is inconsistent."""


class ProjectionMatchError(RuntimeError):
    """Projection identifiers and angle identifiers do not correspond."""


class ExportError(RuntimeError):
    """A volume could not be exported."""


# =========================================================================== #
# A. Metadata (.xtekct)
# =========================================================================== #
# Keys this package consumes. Everything else is preserved verbatim, never dropped.
CONSUMED_KEYS = (
    "SrcToObject", "SrcToDetector", "DetectorPixelsX", "DetectorPixelsY",
    "DetectorPixelSizeX", "DetectorPixelSizeY", "DetectorOffsetX", "DetectorOffsetY",
    "Projections", "InitialAngle", "AngularStep", "CentreOfRotationTop",
    "CentreOfRotationBottom", "WhiteLevel", "Scattering", "VoxelsX", "VoxelsY",
    "VoxelsZ", "VoxelSizeX", "VoxelSizeY", "VoxelSizeZ", "RegionStartX",
    "RegionStartY", "RegionPixelsX", "RegionPixelsY", "Units", "XraykV", "XrayuA",
)


@dataclass
class XtekMetadata:
    """Parsed .xtekct: ``sections[section][key] = raw string`` (nothing discarded)."""
    path: Path
    sections: Dict[str, Dict[str, str]]
    warnings: List[str] = field(default_factory=list)

    def get(self, key: str, section: str = "XTekCT") -> Optional[str]:
        return self.sections.get(section, {}).get(key)

    def get_float(self, key: str, section: str = "XTekCT") -> Optional[float]:
        raw = self.get(key, section)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except ValueError as exc:
            raise MetadataError(f"{self.path.name}: [{section}] {key}={raw!r} is not a number") from exc

    def get_int(self, key: str, section: str = "XTekCT") -> Optional[int]:
        v = self.get_float(key, section)
        if v is None:
            return None
        if abs(v - round(v)) > 1e-9:
            raise MetadataError(f"{self.path.name}: [{section}] {key}={v} is not an integer")
        return int(round(v))

    @property
    def length_unit(self) -> Optional[str]:
        """The file's own 'Units' entry (evidence for the unit of lengths), else None."""
        return self.get("Units") or None

    def unconsumed_keys(self) -> Dict[str, Dict[str, str]]:
        """Fields present in the file that the pipeline does not interpret."""
        out: Dict[str, Dict[str, str]] = {}
        for sec, kv in self.sections.items():
            rest = {k: v for k, v in kv.items() if not (sec == "XTekCT" and k in CONSUMED_KEYS)
                    and not (sec == "Xrays" and k in CONSUMED_KEYS)}
            if rest:
                out[sec] = rest
        return out


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeError:
            continue
    raise MetadataError(f"Cannot decode {path}")  # latin-1 never fails; defensive


def parse_xtekct(path: PathLike) -> XtekMetadata:
    """Parse an INI-style .xtekct file. Duplicate keys and malformed lines are recorded."""
    p = Path(path)
    if not p.is_file():
        raise MetadataError(f"Metadata file not found: {p}")
    sections: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    current: Optional[str] = None
    for lineno, raw in enumerate(_read_text(p).splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" not in line:
            warnings.append(f"line {lineno}: ignored (no '='): {line!r}")
            continue
        if current is None:
            raise MetadataError(f"{p.name} line {lineno}: key before any [section]")
        key, value = (s.strip() for s in line.split("=", 1))
        if key in sections[current]:
            warnings.append(f"line {lineno}: duplicate key [{current}] {key}; last value kept")
        sections[current][key] = value
    if "XTekCT" not in sections:
        raise MetadataError(f"{p.name}: no [XTekCT] section; not a Nikon/X-Tek metadata file?")
    return XtekMetadata(p, sections, warnings)


# --------------------------------------------------------------------------- #
# Scanner geometry with provenance
# --------------------------------------------------------------------------- #
@dataclass
class ScannerGeometry:
    """Every geometric quantity as a ParameterValue (value + provenance + caveat)."""
    params: Dict[str, ParameterValue]

    def value(self, name: str) -> Any:
        return self.params[name].value

    def require(self, *names: str) -> Dict[str, Any]:
        missing = [n for n in names if not self.params[n].known]
        if missing:
            raise ValueError(
                "Required geometry parameter(s) unavailable: " +
                ", ".join(f"{n} ({self.params[n].note or 'no source'})" for n in missing))
        return {n: self.params[n].value for n in names}

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in self.params.items()}


REQUIRED_FOR_RECONSTRUCTION = ("sod_mm", "sdd_mm", "detector_pixels_x", "detector_pixels_y",
                               "pixel_size_x_mm", "pixel_size_y_mm")


def _resolve(meta: XtekMetadata, key: str, override: Optional[float], unit: Optional[str],
             cast: str, note: str = "") -> ParameterValue:
    """override (USER) > metadata (METADATA) > UNAVAILABLE."""
    if override is not None:
        return ParameterValue(override, Provenance.USER, unit, "config", note)
    val = meta.get_int(key) if cast == "int" else meta.get_float(key)
    if val is None:
        return ParameterValue(None, Provenance.UNAVAILABLE, unit, "",
                              f"'{key}' absent from {meta.path.name}; supply it in the config")
    return ParameterValue(val, Provenance.METADATA, unit, f"{meta.path.name}:{key}", note)


def build_scanner_geometry(meta: XtekMetadata, cfg: GeometryConfig) -> ScannerGeometry:
    """Combine metadata with user overrides; derive secondary quantities.

    Derived quantities (CALCULATED):
        magnification M      = SDD / SOD
        voxel size at axis   = pixel_pitch_x / M = pitch * SOD / SDD
        detector width/height= n_pixels * pitch
        fan half-angle       = atan( (W/2 + |offset_x|) / SDD )
    """
    unit = meta.length_unit
    lu = unit or None
    unit_note = "" if unit else "no 'Units' entry in metadata; length unit unverified"
    P: Dict[str, ParameterValue] = {}
    P["sod_mm"] = _resolve(meta, "SrcToObject", cfg.sod_mm, lu, "float", unit_note)
    P["sdd_mm"] = _resolve(meta, "SrcToDetector", cfg.sdd_mm, lu, "float", unit_note)
    P["detector_pixels_x"] = _resolve(meta, "DetectorPixelsX", cfg.detector_pixels_x, "px", "int")
    P["detector_pixels_y"] = _resolve(meta, "DetectorPixelsY", cfg.detector_pixels_y, "px", "int")
    P["pixel_size_x_mm"] = _resolve(meta, "DetectorPixelSizeX", cfg.pixel_size_x_mm, lu, "float", unit_note)
    P["pixel_size_y_mm"] = _resolve(meta, "DetectorPixelSizeY", cfg.pixel_size_y_mm, lu, "float", unit_note)
    off_note = ("unit and sign convention unverified; 0.0 may mean 'not calibrated' "
                "rather than a measured zero")
    P["detector_offset_x_mm"] = _resolve(meta, "DetectorOffsetX", cfg.detector_offset_x_mm, None, "float", off_note)
    P["detector_offset_y_mm"] = _resolve(meta, "DetectorOffsetY", cfg.detector_offset_y_mm, None, "float", off_note)
    # Raw CoR fields: reported, NOT used unless cor_source == 'metadata' (Pass 2).
    cor_note = "raw metadata value; unit/sign unverified; unused unless cor_source='metadata'"
    P["cor_top_raw"] = _resolve(meta, "CentreOfRotationTop", None, None, "float", cor_note)
    P["cor_bottom_raw"] = _resolve(meta, "CentreOfRotationBottom", None, None, "float", cor_note)
    P["region_start_x"] = _resolve(meta, "RegionStartX", None, "px", "int")
    P["region_start_y"] = _resolve(meta, "RegionStartY", None, "px", "int")
    P["region_pixels_x"] = _resolve(meta, "RegionPixelsX", None, "px", "int")
    P["region_pixels_y"] = _resolve(meta, "RegionPixelsY", None, "px", "int")
    P["voxel_size_metadata_mm"] = _resolve(meta, "VoxelSizeX", cfg.voxel_size_mm, lu, "float", unit_note)
    for ax, key in zip("xyz", ("VoxelsX", "VoxelsY", "VoxelsZ")):
        v = None if cfg.volume_voxels is None else cfg.volume_voxels["xyz".index(ax)]
        P[f"volume_voxels_{ax}"] = _resolve(meta, key, v, "vox", "int")
    P["white_level_raw"] = _resolve(meta, "WhiteLevel", None, None, "float",
                                    "meaning unverified: do not treat as I0 until checked against the data")

    # Misalignment angles: never silently defaulted to zero without saying so.
    for name, val in (("rotation_axis_tilt_deg", cfg.rotation_axis_tilt_deg),
                      ("detector_roll_deg", cfg.detector_roll_deg),
                      ("detector_tilt_deg", cfg.detector_tilt_deg)):
        if val is not None:
            P[name] = ParameterValue(val, Provenance.USER, "deg", "config")
        else:
            P[name] = ParameterValue(0.0, Provenance.ASSUMED, "deg", "",
                                     "assumed ideal (0) because no calibration is available")
    # CoR state: exactly one source (validated in config).
    if cfg.cor_source == "manual":
        P["cor_offset_px"] = ParameterValue(cfg.cor_manual_px, Provenance.USER, "px", "config",
                                            "sign convention to be fixed in Pass 2 (ASTRA)")
    elif cfg.cor_source == "none":
        P["cor_offset_px"] = ParameterValue(0.0, Provenance.ASSUMED, "px", "",
                                            "rotation axis assumed at the (offset-corrected) detector centre")
    else:
        P["cor_offset_px"] = ParameterValue(None, Provenance.UNAVAILABLE, "px", "",
                                            f"cor_source='{cfg.cor_source}' is resolved in Pass 2")

    # Derived quantities.
    sod, sdd = P["sod_mm"], P["sdd_mm"]
    if sod.known and sdd.known:
        M = sdd.value / sod.value
        P["magnification"] = ParameterValue(M, Provenance.CALCULATED, None, "SDD/SOD")
        px = P["pixel_size_x_mm"]
        if px.known:
            P["voxel_size_from_geometry_mm"] = ParameterValue(
                px.value / M, Provenance.CALCULATED, lu, "pixel_pitch_x*SOD/SDD")
    for ax in "xy":
        n, p = P[f"detector_pixels_{ax}"], P[f"pixel_size_{ax}_mm"]
        P[f"detector_size_{ax}_mm"] = (ParameterValue(n.value * p.value, Provenance.CALCULATED, lu, "n_pixels*pitch")
                                       if n.known and p.known else
                                       ParameterValue(None, Provenance.UNAVAILABLE, lu, "", "needs pixels and pitch"))
    w, off = P["detector_size_x_mm"], P["detector_offset_x_mm"]
    if w.known and sdd.known and off.known:
        half = math.degrees(math.atan((w.value / 2 + abs(off.value)) / sdd.value))
        P["fan_half_angle_deg"] = ParameterValue(half, Provenance.CALCULATED, "deg",
                                                 "atan((W/2+|offset|)/SDD)",
                                                 "uses detector offset as given; its unit is unverified")
    else:
        P["fan_half_angle_deg"] = ParameterValue(None, Provenance.UNAVAILABLE, "deg", "",
                                                 "needs detector width, SDD and offset")
    return ScannerGeometry(P)


def validate_geometry(g: ScannerGeometry, rel_tol: float = 0.01) -> List[Issue]:
    """Self-consistency checks. Passing them does NOT prove the geometry is right."""
    I: List[Issue] = []
    for name in REQUIRED_FOR_RECONSTRUCTION:
        if not g.params[name].known:
            I.append(Issue("error", "geometry_missing", f"{name}: {g.params[name].note}"))
    if g.params["sod_mm"].known and g.params["sdd_mm"].known:
        if g.value("sdd_mm") <= g.value("sod_mm"):
            I.append(Issue("error", "sdd_le_sod", "SDD must exceed SOD"))
    vg, vm = g.params.get("voxel_size_from_geometry_mm"), g.params["voxel_size_metadata_mm"]
    if vg is not None and vg.known and vm.known:
        rel = abs(vg.value - vm.value) / vm.value
        sev = "info" if rel <= rel_tol else "warning"
        I.append(Issue(sev, "voxel_consistency",
                       f"pitch*SOD/SDD = {vg.value:.6g} vs metadata VoxelSize = {vm.value:.6g} "
                       f"(relative difference {rel:.2e})"))
    px, py = g.params["pixel_size_x_mm"], g.params["pixel_size_y_mm"]
    if px.known and py.known and abs(px.value - py.value) / px.value > 1e-6:
        I.append(Issue("warning", "anisotropic_pixels", "Detector pixels are not square"))
    for name in ("detector_offset_x_mm", "detector_offset_y_mm", "cor_top_raw", "cor_bottom_raw"):
        p = g.params[name]
        if p.known and p.value == 0.0:
            I.append(Issue("info", "zero_uncalibrated",
                           f"{name} = 0.0 in metadata: cannot distinguish 'measured zero' from "
                           f"'not calibrated'. An image-based CoR check is required."))
    rx, ry = g.params["region_pixels_x"], g.params["region_pixels_y"]
    dx, dy = g.params["detector_pixels_x"], g.params["detector_pixels_y"]
    if rx.known and dx.known and (rx.value != dx.value or (ry.known and dy.known and ry.value != dy.value)):
        I.append(Issue("warning", "region_of_interest",
                       f"Acquired region {rx.value}x{ry.value} differs from detector "
                       f"{dx.value}x{dy.value}: the detector centre shifts; handle in geometry."))
    return I


def expected_image_shape(g: ScannerGeometry) -> Optional[Tuple[int, int]]:
    """(rows, cols) the projection images should have: acquired region if stated, else detector."""
    ry, rx = g.params["region_pixels_y"], g.params["region_pixels_x"]
    if ry.known and rx.known:
        return int(ry.value), int(rx.value)
    dy, dx = g.params["detector_pixels_y"], g.params["detector_pixels_x"]
    if dy.known and dx.known:
        return int(dy.value), int(dx.value)
    return None


# =========================================================================== #
# B. Angle files
# =========================================================================== #
@dataclass
class AngleTable:
    """Rows of the angle file, in file order, exactly as read (angles NOT unwrapped)."""
    source: Path
    file_format: str
    ids: np.ndarray                         # int64 projection identifiers
    angles_deg: np.ndarray                  # float64 as stored
    extra: Dict[str, np.ndarray]            # e.g. {"Time(s)": ...}
    ids_are_positional: bool                # True when the file had no identifier column
    declared_projections: Optional[int] = None
    declared_frames: Optional[int] = None
    exposure_ms: Optional[float] = None
    invalid_rows: List[Tuple[int, str, str]] = field(default_factory=list)  # (line, text, reason)


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_angle_file(path: PathLike) -> AngleTable:
    """Read a Nikon ``_ctdata.txt`` (headed blocks) or a plain ``.ang`` (rows only).

    Rules: a line with no numeric token is a header. The block whose header mentions
    'angle' is the angle table. A header containing 'Projections' and 'Frames' gives the
    declared counts. For headerless files: column 1 = identifier, column 2 = angle (deg),
    further columns kept as extras; a single column is angle-only (identifiers positional,
    flagged). NOTE: the plain '.ang' layout was not verified against a real sample here;
    anything that does not fit these rules is reported as an invalid row, not guessed.
    """
    p = Path(path)
    if not p.is_file():
        raise AngleFileError(f"Angle file not found: {p}")
    blocks: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {"header": None, "rows": []}
    for lineno, line in enumerate(_read_text(p).splitlines(), 1):
        toks = [t for t in re.split(r"[\s,;]+", line.strip()) if t]
        if not toks:
            continue
        if not any(_is_number(t) for t in toks):          # header line
            if cur["rows"] or cur["header"] is not None:
                blocks.append(cur)
            cur = {"header": toks, "rows": []}
        else:
            cur["rows"].append((lineno, line.strip(), toks))
    blocks.append(cur)

    angle_block = next((b for b in blocks if b["header"] and
                        any("angle" in t.lower() for t in b["header"])), None)
    declared_n = frames = exposure = None
    for b in blocks:
        h = [t.lower() for t in (b["header"] or [])]
        if h and any(t.startswith("projections") for t in h) and not any("angle" in t for t in h) and b["rows"]:
            toks = b["rows"][0][2]
            try:
                declared_n = int(float(toks[0]))
                frames = int(float(toks[1])) if len(toks) > 1 else None
                exposure = float(toks[2]) if len(toks) > 2 else None
            except ValueError:
                pass
    fmt = "ctdata"
    if angle_block is None:
        headerless = [b for b in blocks if b["header"] is None and b["rows"]]
        if not headerless:
            raise AngleFileError(f"{p.name}: no angle table found")
        angle_block, fmt = max(headerless, key=lambda b: len(b["rows"])), "headerless"

    header = angle_block["header"]
    if header:
        low = [t.lower() for t in header]
        a_idx = next(i for i, t in enumerate(low) if "angle" in t)
        i_idx = next((i for i, t in enumerate(low) if t.startswith("projection") or t in ("id", "index", "frame")), None)
        extra_names = {i: header[i] for i in range(len(header)) if i not in (a_idx, i_idx)}
    else:
        ncol = max(len(r[2]) for r in angle_block["rows"])
        i_idx, a_idx = (0, 1) if ncol >= 2 else (None, 0)
        extra_names = {i: f"col{i + 1}" for i in range(ncol) if i not in (i_idx, a_idx)}

    ids: List[int] = []
    angles: List[float] = []
    extras: Dict[str, List[float]] = {n: [] for n in extra_names.values()}
    invalid: List[Tuple[int, str, str]] = []
    for lineno, text, toks in angle_block["rows"]:
        need = max(a_idx, i_idx if i_idx is not None else 0) + 1
        if len(toks) < need:
            invalid.append((lineno, text, f"expected >= {need} columns, found {len(toks)}"))
            continue
        try:
            a = float(toks[a_idx])
            if not math.isfinite(a):
                raise ValueError("non-finite angle")
            if i_idx is not None:
                fid = float(toks[i_idx])
                if abs(fid - round(fid)) > 1e-9:
                    raise ValueError("identifier is not an integer")
                pid = int(round(fid))
            else:
                pid = len(ids) + 1
        except ValueError as exc:
            invalid.append((lineno, text, str(exc)))
            continue
        ids.append(pid)
        angles.append(a)
        for i, n in extra_names.items():
            try:
                extras[n].append(float(toks[i]))
            except (ValueError, IndexError):
                extras[n].append(float("nan"))
    if not ids:
        raise AngleFileError(f"{p.name}: angle table contains no valid rows ({len(invalid)} invalid)")
    return AngleTable(p, fmt, np.array(ids, dtype=np.int64), np.array(angles, dtype=np.float64),
                      {k: np.array(v) for k, v in extras.items()}, i_idx is None,
                      declared_n, frames, exposure, invalid)


@dataclass
class AngleReport:
    n_entries: int
    id_min: int
    id_max: int
    duplicate_ids: Dict[int, int]
    missing_ids: List[int]
    ids_monotonic: bool
    direction: str                          # increasing | decreasing | mixed
    unwrapped_deg: np.ndarray               # table order, continuous angle
    n_wrap_events: int
    step_median_deg: float
    step_min_deg: float
    step_max_deg: float
    step_std_deg: float
    nonuniform_step_indices: List[int]      # step i lies between entries i and i+1
    issues: List[Issue]


def validate_angle_table(t: AngleTable, *, expected_projections: Optional[int] = None,
                         uniform_rel_tol: float = 0.25) -> AngleReport:
    """Check identifiers and angular sampling. Nothing is deleted or reordered.

    Angles are unwrapped with period 360 deg (a jump larger than 180 deg is treated as a
    wrap). 'Non-uniform' means |step - median| > uniform_rel_tol * |median|.
    """
    I: List[Issue] = []
    n = len(t.ids)
    uniq, counts = np.unique(t.ids, return_counts=True)
    dups = {int(i): int(c) for i, c in zip(uniq, counts) if c > 1}
    if dups:
        I.append(Issue("error", "duplicate_ids", f"{len(dups)} duplicated projection id(s), e.g. {list(dups)[:5]}"))
    full = set(range(int(uniq[0]), int(uniq[-1]) + 1))
    missing = sorted(full - set(uniq.tolist()))
    if missing:
        I.append(Issue("warning", "missing_ids", f"{len(missing)} id(s) missing inside "
                                                  f"[{uniq[0]}, {uniq[-1]}], e.g. {missing[:5]}"))
    monotonic = bool(np.all(np.diff(t.ids) > 0))
    if not monotonic:
        I.append(Issue("warning", "ids_not_monotonic", "Identifiers are not strictly increasing in file order"))
    if t.invalid_rows:
        I.append(Issue("warning", "invalid_rows", f"{len(t.invalid_rows)} invalid row(s) skipped; first: "
                                                   f"line {t.invalid_rows[0][0]}: {t.invalid_rows[0][2]}"))
    if t.declared_projections is not None and t.declared_projections != n:
        I.append(Issue("warning", "declared_count",
                       f"{t.source.name} header declares {t.declared_projections} projections but the "
                       f"table has {n} valid entries (difference {n - t.declared_projections:+d}). "
                       f"No entry has been removed."))
    if expected_projections is not None and expected_projections != n:
        I.append(Issue("warning", "metadata_count",
                       f"Metadata expects {expected_projections} projections; angle table has {n} "
                       f"(difference {n - expected_projections:+d}). No entry has been removed."))
    if t.ids_are_positional:
        I.append(Issue("warning", "positional_ids", "Angle file has no identifier column; ids are row numbers. "
                                                    "Pairing with images is only as reliable as that assumption."))

    unw = np.unwrap(t.angles_deg, period=360.0) if n > 1 else t.angles_deg.copy()
    raw_steps = np.diff(t.angles_deg)
    steps = np.diff(unw)
    n_wrap = int(np.sum(~np.isclose(raw_steps, steps)))
    if n_wrap:
        I.append(Issue("info", "angular_wrap", f"{n_wrap} angular wrap event(s) unwrapped (period 360 deg)"))
    if n < 2:
        raise AngleFileError("At least two angle entries are required")
    med = float(np.median(steps))
    if med == 0:
        direction = "mixed"
    else:
        direction = "increasing" if med > 0 else "decreasing"
    if np.any(np.sign(steps) != np.sign(med)):
        direction = "mixed"
        I.append(Issue("error", "mixed_direction", "Angle steps change sign: not a monotonic rotation"))
    nonuni = np.nonzero(np.abs(steps - med) > uniform_rel_tol * abs(med))[0].tolist() if med else []
    if nonuni:
        I.append(Issue("warning", "nonuniform_steps",
                       f"{len(nonuni)} step(s) deviate > {uniform_rel_tol:.0%} from the median step "
                       f"{med:.5f} deg (range {steps.min():.5f} to {steps.max():.5f})"))
    return AngleReport(n, int(uniq[0]), int(uniq[-1]), dups, missing, monotonic, direction, unw, n_wrap,
                       med, float(steps.min()), float(steps.max()), float(steps.std()), nonuni, I)


@dataclass
class CoverageReport:
    kind: str                       # full_scan | short_scan | incomplete | indeterminate
    span_deg: float                 # |last - first|
    closure_gap_deg: float          # 360 - (span + median step); small => closes the circle
    closure_gap_steps: float
    min_short_scan_deg: Optional[float]
    sampling_ratio: Optional[float] # n_proj / (pi/2 * n_cols); ~1 is the classic rule of thumb
    sparse: bool
    notes: List[str]


def classify_angular_coverage(unwrapped_deg: np.ndarray, *, fan_half_angle_deg: Optional[float] = None,
                              n_detector_cols: Optional[int] = None,
                              full_scan_tolerance_deg: float = 1.0) -> CoverageReport:
    """Decide full / short / incomplete BEFORE choosing an algorithm.

    * span      = |a_last - a_first|; each projection 'owns' one step, so the sampled arc is
                  span + median_step. closure_gap = 360 - that arc.
    * full_scan : closure_gap <= tolerance (an arc >360 is an over-scan; redundant data).
    * Parker short scan needs >= 180 + 2*fan_half_angle degrees (pi + fan angle).
      Plain FDK on a short scan without Parker weighting is NOT equivalent to a full scan.
    * sampling ratio: classic rule N >= (pi/2)*n_cols for a full-width object; a ratio well
      below 1 means streaking from angular under-sampling is likely.
    """
    steps = np.abs(np.diff(unwrapped_deg))
    med = float(np.median(steps))
    span = float(abs(unwrapped_deg[-1] - unwrapped_deg[0]))
    arc = span + med
    gap = 360.0 - arc
    notes: List[str] = []
    min_short = None if fan_half_angle_deg is None else 180.0 + 2.0 * fan_half_angle_deg
    if gap <= full_scan_tolerance_deg:
        kind = "full_scan"
        if gap < -full_scan_tolerance_deg:
            notes.append(f"Arc exceeds 360 deg by {-gap:.3f} deg: redundant over-scan.")
        notes.append(f"Missing arc {gap:.3f} deg ({gap / med:.2f} steps), within tolerance "
                     f"{full_scan_tolerance_deg} deg; treated as a full scan. This is a tolerance "
                     f"judgement, not a proof of closure.")
    elif min_short is None:
        kind = "indeterminate"
        notes.append("Arc < 360 deg and fan angle unknown: cannot decide between short-scan and incomplete.")
    elif arc >= min_short:
        kind = "short_scan"
        notes.append(f"Arc {arc:.2f} deg >= 180 + fan angle = {min_short:.2f} deg. Needs Parker (short-scan) "
                     f"weighting; plain FDK is not valid.")
    else:
        kind = "incomplete"
        notes.append(f"Arc {arc:.2f} deg < {min_short:.2f} deg required for exact short-scan data.")
    ratio = None if n_detector_cols is None else len(unwrapped_deg) / (math.pi / 2 * n_detector_cols)
    sparse = ratio is not None and ratio < 0.5
    if ratio is not None:
        notes.append(f"Sampling ratio N/((pi/2)*n_cols) = {ratio:.2f}"
                     + (" (sparse-angle regime)" if sparse else
                        " (below 1: mild angular under-sampling if the object fills the detector width)"
                        if ratio < 1 else ""))
    return CoverageReport(kind, span, gap, gap / med, min_short, ratio, sparse, notes)


def compare_metadata_to_angles(meta: XtekMetadata, t: AngleTable, rep: AngleReport) -> List[Issue]:
    """Compare header values with the measured angle table; report, never 'fix'."""
    I: List[Issue] = []
    n_meta, step_meta, init = meta.get_int("Projections"), meta.get_float("AngularStep"), meta.get_float("InitialAngle")
    if step_meta is not None:
        rel = abs(abs(step_meta) - abs(rep.step_median_deg)) / abs(step_meta)
        I.append(Issue("info" if rel < 0.01 else "warning", "angular_step",
                       f"AngularStep={step_meta:.6f} vs measured median step {rep.step_median_deg:.6f} deg "
                       f"(relative difference {rel:.2e})"))
    if init is not None:
        d = float(t.angles_deg[0]) - init
        I.append(Issue("info" if abs(d) < 0.5 else "warning", "initial_angle",
                       f"InitialAngle={init:.4f} deg vs first angle-table entry {t.angles_deg[0]:.4f} deg "
                       f"(difference {d:+.4f} deg). The meaning of InitialAngle is unverified; the angle "
                       f"table is used for pairing. A constant offset only rotates the volume about the "
                       f"axis; it matters for orientation comparison with Nikon CT Pro 3D."))
    if n_meta is not None and step_meta is not None:
        I.append(Issue("info", "header_sweep", f"Projections*AngularStep = {n_meta * step_meta:.3f} deg"))
    return I


# =========================================================================== #
# C/D. Projection discovery, matching, loading
# =========================================================================== #
@dataclass
class DiscoveryResult:
    directory: Path
    files_by_id: Dict[int, Path]
    duplicate_ids: Dict[int, List[Path]]
    unparsed_files: List[Path]


def discover_tiff_projections(directory: PathLike, glob: str = "*.tif*",
                              id_regex: str = r"(\d+)\.tiff?$") -> DiscoveryResult:
    """Find TIFFs and extract an integer identifier from each file name (regex group 1)."""
    d = Path(directory)
    if not d.is_dir():
        raise ProjectionLoadError(f"Projection directory not found: {d}")
    rx = re.compile(id_regex, re.IGNORECASE)
    if rx.groups < 1:
        raise ProjectionLoadError("projection_id_regex needs a capture group for the identifier")
    by_id: Dict[int, List[Path]] = {}
    unparsed: List[Path] = []
    for f in sorted(d.glob(glob)):
        if not f.is_file():
            continue
        m = rx.search(f.name)
        if not m:
            unparsed.append(f)
            continue
        by_id.setdefault(int(m.group(1)), []).append(f)
    dups = {k: v for k, v in by_id.items() if len(v) > 1}
    files = {k: v[0] for k, v in by_id.items() if len(v) == 1}
    return DiscoveryResult(d, files, dups, unparsed)


@dataclass
class MatchedProjection:
    angle_id: int                   # identifier in the angle table (after id_offset)
    file_id: int                    # identifier as parsed from the file name / DAT position
    path: Optional[Path]            # None for DAT stacks
    angle_deg: float                # as stored in the angle table
    table_index: int                # row in the angle table


@dataclass
class MatchResult:
    matched: List[MatchedProjection]          # in angle-table (acquisition) order
    images_without_angle: List[int]           # image ids (after offset) with no angle row
    angles_without_image: List[int]           # angle ids with no image
    ambiguous_ids: List[int]                  # duplicated ids: excluded from matching, NOT deleted
    offset_hint: Optional[int]
    issues: List[Issue]

    @property
    def is_clean(self) -> bool:
        return not (self.images_without_angle or self.angles_without_image or self.ambiguous_ids)


def match_projections(image_ids: Dict[int, Optional[Path]], table: AngleTable, *,
                      id_offset: int = 0,
                      duplicate_image_ids: Optional[Sequence[int]] = None) -> MatchResult:
    """Pair images with angles by IDENTIFIER (never by sorted position).

    Unmatched or duplicated identifiers are reported and kept out of ``matched``; the angle
    table itself is never modified. If a constant id shift would increase the number of
    matches, a hint is produced but NOT applied.
    """
    I: List[Issue] = []
    shifted = {fid + id_offset: (fid, p) for fid, p in image_ids.items()}
    uniq, counts = np.unique(table.ids, return_counts=True)
    angle_dups = {int(i) for i, c in zip(uniq, counts) if c > 1}
    dup_img = set(int(x) + id_offset for x in (duplicate_image_ids or []))
    ambiguous = sorted(angle_dups | dup_img)
    matched: List[MatchedProjection] = []
    for idx, (aid, ang) in enumerate(zip(table.ids.tolist(), table.angles_deg.tolist())):
        if aid in angle_dups or aid in dup_img or aid not in shifted:
            continue
        fid, path = shifted[aid]
        matched.append(MatchedProjection(aid, fid, path, ang, idx))
    angle_set = set(table.ids.tolist())
    img_wo = sorted(k for k in shifted if k not in angle_set and k not in dup_img)
    ang_wo = sorted(angle_set - set(shifted) - angle_dups)
    hint = None
    if shifted and (img_wo or ang_wo):
        d = int(min(angle_set) - min(shifted))
        if d != 0 and len({k + d for k in shifted} & angle_set) > len(set(shifted) & angle_set):
            hint = id_offset + d
            I.append(Issue("warning", "offset_hint", f"Shifting image ids by {d:+d} (id_offset={hint}) would "
                                                      f"match more projections. NOT applied; verify first."))
    if img_wo:
        I.append(Issue("error", "image_without_angle", f"{len(img_wo)} image(s) have no angle entry, e.g. {img_wo[:5]}"))
    if ang_wo:
        I.append(Issue("warning", "angle_without_image",
                       f"{len(ang_wo)} angle entr(ies) have no image, e.g. {ang_wo[:5]}. Entries kept in "
                       f"the table; they are not used for reconstruction."))
    if ambiguous:
        I.append(Issue("error", "ambiguous_ids", f"Duplicated identifier(s) excluded from matching: {ambiguous[:5]}"))
    return MatchResult(matched, img_wo, ang_wo, ambiguous, hint, I)


def require_match(m: MatchResult, *, allow_partial: bool = False) -> None:
    """Raise unless every image and angle entry is matched (or partial matching is explicitly allowed)."""
    if not m.matched:
        raise ProjectionMatchError("No projection could be matched to an angle:\n  " +
                                   "\n  ".join(str(i) for i in m.issues))
    if not allow_partial and not m.is_clean:
        raise ProjectionMatchError(
            "Projection/angle matching is not clean (set input.require_complete_match=false only after "
            "reviewing these):\n  " + "\n  ".join(str(i) for i in m.issues))


@dataclass
class TiffInfo:
    path: Path
    shape: Tuple[int, ...]
    dtype: str
    bits_per_sample: int
    n_pages: int
    compression: str
    photometric: str


def inspect_tiff(path: PathLike) -> TiffInfo:
    """Read only the TIFF header (no pixel data)."""
    try:
        with tifffile.TiffFile(str(path)) as tf:
            pg = tf.pages[0]
            return TiffInfo(Path(path), tuple(pg.shape), str(np.dtype(pg.dtype)), int(pg.bitspersample),
                            len(tf.pages), getattr(pg.compression, "name", str(pg.compression)),
                            getattr(pg.photometric, "name", str(pg.photometric)))
    except Exception as exc:
        raise ProjectionLoadError(f"Cannot read TIFF header of {path}: {exc}") from exc


def read_tiff_projection(path: PathLike) -> np.ndarray:
    """Read page 0 unchanged (native dtype, no rescaling). Must be 2-D."""
    try:
        arr = tifffile.imread(str(path), key=0)
    except Exception as exc:
        raise ProjectionLoadError(f"Cannot read {path}: {exc}") from exc
    if arr.ndim != 2:
        raise ProjectionLoadError(f"{path}: expected a 2-D grayscale image, got shape {arr.shape}")
    return arr


class TiffSequenceSource:
    """Projections stored as single-page TIFFs, read one (or a batch) at a time.

    Independent files cannot be one memory map, so we read lazily. ``paths`` must already be
    in acquisition order (use the order of MatchResult.matched).
    """

    def __init__(self, paths: Sequence[Path], ids: Sequence[int]):
        if len(paths) != len(ids) or not paths:
            raise ProjectionLoadError("TiffSequenceSource needs equal, non-empty paths and ids")
        self.paths, self.ids = list(paths), list(ids)
        first = inspect_tiff(self.paths[0])
        if len(first.shape) != 2:
            raise ProjectionLoadError(f"{first.path}: not a single 2-D image (shape {first.shape})")
        self.image_shape: Tuple[int, int] = (first.shape[0], first.shape[1])
        self.dtype = np.dtype(first.dtype)
        self.first_info = first

    def __len__(self) -> int:
        return len(self.paths)

    def read(self, i: int) -> np.ndarray:
        arr = read_tiff_projection(self.paths[i])
        if arr.shape != self.image_shape or arr.dtype != self.dtype:
            raise ProjectionLoadError(f"{self.paths[i]}: shape/dtype {arr.shape}/{arr.dtype} differs from "
                                      f"first image {self.image_shape}/{self.dtype}")
        return arr

    def read_batch(self, start: int, stop: int) -> np.ndarray:
        stop = min(stop, len(self))
        out = np.empty((stop - start,) + self.image_shape, dtype=self.dtype)
        for k, i in enumerate(range(start, stop)):
            out[k] = self.read(i)
        return out

    def iter_batches(self, batch_size: int) -> Iterator[Tuple[int, np.ndarray]]:
        for s in range(0, len(self), batch_size):
            yield s, self.read_batch(s, s + batch_size)

    def verify_headers(self) -> List[Issue]:
        """Fast whole-folder consistency check from headers only."""
        I: List[Issue] = []
        for p in self.paths:
            info = inspect_tiff(p)
            if info.shape != self.image_shape or info.dtype != str(self.dtype) or info.n_pages != 1:
                I.append(Issue("error", "tiff_inconsistent",
                               f"{p.name}: shape {info.shape}, dtype {info.dtype}, pages {info.n_pages} "
                               f"(expected {self.image_shape}, {self.dtype}, 1 page)"))
        return I


def open_dat_stack(cfg: InputConfig) -> np.ndarray:
    """Open a raw binary stack as a read-only memmap of shape (N, H, W) (no copy).

    The file size must equal header + N*H*W*itemsize EXACTLY; anything else is an error
    because a wrong header offset or dtype would silently scramble the images.
    """
    for n in ("dat_path", "dat_width", "dat_height", "dat_n_projections"):
        if getattr(cfg, n) is None:
            raise ProjectionLoadError(f"input.{n} must be specified for a DAT stack")
    p = Path(cfg.dat_path)
    if not p.is_file():
        raise ProjectionLoadError(f"DAT file not found: {p}")
    if cfg.dat_dtype not in ("uint16", "float32") or cfg.dat_byte_order not in ("little", "big"):
        raise ProjectionLoadError("dat_dtype must be uint16/float32 and dat_byte_order little/big")
    dt = np.dtype(("<" if cfg.dat_byte_order == "little" else ">") + {"uint16": "u2", "float32": "f4"}[cfg.dat_dtype])
    n, h, w = cfg.dat_n_projections, cfg.dat_height, cfg.dat_width
    expected = cfg.dat_header_bytes + n * h * w * dt.itemsize
    actual = p.stat().st_size
    if actual != expected:
        raise ProjectionLoadError(
            f"{p.name}: size {actual} B != expected {expected} B = header {cfg.dat_header_bytes} + "
            f"{n}*{h}*{w}*{dt.itemsize} ({'+' if actual > expected else '-'}{abs(actual - expected)} B). "
            f"Check width/height/count/dtype/header offset.")
    shape = (n, h, w) if cfg.dat_order == "NHW" else (h, n, w)
    mm = np.memmap(p, dtype=dt, mode="r", offset=cfg.dat_header_bytes, shape=shape)
    return mm if cfg.dat_order == "NHW" else mm.transpose(1, 0, 2)


class DatStackSource:
    """Same interface as TiffSequenceSource, backed by a DAT memmap (N, H, W)."""

    def __init__(self, stack: np.ndarray, ids: Sequence[int], positions: Optional[Sequence[int]] = None):
        self.stack = stack
        self.positions = list(range(stack.shape[0])) if positions is None else list(positions)
        self.ids = list(ids)
        self.image_shape = (stack.shape[1], stack.shape[2])
        self.dtype = stack.dtype

    def __len__(self) -> int:
        return len(self.positions)

    def read(self, i: int) -> np.ndarray:
        return np.array(self.stack[self.positions[i]])

    def read_batch(self, start: int, stop: int) -> np.ndarray:
        stop = min(stop, len(self))
        return np.stack([self.read(i) for i in range(start, stop)])

    def iter_batches(self, batch_size: int) -> Iterator[Tuple[int, np.ndarray]]:
        for s in range(0, len(self), batch_size):
            yield s, self.read_batch(s, s + batch_size)

    def verify_headers(self) -> List[Issue]:
        return []


def build_source(cfg: InputConfig, table: AngleTable) -> Tuple[Union[TiffSequenceSource, DatStackSource], MatchResult]:
    """Discover images, match to angles by identifier, return an ordered source + match report."""
    if cfg.projection_dir:
        disc = discover_tiff_projections(cfg.projection_dir, cfg.projection_glob, cfg.projection_id_regex)
        m = match_projections(disc.files_by_id, table, id_offset=cfg.id_offset,
                              duplicate_image_ids=list(disc.duplicate_ids))
        if disc.unparsed_files:
            m.issues.append(Issue("warning", "unparsed_files",
                                  f"{len(disc.unparsed_files)} file(s) matched the glob but no id could be "
                                  f"parsed, e.g. {disc.unparsed_files[0].name}"))
        require_match(m, allow_partial=not cfg.require_complete_match)
        if not m.matched:
            raise ProjectionMatchError("No projection could be matched to an angle")
        src: Union[TiffSequenceSource, DatStackSource] = TiffSequenceSource(
            [x.path for x in m.matched], [x.file_id for x in m.matched])
        if cfg.expected_dtype and str(src.dtype) != cfg.expected_dtype:
            raise ProjectionLoadError(f"TIFF dtype {src.dtype} != input.expected_dtype {cfg.expected_dtype}")
        return src, m
    stack = open_dat_stack(cfg)
    ids = {cfg.dat_first_projection_id + k: None for k in range(stack.shape[0])}
    m = match_projections(ids, table, id_offset=cfg.id_offset)
    require_match(m, allow_partial=not cfg.require_complete_match)
    pos = [x.file_id - cfg.dat_first_projection_id for x in m.matched]
    return DatStackSource(stack, [x.file_id for x in m.matched], pos), m


def load_reference_image(path: PathLike) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dark/flat reference: one TIFF, or a folder of TIFFs averaged (float64 accumulation).

    Returns (float32 image, info). Averaging N frames is a *computed* quantity, recorded
    in info; the original files are untouched.
    """
    p = Path(path)
    files = sorted(f for f in p.glob("*.tif*") if f.is_file()) if p.is_dir() else [p]
    if not files or not all(f.is_file() for f in files):
        raise ProjectionLoadError(f"No reference TIFF found at {p}")
    acc, dtype, shape = None, None, None
    for f in files:
        a = read_tiff_projection(f)
        if acc is None:
            acc, dtype, shape = np.zeros(a.shape, np.float64), a.dtype, a.shape
        elif a.shape != shape or a.dtype != dtype:
            raise ProjectionLoadError(f"{f}: reference frame shape/dtype differs from the first")
        acc += a
    img = (acc / len(files)).astype(np.float32)
    return img, {"n_frames": len(files), "source_dtype": str(dtype), "shape": tuple(shape),
                 "averaged": len(files) > 1, "path": str(p)}


# =========================================================================== #
# 16-bit / intensity verification
# =========================================================================== #
@dataclass
class IntensityReport:
    dtype: str
    stored_bits: int
    n_samples: int
    minimum: float
    maximum: float
    percentiles: Dict[str, float]
    bits_needed_for_max: Optional[int]
    n_unique_values: int
    level_spacing_gcd: Optional[int]
    fraction_at_dtype_max: Optional[float]
    fraction_at_zero: float
    fraction_at_observed_max: float
    white_level: Optional[float]
    fraction_above_white_level: Optional[float]
    p99_over_white_level: Optional[float]
    verdict: List[str]


def inspect_intensity(samples: np.ndarray, *, white_level: Optional[float] = None,
                      max_samples: int = 20_000_000) -> IntensityReport:
    """Describe the intensity values WITHOUT altering them.

    * bits_needed_for_max : ceil(log2(max+1)); a 16-bit container whose max is 4095 holds
      12-bit data. This is an *upper-range* indicator only; noise floors matter too.
    * level_spacing_gcd   : gcd of the differences between occurring values. >1 means the
      values sit on a coarse lattice (e.g. 12-bit data multiplied by 16), i.e. stretched.
    * WhiteLevel is compared, never assumed to be I0.
    """
    arr = np.asarray(samples).ravel()
    if arr.size > max_samples:
        arr = arr[::int(math.ceil(arr.size / max_samples))]
    dt = arr.dtype
    is_int = np.issubdtype(dt, np.integer)
    finite = arr[np.isfinite(arr)] if not is_int else arr
    pct = {f"p{q}": float(np.percentile(finite, q)) for q in (0.1, 1, 50, 99, 99.9)}
    mn, mx = float(finite.min()), float(finite.max())
    bits_max = int(int(mx).bit_length()) if is_int and mx >= 0 else None
    uniq = np.unique(finite)
    gcd = None
    if is_int and uniq.size > 1:
        gcd = int(np.gcd.reduce((uniq.astype(np.int64) - int(uniq[0]))[1:]))
    dtmax = float(np.iinfo(dt).max) if is_int else None
    verdict: List[str] = []
    f_dtmax = float(np.mean(arr == dtmax)) if dtmax is not None else None
    f_zero = float(np.mean(arr == 0))
    f_max = float(np.mean(arr == mx))
    if is_int:
        verdict.append(f"stored as {dt} ({dt.itemsize * 8}-bit container); observed range "
                       f"[{mn:.0f}, {mx:.0f}] -> needs {bits_max} bit(s) for the maximum")
        if bits_max is not None and bits_max < dt.itemsize * 8 - 1:
            verdict.append(f"Maximum uses only {bits_max} of {dt.itemsize * 8} bits: smaller effective range "
                           f"in a larger container (or an under-exposed sample).")
        elif bits_max is not None:
            verdict.append(f"Values reach into the top bit(s) of the {dt.itemsize * 8}-bit range in this sample.")
        if gcd and gcd > 1:
            verdict.append(f"Occurring values are all multiples-of-{gcd} apart: data look scaled/stretched "
                           f"(true quantisation step = {gcd}). Do NOT rescale further.")
        if f_dtmax and f_dtmax > 0:
            verdict.append(f"{f_dtmax:.2e} of samples equal {dtmax:.0f}: detector/container saturation likely.")
        elif f_max > 1e-3 and mx < (dtmax or math.inf):
            verdict.append(f"{f_max:.2e} of samples sit at the observed maximum {mx:.0f}: a clipping plateau "
                           f"below the container maximum (possible saturation or digital clamp).")
    else:
        verdict.append(f"floating-point data ({dt}); range [{mn:.6g}, {mx:.6g}]; no integer bit-depth analysis")
    ratio = frac_above = None
    if white_level:
        ratio = pct["p99"] / white_level
        frac_above = float(np.mean(arr > white_level))
        verdict.append(f"p99 = {pct['p99']:.0f} is {ratio:.3f} x WhiteLevel({white_level:g}); "
                       f"{frac_above:.2e} of samples exceed it. The meaning of WhiteLevel is UNVERIFIED: if "
                       f"open-beam regions sit near it, the data are probably already normalised to it; if "
                       f"far from it, it is not the air level of these images.")
    return IntensityReport(str(dt), dt.itemsize * 8, int(arr.size), mn, mx, pct, bits_max, int(uniq.size),
                           gcd, f_dtmax, f_zero, f_max, white_level, frac_above, ratio, verdict)


def inspect_projection_intensity(source: Union[TiffSequenceSource, DatStackSource], n_samples: int = 8,
                                 white_level: Optional[float] = None) -> IntensityReport:
    """Inspect evenly spaced projections of a source."""
    idx = np.unique(np.linspace(0, len(source) - 1, min(n_samples, len(source))).astype(int))
    return inspect_intensity(np.stack([source.read(int(i)) for i in idx]), white_level=white_level)


# =========================================================================== #
# E. Exporters
# =========================================================================== #
# Volume convention used everywhere in this package: array shape (nz, ny, nx), C order, so
# x varies fastest in memory. Spacing is (sx, sy, sz) in mm.
@dataclass
class ExportScaling:
    """How stored grey values relate to reconstructed attenuation values."""
    dtype: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    n_levels: Optional[int] = None            # 65535 steps for uint16
    clipped_low_fraction: Optional[float] = None
    clipped_high_fraction: Optional[float] = None
    formula: str = "value = stored (no scaling)"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def choose_window(volume: np.ndarray, mode: str, vmin: Optional[float], vmax: Optional[float],
                  p_low: float = 0.2, p_high: float = 99.8, max_samples: int = 5_000_000) -> Tuple[float, float]:
    """Pick the [vmin, vmax] that maps to 0..65535. 'percentile' uses a strided subsample."""
    if mode == "explicit":
        if vmin is None or vmax is None or not vmin < vmax:
            raise ExportError("explicit window requires vmin < vmax")
        return float(vmin), float(vmax)
    if mode != "percentile":
        raise ExportError(f"unknown window mode {mode!r}")
    flat = volume.reshape(-1)
    sub = np.asarray(flat[::max(1, flat.size // max_samples)], dtype=np.float64)
    sub = sub[np.isfinite(sub)]
    if sub.size == 0:
        raise ExportError("volume contains no finite values")
    lo, hi = np.percentile(sub, [p_low, p_high])
    if not lo < hi:
        raise ExportError(f"degenerate percentile window [{lo}, {hi}]")
    return float(lo), float(hi)


def to_uint16(slice_f: np.ndarray, vmin: float, vmax: float) -> Tuple[np.ndarray, int, int]:
    """code = round( clip((x - vmin)/(vmax - vmin), 0, 1) * 65535 ), computed in float64.

    No integer arithmetic happens before clipping, so nothing can overflow or wrap. NaN
    maps to 0 and is counted as 'low'. Returns (uint16 slice, n_low, n_high).
    """
    x = np.asarray(slice_f, dtype=np.float64)
    bad = ~np.isfinite(x)
    n_low = int(np.sum(x < vmin) + bad.sum())
    n_high = int(np.sum(x > vmax))
    t = np.clip((np.where(bad, vmin, x) - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.rint(t * 65535.0).astype(np.uint16), n_low, n_high


def _check_volume(volume: np.ndarray, spacing_mm: Tuple[float, float, float]) -> None:
    if volume.ndim != 3:
        raise ExportError(f"volume must be 3-D (nz, ny, nx), got shape {volume.shape}")
    if len(spacing_mm) != 3 or any(not s > 0 for s in spacing_mm):
        raise ExportError(f"spacing_mm must be three positive numbers, got {spacing_mm}")


def write_tiff_float32(volume: np.ndarray, path: PathLike, spacing_mm: Tuple[float, float, float],
                       extra: Optional[Dict[str, Any]] = None) -> ExportScaling:
    """Multi-page 32-bit float TIFF, one slice per page; values are the reconstructed values."""
    _check_volume(volume, spacing_mm)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    desc = json.dumps({"axes": "ZYX", "dtype": "float32", "spacing_mm_xyz": list(spacing_mm),
                       "values": "reconstructed attenuation, unscaled", **(extra or {})})
    with tifffile.TiffWriter(str(p), bigtiff=volume.nbytes > 3.5e9) as tw:
        for k in range(volume.shape[0]):
            tw.write(np.ascontiguousarray(volume[k], dtype=np.float32), photometric="minisblack",
                     description=desc if k == 0 else None, metadata=None)
    sc = ExportScaling("float32")
    _write_sidecar(p, spacing_mm, sc, volume.shape, "little")
    return sc


def write_tiff_uint16(volume: np.ndarray, path: PathLike, spacing_mm: Tuple[float, float, float],
                      vmin: float, vmax: float, extra: Optional[Dict[str, Any]] = None) -> ExportScaling:
    """Multi-page uint16 TIFF with an explicit linear window: [vmin, vmax] -> [0, 65535].

    Recover values with  x = vmin + code * (vmax - vmin) / 65535  (stored in the sidecar JSON).
    Values outside the window are clipped and the clipped fractions are recorded.
    """
    _check_volume(volume, spacing_mm)
    if not vmin < vmax:
        raise ExportError("vmin must be < vmax")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lo = hi = 0
    desc = json.dumps({"axes": "ZYX", "dtype": "uint16", "window_min": vmin, "window_max": vmax,
                       "formula": "x = window_min + code*(window_max-window_min)/65535",
                       "spacing_mm_xyz": list(spacing_mm), **(extra or {})})
    with tifffile.TiffWriter(str(p), bigtiff=volume.shape[0] * volume.shape[1] * volume.shape[2] * 2 > 3.5e9) as tw:
        for k in range(volume.shape[0]):
            s16, nl, nh = to_uint16(volume[k], vmin, vmax)
            lo, hi = lo + nl, hi + nh
            tw.write(s16, photometric="minisblack", description=desc if k == 0 else None, metadata=None)
    n = volume.size
    sc = ExportScaling("uint16", vmin, vmax, 65535, lo / n, hi / n,
                       "x = vmin + code*(vmax - vmin)/65535")
    _write_sidecar(p, spacing_mm, sc, volume.shape, "little")
    return sc


def write_vol_vgi(volume: np.ndarray, vol_path: PathLike, spacing_mm: Tuple[float, float, float],
                  origin_mm: Optional[Tuple[float, float, float]] = None,
                  vmin: Optional[float] = None, vmax: Optional[float] = None,
                  as_uint16: bool = False) -> ExportScaling:
    """Raw .vol plus a .vgi descriptor, and a JSON sidecar listing exact raw-import parameters.

    Binary layout (documented, certain): little-endian, no header, C order of (nz, ny, nx) --
    x fastest, then y, then z -- float32 or uint16 (window vmin..vmax -> 0..65535).
    The .vgi text follows the layout commonly used for VGSTUDIO raw volumes (sections
    {volume1}/[file1]/[geometry]) but is UNVERIFIED against your VGSTUDIO MAX version. If it
    does not load, use File > Import > Raw Data with the values in the sidecar JSON. The
    physical origin is stored only in the sidecar; the .vgi has no origin entry here.
    """
    _check_volume(volume, spacing_mm)
    vp = Path(vol_path).with_suffix(".vol")
    vp.parent.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = volume.shape
    lo = hi = 0
    with open(vp, "wb") as fh:
        for k in range(nz):
            if as_uint16:
                if vmin is None or vmax is None:
                    raise ExportError("uint16 .vol needs vmin and vmax")
                s, nl, nh = to_uint16(volume[k], vmin, vmax)
                lo, hi = lo + nl, hi + nh
                s.astype("<u2").tofile(fh)
            else:
                np.asarray(volume[k], dtype="<f4").tofile(fh)
    dtype_txt, bits = ("unsigned integer", 16) if as_uint16 else ("float", 32)
    sx, sy, sz = spacing_mm
    vgi = (f"{{volume1}}\n[file1]\nSkipHeader = 0\nFileFormat = raw\nSize = {nx} {ny} {nz}\n"
           f"Name = {vp.name}\nDatatype = {dtype_txt}\nBitsPerElement = {bits}\n"
           f"[geometry]\nresolution = {sx:.9g} {sy:.9g} {sz:.9g}\nunit = mm\n")
    vp.with_suffix(".vgi").write_text(vgi, encoding="utf-8")
    n = volume.size
    sc = (ExportScaling("uint16", vmin, vmax, 65535, lo / n, hi / n, "x = vmin + code*(vmax - vmin)/65535")
          if as_uint16 else ExportScaling("float32"))
    _write_sidecar(vp, spacing_mm, sc, volume.shape, "little", origin_mm, vgi_verified=False)
    return sc


def _write_sidecar(data_path: Path, spacing_mm: Tuple[float, float, float], sc: ExportScaling,
                   shape: Tuple[int, int, int], byte_order: str,
                   origin_mm: Optional[Tuple[float, float, float]] = None, vgi_verified: Optional[bool] = None) -> None:
    nz, ny, nx = shape
    info = {"file": data_path.name, "dimensions_xyz": [nx, ny, nz], "array_order": "(z, y, x), x fastest",
            "byte_order": byte_order, "spacing_mm_xyz": list(spacing_mm),
            "origin_mm_xyz": list(origin_mm) if origin_mm else None,
            "origin_note": "None = not specified (do not assume centred)", "intensity": sc.to_dict()}
    if vgi_verified is not None:
        info["vgi_verified_against_vgstudio"] = vgi_verified
    data_path.with_name(data_path.name + ".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
