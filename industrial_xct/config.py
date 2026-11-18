"""config.py - configuration, provenance vocabulary and validation.

PROBLEM THIS MODULE SOLVES
    A reconstruction depends on dozens of numbers (distances, pixel pitch, angles,
    correction switches). If they are scattered through the code, you can never say
    *where a number came from*. This module (1) holds every user-controllable setting
    in typed dataclasses, (2) loads them from YAML without silently ignoring typos,
    (3) validates them before any heavy computation, and (4) defines the vocabulary
    (`Provenance`, `ParameterValue`, `Issue`) that every other module uses to say
    "measured / calculated / estimated / assumed / unavailable".

DATA FLOW
    default.yaml (+ CLI overrides) --load_config--> PipelineConfig --validate_config--> Issues
    io_xtek.build_scanner_geometry() combines PipelineConfig.geometry (user overrides)
    with the .xtekct metadata into ParameterValue objects.

DESIGN RULE
    A geometry field set to ``None`` means "take it from the metadata or report it as
    unavailable". No scanner geometry is ever hard-coded in this package.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (Any, Dict, List, Optional, Tuple, Union, get_args,
                    get_origin, get_type_hints)

import yaml

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised for malformed or invalid configuration."""


# --------------------------------------------------------------------------- #
# Shared vocabulary: provenance, parameter values, issues
# --------------------------------------------------------------------------- #
class Provenance(str, Enum):
    """Where a parameter value came from. Reports must always show this."""
    METADATA = "metadata"          # read directly from a scanner file
    USER = "user_override"         # supplied by the user in config / CLI
    CALCULATED = "calculated"      # derived arithmetically from other known values
    ESTIMATED = "estimated"        # obtained by an image-based calibration
    ASSUMED = "assumed"            # an explicit assumption (always flagged in reports)
    UNAVAILABLE = "unavailable"    # not known; something is needed to determine it


@dataclass(frozen=True)
class ParameterValue:
    """A value together with its unit, provenance and any caveat."""
    value: Any
    provenance: Provenance
    unit: Optional[str] = None
    source: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        return self.value is not None and self.provenance is not Provenance.UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "provenance": self.provenance.value,
                "unit": self.unit, "source": self.source, "note": self.note}


@dataclass(frozen=True)
class Issue:
    """A validation finding. severity is 'error', 'warning' or 'info'."""
    severity: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}:{self.code}] {self.message}"


def raise_on_errors(issues: List[Issue], context: str) -> None:
    """Raise ConfigError listing every error-severity issue (not just the first)."""
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ConfigError(f"{context}: {len(errors)} error(s):\n  " +
                          "\n  ".join(str(e) for e in errors))


# --------------------------------------------------------------------------- #
# Configuration dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class InputConfig:
    """Where the data are and how filenames map to projection identifiers."""
    metadata_path: Optional[str] = None       # .xtekct
    angle_path: Optional[str] = None          # .ang or _ctdata.txt
    projection_dir: Optional[str] = None      # folder of single-page TIFFs
    dat_path: Optional[str] = None            # raw binary stack (alternative to TIFFs)
    dark_path: Optional[str] = None           # TIFF file, or folder of TIFFs (averaged)
    flat_path: Optional[str] = None
    projection_glob: str = "*.tif*"
    # Regex whose group(1) is the integer projection identifier in the file name.
    projection_id_regex: str = r"(\d+)\.tiff?$"
    # Added to identifiers parsed from file names BEFORE matching to the angle table.
    # Default 0: pairing is never shifted unless you say so.
    id_offset: int = 0
    require_complete_match: bool = True       # error if any image/angle is unmatched
    expected_dtype: Optional[str] = None      # e.g. "uint16"; None = accept any, report it
    # Raw DAT stack: everything must be stated explicitly.
    dat_width: Optional[int] = None
    dat_height: Optional[int] = None
    dat_n_projections: Optional[int] = None
    dat_dtype: str = "uint16"                 # "uint16" | "float32"
    dat_byte_order: str = "little"            # "little" | "big"
    dat_header_bytes: int = 0
    dat_order: str = "NHW"                    # "NHW" projection-major | "HNW" row-major
    dat_first_projection_id: int = 1          # identifier of the first stored projection


@dataclass
class GeometryConfig:
    """User overrides of scanner geometry. ``None`` = use metadata / report unavailable."""
    sod_mm: Optional[float] = None
    sdd_mm: Optional[float] = None
    detector_pixels_x: Optional[int] = None
    detector_pixels_y: Optional[int] = None
    pixel_size_x_mm: Optional[float] = None
    pixel_size_y_mm: Optional[float] = None
    detector_offset_x_mm: Optional[float] = None
    detector_offset_y_mm: Optional[float] = None
    # Exactly ONE source of CoR correction is allowed (applied once, see validate_config):
    #   "none"     : no correction beyond the detector offset (CoR assumed at the
    #                geometric detector centre; reported as ASSUMED)
    #   "manual"   : cor_manual_px below
    #   "metadata" : CentreOfRotationTop/Bottom from .xtekct (unit/sign unverified!)
    #   "estimate" : image-based estimate (artifacts.cor, Pass 2)
    cor_source: str = "none"
    cor_manual_px: Optional[float] = None
    rotation_axis_tilt_deg: Optional[float] = None
    detector_roll_deg: Optional[float] = None
    detector_tilt_deg: Optional[float] = None
    voxel_size_mm: Optional[float] = None
    detector_offset_v_px: float = 0.0
    rotation_sign: int = 1
    angle_offset_deg: float = 0.0
    detector_u_sign: int = 1
    detector_v_sign: int = -1
    volume_offset_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_voxels: Optional[Tuple[int, int, int]] = None     # (nx, ny, nz)
    volume_offset_mm: Optional[Tuple[float, float, float]] = None

    acquisition_direction: str = "auto"       # "auto" | "increasing" | "decreasing"
    full_scan_tolerance_deg: float = 1.0      # allowed missing arc for "full scan"
    volume_offset_mm: Optional[Tuple[float, float, float]] = None


@dataclass
class PreprocessingConfig:
    # "dark_flat": (I-D)/(F-D) with calibration images
    # "scalar_i0": (I-D)/I0 with an explicit, user-verified I0 (e.g. data already
    #              flat-corrected by the scanner software)
    # "none"     : input is already transmission (float data only)
    normalisation_mode: str = "dark_flat"
    apply_dark: bool = True                   # False = dark treated as 0 (explicit choice)
    i0_value: Optional[float] = None          # required for "scalar_i0"
    min_denominator_counts: float = 1.0       # F-D below this => pixel invalid
    # Transmission clamp before -ln(); counts are reported, never hidden.
    transmission_floor: float = 1.0e-6
    transmission_ceiling: Optional[float] = None
    # Bad pixels (needs dark+flat to detect). Disabled until you inspect the mask.
    bad_pixel_correction: bool = False
    bad_pixel_gain_ratio_range: Tuple[float, float] = (0.5, 1.5)
    bad_pixel_hot_mad_factor: float = 8.0
    bad_pixel_window: int = 3
    # Frame-to-frame flux normalisation from a VERIFIED air region.
    air_normalisation: bool = False
    air_region: Optional[Tuple[int, int, int, int]] = None   # (x0, x1, y0, y1), end-exclusive
    air_min_transmission: float = 0.9
    air_max_spatial_cv: float = 0.02
    batch_size: int = 32
    memory_limit_gb: float = 8.0
    work_dir: Optional[str] = None            # used if the stack does not fit in RAM
    inspect_sample_projections: int = 8


@dataclass
class ReconstructionConfig:
    algorithm: str = "FDK_CUDA"               # "FDK_CUDA" | "SIRT3D_CUDA" | "CGLS3D_CUDA"
    filter_name: str = "ram-lak"
    filter_cutoff: Optional[float] = None     # not exposed until verified against ASTRA
    gpu_index: int = 0
    gpu_memory_limit_gb: Optional[float] = None
    host_memory_limit_gb: float = 4.5
    max_slab_voxels_z: Optional[int] = 64
    sweep_slab_slices: int = 5
    voxel_size_mm: Optional[float] = None
    volume_voxels: Optional[Tuple[int, int, int]] = None
    iterations: int = 100
    non_negativity: bool = False


@dataclass
class CoRConfig:
    enabled: bool = False
    method: str = "phase_cross_correlation"
    n_pairs: int = 5
    min_pairs: int = 3
    rows_margin_px: int = 64
    row_variation_tolerance_px: float = 0.5
    sweep_half_range_px: float = 2.0
    sweep_step_px: float = 0.5
    pair_angle_tolerance_deg: float = 0.5
    upsample_factor: int = 100
    n_row_bands: int = 3
    consistency_tolerance_px: float = 0.25
    sweep_values_px: Optional[List[float]] = None


@dataclass
class RingConfig:
    enabled: bool = False
    method: str = "wavelet_fft"               # "wavelet_fft" | "column_median"
    wavelet: str = "db5"
    levels: int = 4
    sigma: float = 2.0
    window_px: int = 31


@dataclass
class BeamHardeningConfig:
    enabled: bool = False
    coefficients: Optional[Tuple[float, float, float]] = None   # c1, c2, c3
    coefficients_are_calibrated: bool = False
    label: str = ""


@dataclass
class ScatterConfig:
    enabled: bool = False
    method: str = "none"


@dataclass
class MetalConfig:
    enabled: bool = False
    threshold: Optional[float] = None


@dataclass
class ArtifactConfig:
    cor: CoRConfig = field(default_factory=CoRConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    beam_hardening: BeamHardeningConfig = field(default_factory=BeamHardeningConfig)
    scatter: ScatterConfig = field(default_factory=ScatterConfig)
    metal: MetalConfig = field(default_factory=MetalConfig)


@dataclass
class ExportConfig:
    output_dir: str = "output"
    basename: str = "reconstruction"
    formats: List[str] = field(default_factory=lambda: ["tiff32"])   # tiff32|tiff16|vol
    tiff16_window_mode: str = "explicit"      # "explicit" | "percentile"
    window_min: Optional[float] = None
    window_max: Optional[float] = None
    percentile_low: float = 0.2
    percentile_high: float = 99.8


@dataclass
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    log_level: str = "INFO"


# --------------------------------------------------------------------------- #
# Loading (strict: unknown keys and wrong types are errors, not silently ignored)
# --------------------------------------------------------------------------- #
def _coerce(hint: Any, value: Any, where: str) -> Any:
    origin = get_origin(hint)
    if origin is Union:
        inner = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        return _coerce(inner[0], value, where)
    if dataclasses.is_dataclass(hint):
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ConfigError(f"'{where}' must be a mapping, got {type(value).__name__}")
        return _build(hint, value, where)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"'{where}' must be a list, got {value!r}")
        return tuple(value)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"'{where}' must be a list, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{where}' must be a number, got {value!r}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{where}' must be an integer, got {value!r}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{where}' must be true/false, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{where}' must be a string, got {value!r}")
        return value
    return value


def _build(cls: type, data: Dict[str, Any], where: str) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) {unknown} in section '{where}'. "
                          f"Valid keys: {sorted(known)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[Union[str, Path]] = None,
                overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """Build a PipelineConfig from optional YAML plus an optional override dict.

    Missing keys take the dataclass defaults (all "safe": corrections disabled).
    """
    data: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Top level of {p} must be a mapping")
    if overrides:
        data = _deep_merge(data, overrides)
    return _build(PipelineConfig, data, "config")


def config_to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    """Plain-dict form for the machine-readable report (tuples become lists)."""
    def conv(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x
    return conv(dataclasses.asdict(cfg))


def dump_yaml(cfg: PipelineConfig, path: Union[str, Path]) -> None:
    Path(path).write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False),
                          encoding="utf-8")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
# Filters the spec lists. To be confirmed against the installed ASTRA in Pass 2.
SUPPORTED_FILTERS = ("ram-lak", "shepp-logan", "cosine", "hamming", "hann", "none")
SUPPORTED_ALGORITHMS = ("FDK_CPU", "FDK_CUDA", "SIRT3D_CUDA", "CGLS3D_CUDA")
COR_SOURCES = ("none", "manual", "metadata", "estimate")


def _positive(issues: List[Issue], name: str, value: Optional[float]) -> None:
    if value is not None and not value > 0:
        issues.append(Issue("error", "non_positive", f"{name} must be > 0, got {value}"))


def validate_config(cfg: PipelineConfig, *, require_inputs: bool = True) -> List[Issue]:
    """Return every problem found (does not stop at the first)."""
    I: List[Issue] = []
    inp, geo, pre = cfg.input, cfg.geometry, cfg.preprocessing
    rec, art, exp = cfg.reconstruction, cfg.artifacts, cfg.export

    # --- inputs -------------------------------------------------------------
    if require_inputs:
        if bool(inp.projection_dir) == bool(inp.dat_path):
            I.append(Issue("error", "projection_source",
                           "Set exactly one of input.projection_dir or input.dat_path"))
        if not inp.angle_path:
            I.append(Issue("error", "angle_path", "input.angle_path is required"))
        for name in ("metadata_path", "angle_path", "projection_dir", "dat_path",
                     "dark_path", "flat_path"):
            v = getattr(inp, name)
            if v and not Path(v).exists():
                I.append(Issue("error", "path_missing", f"input.{name} does not exist: {v}"))
    if inp.dat_path:
        for n in ("dat_width", "dat_height", "dat_n_projections"):
            if getattr(inp, n) is None:
                I.append(Issue("error", "dat_param", f"input.{n} must be given explicitly for DAT"))
        if inp.dat_dtype not in ("uint16", "float32"):
            I.append(Issue("error", "dat_dtype", "input.dat_dtype must be uint16 or float32"))
        if inp.dat_byte_order not in ("little", "big"):
            I.append(Issue("error", "dat_order", "input.dat_byte_order must be little or big"))
        if inp.dat_order not in ("NHW", "HNW"):
            I.append(Issue("error", "dat_layout", "input.dat_order must be NHW or HNW"))
        if inp.dat_header_bytes < 0:
            I.append(Issue("error", "dat_header", "input.dat_header_bytes must be >= 0"))

    # --- geometry -----------------------------------------------------------
    for n in ("sod_mm", "sdd_mm", "pixel_size_x_mm", "pixel_size_y_mm", "voxel_size_mm"):
        _positive(I, f"geometry.{n}", getattr(geo, n))
    if geo.sod_mm and geo.sdd_mm and geo.sdd_mm <= geo.sod_mm:
        I.append(Issue("error", "sdd_le_sod", "geometry.sdd_mm must exceed geometry.sod_mm"))
    if geo.cor_source not in COR_SOURCES:
        I.append(Issue("error", "cor_source", f"geometry.cor_source must be one of {COR_SOURCES}"))
    if geo.cor_source == "manual" and geo.cor_manual_px is None:
        I.append(Issue("error", "cor_manual", "cor_source='manual' requires geometry.cor_manual_px"))
    if geo.cor_source != "manual" and geo.cor_manual_px is not None:
        I.append(Issue("error", "cor_manual_unused",
                       "geometry.cor_manual_px is set but cor_source is not 'manual'; "
                       "it would be ignored. Remove it or set cor_source='manual'."))
    if geo.cor_source == "metadata":
        I.append(Issue("warning", "cor_metadata",
                       "cor_source='metadata': the unit and sign convention of "
                       "CentreOfRotationTop/Bottom are unverified; Pass 2 must check them."))
    # CoR applied exactly once: the 'estimate' source and artifacts.cor must agree.
    if art.cor.enabled and geo.cor_source != "estimate":
        I.append(Issue("error", "cor_double",
                       "artifacts.cor.enabled=true requires geometry.cor_source='estimate' "
                       "so that the CoR is applied exactly once."))
    if geo.cor_source == "estimate" and not art.cor.enabled:
        I.append(Issue("error", "cor_estimate_off",
                       "geometry.cor_source='estimate' but artifacts.cor.enabled=false."))
    if geo.acquisition_direction not in ("auto", "increasing", "decreasing"):
        I.append(Issue("error", "direction", "geometry.acquisition_direction invalid"))
    if geo.full_scan_tolerance_deg < 0:
        I.append(Issue("error", "tol", "geometry.full_scan_tolerance_deg must be >= 0"))
    for name, vox in (("geometry.volume_voxels", geo.volume_voxels),
                      ("reconstruction.volume_voxels", rec.volume_voxels)):
        if vox is not None and (len(vox) != 3 or any((not isinstance(v, int)) or v < 1 for v in vox)):
            I.append(Issue("error", "volume_voxels", f"{name} must be three positive integers"))

    # --- preprocessing --------------------------------------------------------
    if pre.normalisation_mode not in ("dark_flat", "scalar_i0", "none"):
        I.append(Issue("error", "norm_mode", "preprocessing.normalisation_mode invalid"))
    if pre.normalisation_mode == "scalar_i0" and not (pre.i0_value and pre.i0_value > 0):
        I.append(Issue("error", "i0", "scalar_i0 mode requires preprocessing.i0_value > 0 "
                                      "(verify the meaning of WhiteLevel first)"))
    if pre.normalisation_mode == "dark_flat":
        if not inp.flat_path:
            I.append(Issue("error", "flat_missing", "dark_flat mode requires input.flat_path"))
        if pre.apply_dark and not inp.dark_path:
            I.append(Issue("error", "dark_missing", "apply_dark=true requires input.dark_path "
                                                    "(or set apply_dark=false explicitly)"))
    if pre.normalisation_mode == "dark_flat" and not pre.apply_dark:
        I.append(Issue("warning", "dark_off", "Dark correction disabled: dark level assumed 0."))
    if pre.transmission_floor <= 0:
        I.append(Issue("error", "floor", "preprocessing.transmission_floor must be > 0"))
    if pre.transmission_ceiling is not None and pre.transmission_ceiling <= pre.transmission_floor:
        I.append(Issue("error", "ceiling", "transmission_ceiling must exceed transmission_floor"))
    if pre.air_normalisation:
        r = pre.air_region
        if r is None or len(r) != 4 or r[0] >= r[1] or r[2] >= r[3] or min(r) < 0:
            I.append(Issue("error", "air_region",
                           "air_normalisation requires air_region=(x0,x1,y0,y1) with x0<x1, y0<y1"))
    if pre.bad_pixel_window < 3 or pre.bad_pixel_window % 2 == 0:
        I.append(Issue("error", "bp_window", "bad_pixel_window must be odd and >= 3"))
    if pre.bad_pixel_correction and not (inp.flat_path and (inp.dark_path or not pre.apply_dark)):
        I.append(Issue("error", "bp_needs_cal", "bad-pixel detection needs flat (and dark) images"))
    if pre.batch_size < 1:
        I.append(Issue("error", "batch", "preprocessing.batch_size must be >= 1"))
    _positive(I, "preprocessing.memory_limit_gb", pre.memory_limit_gb)

    # --- reconstruction -------------------------------------------------------
    if rec.algorithm not in SUPPORTED_ALGORITHMS:
        I.append(Issue("error", "algorithm", f"reconstruction.algorithm must be one of {SUPPORTED_ALGORITHMS}"))
    if rec.filter_name.lower() not in SUPPORTED_FILTERS:
        I.append(Issue("error", "filter", f"reconstruction.filter_name must be one of {SUPPORTED_FILTERS}"))
    if rec.filter_cutoff is not None:
        I.append(Issue("error", "filter_cutoff",
                       "filter_cutoff is not exposed: it will be enabled only once the "
                       "installed ASTRA API is verified to support it correctly."))
    if rec.iterations < 1:
        I.append(Issue("error", "iterations", "reconstruction.iterations must be >= 1"))
    if rec.non_negativity and rec.algorithm == "FDK_CUDA":
        I.append(Issue("error", "nonneg_fdk", "Non-negativity is not applicable to FDK_CUDA"))

    # --- artifact corrections ---------------------------------------------------
    bh = art.beam_hardening
    if bh.enabled:
        if bh.coefficients is None or len(bh.coefficients) != 3:
            I.append(Issue("error", "bh_coeff", "beam_hardening.enabled requires 3 coefficients (c1,c2,c3)"))
        if not bh.coefficients_are_calibrated:
            I.append(Issue("warning", "bh_uncal", "Beam-hardening coefficients are NOT flagged as "
                                                  "calibrated: this is an illustrative correction."))
    if art.metal.enabled and art.metal.threshold is None:
        I.append(Issue("error", "metal_threshold", "metal.enabled requires metal.threshold"))
    if art.ring.enabled and art.ring.method not in ("wavelet_fft", "column_median"):
        I.append(Issue("error", "ring_method", "ring.method must be wavelet_fft or column_median"))
    if art.scatter.enabled:
        I.append(Issue("warning", "scatter", "Scatter correction is a separate, approximate model."))

    # --- export -------------------------------------------------------------------
    bad = [f for f in exp.formats if f not in ("tiff32", "tiff16", "vol")]
    if bad:
        I.append(Issue("error", "formats", f"export.formats contains unknown {bad}"))
    if "tiff16" in exp.formats and exp.tiff16_window_mode == "explicit":
        if exp.window_min is None or exp.window_max is None or exp.window_min >= exp.window_max:
            I.append(Issue("error", "window", "tiff16 explicit window needs window_min < window_max"))
    if exp.tiff16_window_mode not in ("explicit", "percentile"):
        I.append(Issue("error", "window_mode", "export.tiff16_window_mode invalid"))
    if not (0 <= exp.percentile_low < exp.percentile_high <= 100):
        I.append(Issue("error", "percentiles", "need 0 <= percentile_low < percentile_high <= 100"))
    return I


def validate_or_raise(cfg: PipelineConfig, *, require_inputs: bool = True) -> List[Issue]:
    """Validate, log warnings, raise ConfigError on any error; return the warnings."""
    issues = validate_config(cfg, require_inputs=require_inputs)
    raise_on_errors(issues, "Invalid configuration")
    for w in issues:
        logger.warning("%s", w)
    return issues
