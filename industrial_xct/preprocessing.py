"""preprocessing.py - from raw detector counts to line integrals.

PROBLEM THIS MODULE SOLVES
    Reconstruction algorithms need the *line integral of the attenuation coefficient* along
    each ray, not detector counts. The Beer-Lambert law says transmitted intensity is
        I = I0 * exp( -integral mu dl )
    so with I0 measured by a flat (open-beam) image and the detector offset by a dark image:
        T = (I_raw - I_dark) / (I_flat - I_dark)        (dimensionless transmission)
        P = -ln(T)                                      (dimensionless line integral)

NUMERICAL MEANING OF EACH STAGE (never interchangeable)
    raw            : integer detector counts (e.g. uint16), unchanged on disk and in memory
    counts_f32     : float32 counts, may be negative after dark subtraction (uint16 -> float32 is exact)
    transmission   : float32, dimensionless, ~1 in air, 0..1 through material (noise can exceed 1)
    line integral  : float32, dimensionless, >= 0 in principle; THIS is the reconstruction input

DATA FLOW
    source.read_batch -> normalise_batch (dark/flat or I0) -> [air flux normalisation]
        -> replace bad/invalid pixels -> beer_lambert -> float32 line integrals
    The raw projections are never modified.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from industrial_xct.config import InputConfig, Issue, PreprocessingConfig
from industrial_xct.io_xtek import (DatStackSource, TiffSequenceSource,
                                    load_reference_image)

logger = logging.getLogger(__name__)
Source = Union[TiffSequenceSource, DatStackSource]


class PreprocessingError(RuntimeError):
    """Preprocessing cannot proceed safely."""


# --------------------------------------------------------------------------- #
# Calibration data
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    """Calibration images (float32 counts) and derived static masks."""
    dark: Optional[np.ndarray] = None
    flat: Optional[np.ndarray] = None
    denominator: Optional[np.ndarray] = None          # flat - dark
    invalid_denominator: Optional[np.ndarray] = None  # bool (H, W)
    bad_pixel_mask: Optional[np.ndarray] = None       # bool (H, W) from detect_bad_pixels
    i0: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)


def load_calibration(inp: InputConfig, pre: PreprocessingConfig,
                     image_shape: Tuple[int, int]) -> Calibration:
    """Load dark/flat images, validate their shape, derive masks. Raises on inconsistency."""
    cal = Calibration()
    if pre.normalisation_mode == "scalar_i0":
        cal.i0 = pre.i0_value
    if pre.normalisation_mode == "dark_flat":
        if not inp.flat_path:
            raise PreprocessingError("dark_flat mode requires a flat-field image")
        cal.flat, cal.info["flat"] = load_reference_image(inp.flat_path)
    if pre.apply_dark and pre.normalisation_mode in ("dark_flat", "scalar_i0") and inp.dark_path:
        cal.dark, cal.info["dark"] = load_reference_image(inp.dark_path)
    elif pre.normalisation_mode == "dark_flat" and pre.apply_dark:
        raise PreprocessingError("apply_dark=true but no dark image path was given")
    for name in ("dark", "flat"):
        a = getattr(cal, name)
        if a is not None and a.shape != tuple(image_shape):
            raise PreprocessingError(f"{name} image shape {a.shape} != projection shape {tuple(image_shape)}")
    if pre.normalisation_mode == "dark_flat":
        cal.denominator = cal.flat - (cal.dark if cal.dark is not None else 0.0)
        cal.invalid_denominator = ~(cal.denominator >= pre.min_denominator_counts)  # also catches NaN
        n = int(cal.invalid_denominator.sum())
        if n:
            cal.issues.append(Issue("warning", "invalid_denominator",
                                    f"{n} pixel(s) with flat-dark < {pre.min_denominator_counts}: "
                                    f"cannot be normalised; they will be replaced from neighbours."))
        if pre.bad_pixel_correction:
            cal.bad_pixel_mask, stats = detect_bad_pixels(cal.dark, cal.flat, pre)
            cal.info["bad_pixels"] = stats
    return cal


# --------------------------------------------------------------------------- #
# Bad pixels
# --------------------------------------------------------------------------- #
def detect_bad_pixels(dark: Optional[np.ndarray], flat: np.ndarray,
                      cfg: PreprocessingConfig) -> Tuple[np.ndarray, Dict[str, int]]:
    """Static bad-pixel mask from calibration images.

    gain = flat - dark. A pixel is
      * dead/abnormal if gain / local_median(gain, 5x5) lies outside bad_pixel_gain_ratio_range
        (a *local* ratio, so smooth beam-profile variation is not flagged);
      * hot if dark - local_median(dark, 5x5) exceeds hot_mad_factor * 1.4826 * MAD(residual).
    The mask is for diagnostics AND replacement; inspect it before enabling correction.
    """
    gain = flat - (dark if dark is not None else 0.0)
    med = ndimage.median_filter(gain, size=5, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = gain / np.where(med > 0, med, np.nan)
    lo, hi = cfg.bad_pixel_gain_ratio_range
    abnormal = ~((ratio >= lo) & (ratio <= hi))                    # NaN -> abnormal
    hot = np.zeros_like(abnormal)
    if dark is not None:
        res = dark - ndimage.median_filter(dark, size=5, mode="nearest")
        mad = 1.4826 * np.median(np.abs(res - np.median(res)))
        hot = res > cfg.bad_pixel_hot_mad_factor * max(float(mad), 1e-6)
    mask = abnormal | hot
    return mask, {"abnormal_gain": int(abnormal.sum()), "hot_dark": int(hot.sum()), "total": int(mask.sum())}


def replace_masked_pixels(stack: np.ndarray, mask: np.ndarray, window: int = 3,
                          max_window: int = 9) -> int:
    """In place: replace pixels where mask is True by the median of their *valid* neighbours.

    Only masked pixels change; everything else is untouched (no smoothing). The window grows
    (3, 5, 7, 9) for pixels whose whole neighbourhood is masked. Returns the number of masked
    pixels that could NOT be filled (0 on success). ``stack`` is (B, H, W) float32; ``mask`` (H, W).
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0
    H, W = mask.shape
    todo = np.ones(ys.size, bool)
    for w in range(window, max_window + 1, 2):
        if not todo.any():
            break
        r = w // 2
        idx = np.nonzero(todo)[0]
        yy, xx = ys[idx], xs[idx]
        cols = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                y2, x2 = yy + dy, xx + dx
                inb = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
                y2c, x2c = np.clip(y2, 0, H - 1), np.clip(x2, 0, W - 1)
                ok = inb & ~mask[y2c, x2c]
                v = stack[:, y2c, x2c].astype(np.float32)          # (B, n)
                v[:, ~ok] = np.nan
                cols.append(v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN slices handled below
            med = np.nanmedian(np.stack(cols, axis=2), axis=2)      # (B, n)
        done = np.all(np.isfinite(med), axis=0)
        stack[:, yy[done], xx[done]] = med[:, done]
        todo[idx[done]] = False
    return int(todo.sum())


# --------------------------------------------------------------------------- #
# Normalisation, air region, logarithm
# --------------------------------------------------------------------------- #
def normalise_batch(raw: np.ndarray, cal: Calibration, cfg: PreprocessingConfig) -> np.ndarray:
    """Raw counts -> transmission (float32). Invalid-denominator pixels are set to NaN.

    dark_flat : T = (I - D) / (F - D)     (F-D < min_denominator_counts => invalid)
    scalar_i0 : T = (I - D) / I0           (I0 must be user-verified)
    none      : input already transmission (float only)
    """
    if raw.ndim != 3:
        raise PreprocessingError(f"expected a (B, H, W) batch, got {raw.shape}")
    x = raw.astype(np.float32)           # uint16 -> float32 is exact; original array untouched
    if cfg.normalisation_mode == "none":
        if np.issubdtype(raw.dtype, np.integer):
            raise PreprocessingError("normalisation_mode='none' needs float transmission data, not integer counts")
        return x
    if cal.dark is not None:
        x -= cal.dark
    if cfg.normalisation_mode == "scalar_i0":
        if not cal.i0 or cal.i0 <= 0:
            raise PreprocessingError("scalar_i0 mode needs a positive I0")
        x /= np.float32(cal.i0)
        return x
    den = np.where(cal.invalid_denominator, np.float32(1.0), cal.denominator).astype(np.float32)
    x /= den
    x[:, cal.invalid_denominator] = np.nan
    return x


@dataclass
class AirRegionReport:
    ok: bool
    factors: np.ndarray                 # per-projection median transmission in the region
    reasons: List[str]


def verify_air_region(t: np.ndarray, region: Tuple[int, int, int, int],
                      min_transmission: float, max_cv: float) -> AirRegionReport:
    """Check that ``region`` really is unattenuated air in every projection of the batch.

    A real air region has median transmission near 1 and low spatial variation. If the
    region ever dips (sample, holder, scatter halo), its change would be mistaken for flux
    drift -- so we refuse to normalise instead of guessing.
    """
    x0, x1, y0, y1 = region
    H, W = t.shape[1:]
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise PreprocessingError(f"air_region {region} lies outside the {W}x{H} detector")
    r = t[:, y0:y1, x0:x1].reshape(t.shape[0], -1)
    med = np.nanmedian(r, axis=1)
    cv = np.nanstd(r, axis=1) / np.maximum(med, 1e-12)
    reasons: List[str] = []
    if np.any(med < min_transmission):
        reasons.append(f"median transmission drops to {np.nanmin(med):.3f} < {min_transmission} "
                       f"(region not unattenuated in {int(np.sum(med < min_transmission))} projection(s))")
    if np.any(cv > max_cv):
        reasons.append(f"spatial CV up to {np.nanmax(cv):.3f} > {max_cv} (structure inside region)")
    return AirRegionReport(not reasons, med.astype(np.float64), reasons)


def beer_lambert(t: np.ndarray, floor: float, ceiling: Optional[float]) -> Tuple[np.ndarray, int, int]:
    """P = -ln(T) in place on float32, after clamping T to [floor, ceiling].

    Clamping to a positive floor prevents -ln(0)=inf and -ln(negative)=NaN. Clamped pixels are
    COUNTED (n_low, n_high) so you can see how much information was replaced. Note: a ceiling
    (e.g. 1.0) biases noise in air upward; the default is no ceiling.
    """
    if not np.all(np.isfinite(t)):
        raise PreprocessingError("non-finite transmission reached the logarithm; fill them first")
    n_low = int(np.count_nonzero(t < floor))
    n_high = 0
    if ceiling is not None:
        n_high = int(np.count_nonzero(t > ceiling))
        np.minimum(t, np.float32(ceiling), out=t)
    np.maximum(t, np.float32(floor), out=t)
    np.log(t, out=t)
    np.negative(t, out=t)
    return t, n_low, n_high


# --------------------------------------------------------------------------- #
# Batch / stack pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessStats:
    n_projections: int = 0
    n_clamped_low: int = 0
    n_clamped_high: int = 0
    n_invalid_den_pixels: int = 0
    n_bad_pixels_replaced: int = 0
    n_nonfinite_replaced: int = 0
    n_unfilled: int = 0
    air_factors: List[float] = field(default_factory=list)
    air_factor_projection_index: List[int] = field(default_factory=list)
    air_region_ok: Optional[bool] = None
    units: str = "line integral P = -ln(T), dimensionless, float32"


def preprocess_batch(raw: np.ndarray, cal: Calibration, cfg: PreprocessingConfig,
                     first_index: int = 0, stats: Optional[PreprocessStats] = None
                     ) -> Tuple[np.ndarray, PreprocessStats]:
    """One batch of raw projections -> float32 line integrals (raw is not modified)."""
    stats = stats or PreprocessStats()
    t = normalise_batch(raw, cal, cfg)
    if cfg.air_normalisation:
        rep = verify_air_region(t, cfg.air_region, cfg.air_min_transmission, cfg.air_max_spatial_cv)
        if not rep.ok:
            raise PreprocessingError(f"air region {cfg.air_region} rejected (batch from projection "
                                     f"{first_index}): " + "; ".join(rep.reasons))
        t /= rep.factors.astype(np.float32)[:, None, None]       # flux correction: T' = T / T_air
        stats.air_factors += rep.factors.tolist()
        stats.air_factor_projection_index += list(range(first_index, first_index + t.shape[0]))
        stats.air_region_ok = True
    static = np.zeros(t.shape[1:], bool)
    if cal.invalid_denominator is not None:
        static |= cal.invalid_denominator
        stats.n_invalid_den_pixels = int(cal.invalid_denominator.sum())
    if cfg.bad_pixel_correction and cal.bad_pixel_mask is not None:
        static |= cal.bad_pixel_mask
    if static.any():
        unfilled = replace_masked_pixels(t, static, window=cfg.bad_pixel_window)
        if unfilled:
            raise PreprocessingError(f"{unfilled} masked pixel(s) have no valid neighbour within 9x9")
        stats.n_bad_pixels_replaced = int(static.sum())
    nonfinite = ~np.isfinite(t)
    if nonfinite.any():
        for b in np.nonzero(nonfinite.reshape(t.shape[0], -1).any(axis=1))[0]:
            m = nonfinite[b]
            if replace_masked_pixels(t[b:b + 1], m, window=cfg.bad_pixel_window):
                raise PreprocessingError(f"projection {first_index + int(b)}: non-finite pixels cannot be filled")
            stats.n_nonfinite_replaced += int(m.sum())
    p, lo, hi = beer_lambert(t, cfg.transmission_floor, cfg.transmission_ceiling)
    stats.n_clamped_low += lo
    stats.n_clamped_high += hi
    stats.n_projections += raw.shape[0]
    return p, stats


def estimate_memory(n: int, h: int, w: int, batch_size: int, out_itemsize: int = 4) -> Dict[str, float]:
    """Bytes for the full float32 output and for the per-batch working set (~5 float32 copies)."""
    return {"output_bytes": float(n) * h * w * out_itemsize,
            "working_bytes": 5.0 * batch_size * h * w * 4}


def preprocess_stack(source: Source, cal: Calibration, cfg: PreprocessingConfig
                     ) -> Tuple[np.ndarray, PreprocessStats, Dict[str, Any]]:
    """Preprocess every projection. Output is in RAM if it fits memory_limit_gb, else a
    float32 .npy memmap in cfg.work_dir (error if work_dir is not set)."""
    n = len(source)
    h, w = source.image_shape
    mem = estimate_memory(n, h, w, cfg.batch_size)
    limit = cfg.memory_limit_gb * 1e9
    in_ram = mem["output_bytes"] + mem["working_bytes"] <= limit
    out_path = None
    if in_ram:
        out: np.ndarray = np.empty((n, h, w), np.float32)
    else:
        if not cfg.work_dir:
            raise PreprocessingError(
                f"Preprocessed stack needs {mem['output_bytes'] / 1e9:.1f} GB + "
                f"{mem['working_bytes'] / 1e9:.1f} GB working memory > memory_limit_gb={cfg.memory_limit_gb}. "
                f"Set preprocessing.work_dir to spill to disk, or raise the limit.")
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(cfg.work_dir) / "preprocessed_line_integrals_f32.npy"
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, h, w))
    stats = PreprocessStats()
    for start, raw in source.iter_batches(cfg.batch_size):
        p, stats = preprocess_batch(raw, cal, cfg, start, stats)
        out[start:start + p.shape[0]] = p
    if not in_ram:
        out.flush()
    info = {"in_ram": in_ram, "memmap_path": str(out_path) if out_path else None, **mem}
    return out, stats, info
