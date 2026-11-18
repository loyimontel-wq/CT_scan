"""artifacts.py - geometry calibration, centre-of-rotation estimation, artifact corrections.

PROBLEM THIS MODULE SOLVES
    Real scanners are never perfectly aligned. The rotation axis does not project onto the
    exact centre column of the detector, the detector can be rolled or tilted, individual
    detector columns have gain errors (rings), the polychromatic beam hardens (cupping) and
    metal blocks the beam. This module provides (a) the explicit vector description of the
    geometry (ASTRA 'cone_vec' numbers, no ASTRA import needed), (b) an image-based sub-pixel
    CoR estimate with consistency checks, and (c) optional, individually switchable corrections.

DATA FLOW
    ScannerGeometry + angles + GeometryCalibration --build_cone_vectors--> (N, 12) vectors
    preprocessed line integrals (N, H, W) --estimate_cor--> CoREstimate --> GeometryCalibration
    preprocessed stack --suppress_rings / apply_beam_hardening / metal trace--> corrected stack

THE CoR MATHS (read this one carefully)
    Index convention: detector columns are array indices c = 0..W-1; the *geometric centre*
    of the image is c0 = (W-1)/2. Let c_a be the column where the rotation axis projects.
        delta = c_a - c0          (CoR offset in detector pixels; >0 = axis right of centre)
    Projections taken 180 degrees apart see the object mirrored about the axis, so for the
    same detector row v:   P_theta(c) = P_{theta+180}(2*c_a - c).
    Let G(c) = P_{theta+180}(W-1-c) be the column-flipped opposite view. Then
        P_theta(c) = G(c + s)   with   s = (W-1) - 2*c_a = -2*delta.
    Registering G onto P_theta (skimage returns the shift t that must be APPLIED to G) gives
    t = -s = 2*delta, hence   delta = t / 2.
    Physical size: detector-plane offset = delta*pitch (mm); the same offset at the
    rotation axis is delta*pitch/M = delta*voxel (M = SDD/SOD). It is a *horizontal* shift
    only; roll, tilt and axis tilt are separate parameters (see GeometryCalibration).
    LIMITS: the mirror relation is exact for parallel beams and approximate for cone beams
    (magnification differs for features nearer to / farther from the source), so one pair is
    never trusted alone; many pairs and row bands are compared and a reconstruction sweep is
    the final check.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from industrial_xct.config import (BeamHardeningConfig, CoRConfig, Issue,
                                   MetalConfig, RingConfig)

logger = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """A correction or calibration cannot be performed safely."""


class CoRError(ArtifactError):
    """CoR estimation failed or the data cannot support a reliable estimate."""


# =========================================================================== #
# A. Geometry description (ASTRA cone_vec)
# =========================================================================== #
@dataclass(frozen=True)
class GeometryCalibration:
    """Everything needed to write one ASTRA ``cone_vec`` row per projection.

    Lengths in mm, angles in degrees. Scanner -> ASTRA frame (same as ASTRA's own
    ``geom_2vec`` for a 'cone' geometry, verified against the installed ASTRA source):
        world z = rotation axis; at theta=0 the source is at (0, -SOD, 0), the detector
        centre at (0, SDD-SOD, 0); u (image column direction) = (cos t, sin t, 0);
        v (image row direction) = (0, 0, 1).
    theta = rotation_sign * angle_table_deg + angle_offset_deg.
    """
    sod_mm: float
    sdd_mm: float
    pixel_u_mm: float
    pixel_v_mm: float
    n_rows: int
    n_cols: int
    cor_offset_px: float = 0.0           # delta (see module docstring), image-column pixels
    detector_offset_v_px: float = 0.0    # row of the mid-plane relative to (H-1)/2
    detector_roll_deg: float = 0.0       # in-plane rotation of the detector about its normal
    detector_tilt_deg: float = 0.0       # rotation of the detector about its own u axis
    rotation_sign: int = 1
    angle_offset_deg: float = 0.0
    detector_u_sign: int = 1             # +1: increasing image column = +u direction
    detector_v_sign: int = -1            # -1: image row 0 is the top (+z)

    def validate(self) -> None:
        if not (self.sod_mm > 0 and self.sdd_mm > self.sod_mm):
            raise ArtifactError(f"need 0 < SOD < SDD, got SOD={self.sod_mm}, SDD={self.sdd_mm}")
        if not (self.pixel_u_mm > 0 and self.pixel_v_mm > 0 and self.n_rows > 0 and self.n_cols > 0):
            raise ArtifactError("detector pixel pitch and size must be positive")
        for n in ("rotation_sign", "detector_u_sign", "detector_v_sign"):
            if getattr(self, n) not in (-1, 1):
                raise ArtifactError(f"{n} must be +1 or -1")


def _rotate(vec: np.ndarray, axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of (N,3) vectors about (N,3) unit axes (right-hand rule)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return (vec * c + np.cross(axis_unit, vec) * s +
            axis_unit * np.sum(axis_unit * vec, axis=1, keepdims=True) * (1 - c))


def build_cone_vectors(angles_deg: Sequence[float], cal: GeometryCalibration) -> np.ndarray:
    """(N, 12) ASTRA vectors: [source(3), detector_centre(3), u(3), v(3)] per projection.

    Detector centre with the CoR applied EXACTLY ONCE:
        d = d0 - delta*u - eps*v
    because pixel column c sits at d + (c - (W-1)/2)*u, and the axis (which projects to d0)
    must land on column (W-1)/2 + delta, i.e. d0 = d + delta*u. Same for rows with eps.
    Roll rotates (u, v) about the detector normal; tilt rotates v about u.
    """
    cal.validate()
    a = np.asarray(angles_deg, dtype=np.float64)
    th = cal.rotation_sign * np.deg2rad(a) + np.deg2rad(cal.angle_offset_deg)
    s, c = np.sin(th), np.cos(th)
    z, one = np.zeros_like(th), np.ones_like(th)
    odd = cal.sdd_mm - cal.sod_mm
    src = np.stack([s * cal.sod_mm, -c * cal.sod_mm, z], 1)
    d0 = np.stack([-s * odd, c * odd, z], 1)
    n = np.stack([-s, c, z], 1)                                  # unit, source -> detector
    u = cal.detector_u_sign * cal.pixel_u_mm * np.stack([c, s, z], 1)
    v = cal.detector_v_sign * cal.pixel_v_mm * np.stack([z, z, one], 1)
    if cal.detector_roll_deg:
        r = np.deg2rad(cal.detector_roll_deg)
        u, v = _rotate(u, n, r), _rotate(v, n, r)
    if cal.detector_tilt_deg:
        u_hat = u / np.linalg.norm(u, axis=1, keepdims=True)
        v = _rotate(v, u_hat, np.deg2rad(cal.detector_tilt_deg))
    d = d0 - cal.cor_offset_px * u - cal.detector_offset_v_px * v
    return np.concatenate([src, d, u, v], axis=1)


def project_point(vectors: np.ndarray, point_xyz: Sequence[float], n_rows: int, n_cols: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Where a 3-D point lands on the detector for every view: (row_index, col_index).

    Independent of ASTRA (plane/ray intersection); used for tests and diagnostics. The
    origin of the volume must land on column (W-1)/2 + delta for a correct CoR convention.
    """
    P = np.asarray(point_xyz, float)
    src, d, u, v = vectors[:, 0:3], vectors[:, 3:6], vectors[:, 6:9], vectors[:, 9:12]
    nrm = np.cross(u, v)
    t = np.sum((d - src) * nrm, 1) / np.sum((P - src) * nrm, 1)
    rel = src + t[:, None] * (P - src) - d
    uu, vv, uv = np.sum(u * u, 1), np.sum(v * v, 1), np.sum(u * v, 1)
    ru, rv = np.sum(rel * u, 1), np.sum(rel * v, 1)
    det = uu * vv - uv ** 2
    a = (ru * vv - rv * uv) / det
    b = (rv * uu - ru * uv) / det
    return (n_rows - 1) / 2 + b, (n_cols - 1) / 2 + a


# =========================================================================== #
# B. Centre-of-rotation estimation
# =========================================================================== #
@dataclass
class PairSelection:
    i: int
    j: int
    angle_i_deg: float
    angle_j_deg: float
    separation_deg: float
    residual_deg: float                  # |separation - 180|


@dataclass
class CoREstimate:
    accepted: bool
    delta_px: Optional[float]
    delta_detector_mm: Optional[float]
    delta_object_mm: Optional[float]
    method: str
    n_pairs_used: int
    pair_estimates_px: List[float]       # one per pair (median over row bands)
    pairs: List[PairSelection]
    band_centres_px: List[float]         # detector row centres
    band_estimates_px: List[float]       # one per band (median over pairs)
    pair_std_px: Optional[float]
    row_slope_px_per_row: Optional[float]
    row_variation_px: Optional[float]    # |slope| * detector height
    vertical_shift_px: Optional[float]   # mean registered dy (should be ~0)
    issues: List[Issue] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["pairs"] = [p.__dict__ for p in self.pairs]
        d["issues"] = [str(i) for i in self.issues]
        return d


def cor_physical(delta_px: float, pixel_u_mm: float, magnification: float) -> Tuple[float, float]:
    """(offset on the detector plane in mm, same offset at the rotation axis in mm)."""
    return delta_px * pixel_u_mm, delta_px * pixel_u_mm / magnification


def select_opposing_pairs(unwrapped_deg: np.ndarray, n_pairs: int, tolerance_deg: float
                          ) -> List[PairSelection]:
    """Choose projection pairs ~180 deg apart using the REAL angles (not index arithmetic).

    For each candidate i in the first half of the scan the partner j minimises
    | |theta_j - theta_i| - 180 |. Pairs whose residual exceeds ``tolerance_deg`` are rejected
    (reported by the caller). Pairs are spread evenly over the scan.
    """
    th = np.asarray(unwrapped_deg, float)
    if np.abs(th[-1] - th[0]) < 180.0 - tolerance_deg:
        raise CoRError(f"angular span {abs(th[-1] - th[0]):.2f} deg < 180 deg: no opposing pairs exist")
    rel = np.abs(th - th[0])
    first_half = np.nonzero(rel < 180.0 - tolerance_deg)[0]
    cand: List[PairSelection] = []
    for i in first_half:
        sep = np.abs(th - th[i])
        j = int(np.argmin(np.abs(sep - 180.0)))
        res = float(abs(sep[j] - 180.0))
        if res <= tolerance_deg and j != i:
            cand.append(PairSelection(int(i), j, float(th[i]), float(th[j]), float(sep[j]), res))
    if not cand:
        raise CoRError(f"no projection pair within {tolerance_deg} deg of 180 deg separation")
    pick = np.unique(np.linspace(0, len(cand) - 1, min(n_pairs, len(cand))).round().astype(int))
    return [cand[k] for k in pick]


def _register_pair(ref: np.ndarray, mov: np.ndarray, upsample: int) -> Tuple[float, float]:
    """(shift_x, shift_y) in pixels to apply to ``mov`` so it matches ``ref`` (sub-pixel).

    Windowed (Hann) to suppress edge discontinuities, mean removed, plain (un-whitened)
    cross-correlation: whitening amplifies noise on low-contrast line-integral images.
    """
    from skimage.filters import window
    from skimage.registration import phase_cross_correlation
    w = window("hann", ref.shape)
    a, b = (ref - ref.mean()) * w, (mov - mov.mean()) * w
    shift, _err, _ = phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)
    return float(shift[1]), float(shift[0])


def estimate_cor(stack: np.ndarray, unwrapped_deg: np.ndarray, cfg: CoRConfig, *,
                 pixel_u_mm: float, magnification: float,
                 min_band_contrast: float = 1e-3) -> CoREstimate:
    """Sub-pixel CoR offset delta from opposing projection pairs and row bands.

    ``stack`` is (N, H, W) line integrals (preprocessed, NOT ring-corrected). For each pair
    and each row band the column-flipped opposite view is registered onto the first view;
    delta = shift_x / 2 (derivation in the module docstring). Acceptance requires
      * at least cfg.min_pairs valid pairs,
      * pair-to-pair standard deviation <= cfg.consistency_tolerance_px,
      * change across rows (slope * H) <= cfg.row_variation_tolerance_px.
    If any test fails the estimate is returned with accepted=False and a reason; callers must
    NOT use it silently.
    """
    N, H, W = stack.shape
    if cfg.method != "phase_cross_correlation":
        raise CoRError(f"unsupported CoR method {cfg.method!r}")
    issues: List[Issue] = []
    pairs = select_opposing_pairs(unwrapped_deg, cfg.n_pairs, cfg.pair_angle_tolerance_deg)
    margin = min(cfg.rows_margin_px, H // 4)
    edges = np.linspace(margin, H - margin, cfg.n_row_bands + 1).astype(int)
    bands = [(int(edges[k]), int(edges[k + 1])) for k in range(cfg.n_row_bands)]
    est = np.full((len(pairs), len(bands)), np.nan)
    dys: List[float] = []
    for p_i, p in enumerate(pairs):
        A, B = np.asarray(stack[p.i], np.float32), np.asarray(stack[p.j], np.float32)[:, ::-1]
        for b_i, (r0, r1) in enumerate(bands):
            ra, rb = A[r0:r1], B[r0:r1]
            if ra.std() < min_band_contrast or rb.std() < min_band_contrast:
                continue                                           # nothing but air: no information
            sx, sy = _register_pair(ra, rb, cfg.upsample_factor)
            est[p_i, b_i] = sx / 2.0                               # delta = t / 2
            dys.append(sy)
    centres = [0.5 * (r0 + r1 - 1) for r0, r1 in bands]
    pair_med = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in est])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        band_med = np.array([np.nanmedian(est[:, k]) if np.isfinite(est[:, k]).any() else np.nan
                             for k in range(len(bands))])
    valid_pairs = np.isfinite(pair_med)
    if not valid_pairs.any():
        raise CoRError("no pair/band contained enough contrast (all air?) to register")
    delta = float(np.nanmedian(est))
    pair_std = float(np.std(pair_med[valid_pairs])) if valid_pairs.sum() > 1 else None
    ok_b = np.isfinite(band_med)
    slope = variation = None
    if ok_b.sum() >= 2:
        slope = float(np.polyfit(np.asarray(centres)[ok_b], band_med[ok_b], 1)[0])
        variation = abs(slope) * H
    reasons: List[str] = []
    if valid_pairs.sum() < cfg.min_pairs:
        reasons.append(f"only {int(valid_pairs.sum())} valid pair(s) < min_pairs={cfg.min_pairs}")
    if pair_std is None or pair_std > cfg.consistency_tolerance_px:
        reasons.append(f"pair-to-pair std {pair_std} px exceeds tolerance {cfg.consistency_tolerance_px}")
    if variation is not None and variation > cfg.row_variation_tolerance_px:
        reasons.append(f"CoR changes by {variation:.3f} px across the detector rows (> "
                       f"{cfg.row_variation_tolerance_px}): suggests rotation-axis tilt / detector roll; "
                       f"a single horizontal offset does not describe the geometry")
    if ok_b.sum() < 2:
        issues.append(Issue("warning", "single_band", "fewer than 2 row bands had contrast: "
                                                      "row dependence (tilt) could not be checked"))
    worst = max(p.residual_deg for p in pairs)
    notes = [f"{int(valid_pairs.sum())}/{len(pairs)} pairs usable; max angular residual from 180 deg: {worst:.3f} deg",
             "Opposite-view mirror symmetry is approximate for cone beams; confirm with a reconstruction sweep."]
    d_det, d_obj = cor_physical(delta, pixel_u_mm, magnification)
    for r in reasons:
        issues.append(Issue("error", "cor_rejected", r))
    return CoREstimate(not reasons, delta, d_det, d_obj, "phase_cross_correlation(hann, un-whitened) on opposing pairs",
                       int(valid_pairs.sum()), [float(x) for x in pair_med], pairs, centres,
                       [float(x) for x in band_med], pair_std, slope, variation,
                       float(np.mean(dys)) if dys else None, issues, notes)


def compare_cor_to_metadata(delta_px: float, fields: Dict[str, float], pixel_u_mm: float,
                            magnification: float) -> List[Issue]:
    """Show what each metadata number WOULD mean as an offset under different unit hypotheses.

    No unit is assumed. For a field value x: if pixels -> x px; if mm on the detector -> x/pitch
    px; if mm at the object -> x*M/pitch px. Compare these with the image-based delta; the
    hypothesis that matches (if any) is evidence, not proof.
    """
    out = []
    for name, x in fields.items():
        out.append(Issue("info", "cor_metadata_hypotheses",
                         f"{name}={x:g}: as px -> {x:+.3f}; as mm@detector -> {x / pixel_u_mm:+.3f}; "
                         f"as mm@object -> {x * magnification / pixel_u_mm:+.3f} px "
                         f"(image-based delta = {delta_px:+.3f} px; sign conventions unverified)"))
    return out


# =========================================================================== #
# C. Ring / stripe suppression
# =========================================================================== #
def column_stripe_metric(sino: np.ndarray, window: int = 31) -> float:
    """RMS of the high-pass part of the column-mean profile of a sinogram (A, W)."""
    prof = sino.mean(axis=0)
    hp = prof - ndimage.median_filter(prof, size=window, mode="nearest")
    return float(np.sqrt(np.mean(hp ** 2)))


def remove_stripes_wavelet_fft(sino: np.ndarray, wavelet: str = "db5", levels: int = 4,
                               sigma: float = 2.0) -> np.ndarray:
    """Wavelet-Fourier stripe removal (Muench et al., Opt. Express 17, 2009).

    Sinogram axes: (angle, detector column). A defective detector column is constant along the
    angle axis => a *vertical* stripe, which lives in the vertical-detail wavelet band (cV).
    For each level the cV band is Fourier-transformed along the angle axis and its low
    frequencies (the stripe content) are damped by 1 - exp(-f^2 / (2 sigma^2)); f counts FFT
    bins from zero frequency. Assumption: genuine object features vary with angle, stripes do
    not. Larger sigma removes more (and risks removing true low-frequency angular structure).
    """
    import pywt
    A, W = sino.shape
    coeffs = pywt.wavedec2(sino.astype(np.float64), wavelet, level=levels, mode="symmetric")
    out = [coeffs[0]]
    for cH, cV, cD in coeffs[1:]:
        n = cV.shape[0]
        f = np.fft.fftshift(np.fft.fft(cV, axis=0), axes=0)
        y = np.arange(n) - n // 2
        damp = 1.0 - np.exp(-(y ** 2) / (2.0 * sigma ** 2))
        cV2 = np.real(np.fft.ifft(np.fft.ifftshift(f * damp[:, None], axes=0), axis=0))
        out.append((cH, cV2, cD))
    rec = pywt.waverec2(out, wavelet, mode="symmetric")
    return rec[:A, :W].astype(sino.dtype)


def remove_stripes_column_median(sino: np.ndarray, window: int = 31) -> np.ndarray:
    """Subtract the high-pass part of the per-column mean (detector-column gain error).

    Assumption: the angle-averaged object signal is smooth over ``window`` columns. Sharp
    genuine features in the averaged profile (e.g. a thin wall at the centre) are partly
    removed: check the before/after diagnostics.
    """
    prof = sino.mean(axis=0)
    hp = prof - ndimage.median_filter(prof, size=window, mode="nearest")
    return (sino - hp[None, :]).astype(sino.dtype)


def suppress_rings(stack: np.ndarray, cfg: RingConfig, rows: Optional[range] = None
                   ) -> Dict[str, Any]:
    """In-place stripe suppression on every sinogram stack[:, row, :]. Returns diagnostics."""
    N, H, W = stack.shape
    rows = rows if rows is not None else range(H)
    before: List[float] = []
    after: List[float] = []
    sample = set(np.linspace(rows.start, rows.stop - 1, 5).astype(int).tolist())
    for r in rows:
        s = np.array(stack[:, r, :], dtype=np.float32)
        if r in sample:
            before.append(column_stripe_metric(s, cfg.window_px))
        s2 = (remove_stripes_wavelet_fft(s, cfg.wavelet, cfg.levels, cfg.sigma)
              if cfg.method == "wavelet_fft" else remove_stripes_column_median(s, cfg.window_px))
        if r in sample:
            after.append(column_stripe_metric(s2, cfg.window_px))
        stack[:, r, :] = s2
    return {"method": cfg.method, "sample_rows": sorted(sample),
            "stripe_rms_before": before, "stripe_rms_after": after}


# =========================================================================== #
# D. Beam hardening (polynomial in the line integral)
# =========================================================================== #
def check_polynomial_monotonic(coeffs: Tuple[float, float, float], p_max: float) -> None:
    """P_c = c1 P + c2 P^2 + c3 P^3 must be strictly increasing on [0, p_max], otherwise the
    'correction' would invert contrast somewhere."""
    c1, c2, c3 = coeffs
    p = np.linspace(0.0, p_max, 512)
    if np.any(c1 + 2 * c2 * p + 3 * c3 * p ** 2 <= 0):
        raise ArtifactError(f"beam-hardening polynomial {coeffs} is not monotonically increasing on "
                            f"[0, {p_max:g}]; refusing to apply")


def apply_beam_hardening(p: np.ndarray, cfg: BeamHardeningConfig, p_max: Optional[float] = None
                         ) -> np.ndarray:
    """P_corrected = c1 P + c2 P^2 + c3 P^3 (in place, float32).

    Coefficients are NEVER invented here: they must come from a calibration (step wedge /
    reference object of the same material and spectrum). ``coefficients_are_calibrated`` only
    records what you assert; an uncalibrated set is an illustrative correction.
    """
    if not cfg.enabled:
        return p
    if cfg.coefficients is None or len(cfg.coefficients) != 3:
        raise ArtifactError("beam hardening enabled without three coefficients")
    pm = float(p_max if p_max is not None else np.max(p))
    check_polynomial_monotonic(tuple(cfg.coefficients), max(pm, 1e-6))
    c1, c2, c3 = (np.float32(c) for c in cfg.coefficients)
    q = p.copy() if p.base is not None and not p.flags.writeable else p
    q[...] = c1 * p + c2 * p * p + c3 * p * p * p
    return q


# =========================================================================== #
# E. Scatter (not implemented, on purpose) and F. metal trace interpolation
# =========================================================================== #
def apply_scatter_correction(*_args: Any, **_kwargs: Any) -> None:
    """Scatter is not modelled. A scalar offset is not a scatter correction; a real one needs
    a measured scatter kernel or beam-stop data."""
    raise ArtifactError("scatter correction is not implemented: no validated model without "
                        "scatter measurements")


def metal_trace_interpolate(sino: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
    """Linear interpolation of metal traces (Kalender-style), per projection.

    Trace = sinogram pixels with P > threshold. For each projection (row of the (angle, column)
    sinogram) the masked columns are replaced by linear interpolation between the nearest
    unmasked neighbours (nearest value at the borders). WARNING: this REPLACES measured data
    with an invented smooth profile; it removes streak sources but creates errors near the metal
    and cannot recover true structure. Returns (corrected sinogram, replaced fraction).
    """
    mask = sino > threshold
    if not mask.any():
        return sino, 0.0
    out = sino.copy()
    x = np.arange(sino.shape[1])
    for a in np.nonzero(mask.any(axis=1))[0]:
        m = mask[a]
        if m.all():
            continue                                          # whole projection masked: leave as is
        out[a, m] = np.interp(x[m], x[~m], sino[a, ~m])
    return out, float(mask.mean())


def apply_metal_trace(stack: np.ndarray, cfg: MetalConfig) -> Dict[str, Any]:
    """In place on every sinogram; returns the mean replaced fraction."""
    if not cfg.enabled:
        return {"enabled": False}
    if cfg.threshold is None:
        raise ArtifactError("metal correction needs a threshold (line-integral units)")
    fr = []
    for r in range(stack.shape[1]):
        s = np.array(stack[:, r, :], dtype=np.float32)
        s2, f = metal_trace_interpolate(s, cfg.threshold)
        if f:
            stack[:, r, :] = s2
        fr.append(f)
    return {"enabled": True, "threshold": cfg.threshold, "mean_replaced_fraction": float(np.mean(fr)),
            "max_replaced_fraction_in_a_sinogram": float(np.max(fr))}


# =========================================================================== #
# G. Image-quality indicators for CoR sweeps (descriptive, not selectors)
# =========================================================================== #
def slice_metrics(img: np.ndarray) -> Dict[str, float]:
    """Descriptive indicators of a reconstructed slice.

    * gradient_energy : mean |grad|^2 / (variance + eps)  (higher = sharper edges)
    * total_variation : mean |grad|
    * negative_fraction: share of values below -3 robust sigma (streak/halo indicator)
    A wrong CoR blurs edges, doubles them and adds arcs. These numbers support, but do not
    replace, visual inspection and the geometric consistency tests: the sharpest slice is not
    automatically the right CoR.
    """
    a = np.asarray(img, np.float64)
    gy, gx = np.gradient(a)
    g2 = gx ** 2 + gy ** 2
    mad = 1.4826 * np.median(np.abs(a - np.median(a)))
    return {"gradient_energy": float(g2.mean() / (a.var() + 1e-30)),
            "total_variation": float(np.sqrt(g2).mean()),
            "negative_fraction": float(np.mean(a < -3.0 * max(mad, 1e-12)))}
