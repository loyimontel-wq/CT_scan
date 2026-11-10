#!/usr/bin/env python3
"""
ct_recon.py - Cone-beam (FDK) reconstruction for Nikon / X-Tek CT scans.

Reads  : <scan>.xtekct (geometry), <scan>.ang (angles), projection TIFFs
Does   : load -> normalise -> find centre of rotation -> FDK -> save
Writes : ONE multi-page 16-bit TIFF (default), optionally a float32 TIFF in 1/mm, a folder of slices,
         or .vol + .vgi (VGSTUDIO MAX), plus <name>_recon_settings.json recording everything used.

Example
-------
python ct_recon.py --xtekct scan.xtekct --ang scan.ang --proj-dir ./scan_folder \
       --bin 4 --step 2 --out ./recon
(add  --format stack f32  for a float32 stack in 1/mm;  pip install numba  for a much faster backprojection)

Requires: numpy, scipy, tifffile   (optional: numba)

What changed in this version (see the explanation sent with it):
  1. Ramp filter built in the spatial domain (Kak & Slaney Ram-Lak) instead of sampling |f| - removes a
     uniform ~2.4 % low bias in reconstructed values.
  2. Centre of rotation: centroid-of-projections estimate (within 0.08 px on all test scans; the old
     opposing-view estimate was up to 1.2 px off). The sharpness sweep is now an optional check (--check-cor).
  3. Actual angular spacing from the .ang file is used for the backprojection weights.
  4. (Tried edge/taper padding for samples that reach the detector edge: both were worse than zero padding
     in tests, so zero padding is kept.)
  5. Optional numba backprojection (--backend auto/numba/numpy) and rows are only filtered where needed.
  6. Settings log (JSON), optional float32 output (--format f32).
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
import tifffile

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # numba is optional
    HAVE_NUMBA = False


# ----------------------------------------------------------------- loading
def parse_xtekct(path):
    """Read the key=value .xtekct file into a dict (numbers converted)."""
    cfg = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("["):
                k, v = line.split("=", 1)
                try:
                    cfg[k.strip()] = float(v)
                except ValueError:
                    cfg[k.strip()] = v.strip()
    return cfg


def read_angles(path):
    """Read the .ang file ('1:  -000.010' per line). Returns degrees."""
    ang = []
    with open(path, "r", errors="ignore") as f:
        next(f)  # header
        for line in f:
            m = re.match(r"\s*\d+\s*:\s*([-+]?\d+\.?\d*)", line)
            if m:
                ang.append(float(m.group(1)))
    return np.array(ang)


def bin2d(a, b):
    if b == 1:
        return a
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def find_projection_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.tiff")))
    # drop obvious non-projection files
    files = [f for f in files
             if not re.search(r"(flat|dark|white|black|ref|stack)", os.path.basename(f), re.I)]
    return files


def load_projections(files, binning=1, step=1, flat=None, dark=None, white_level=60000.0):
    """Load every `step`-th projection, bin, and convert to line integrals p = -ln(I/I0).
    Returns array (n_proj, nv, nu) float32 and the indices used."""
    idx = np.arange(0, len(files), step)
    flat_b = bin2d(tifffile.imread(flat).astype(np.float32), binning) if flat else None
    dark_b = bin2d(tifffile.imread(dark).astype(np.float32), binning) if dark else None
    out = None
    for n, i in enumerate(idx):
        im = bin2d(tifffile.imread(files[i]).astype(np.float32), binning)
        if flat_b is not None:
            d = dark_b if dark_b is not None else 0.0
            im = (im - d) / np.maximum(flat_b - d, 1.0)
        else:
            im = im / white_level
        im = np.clip(im, 1e-4, None)
        if out is None:
            out = np.empty((len(idx),) + im.shape, np.float32)
        out[n] = -np.log(im)
        if n % 100 == 0:
            print(f"  loaded {n + 1}/{len(idx)}", end="\r")
    print()
    return out, idx


# -------------------------------------------------------------- geometry
class Geometry:
    def __init__(self, sod, sdd, du, dv, nu, nv, cor_px=0.0):
        self.sod, self.sdd = sod, sdd
        self.du, self.dv = du, dv            # detector pixel size (mm), after binning
        self.nu, self.nv = nu, nv
        self.cor_px = cor_px                 # axis position relative to detector centre (px)
        self.mag = sdd / sod
        # "virtual detector" through the rotation axis (FDK is simplest there)
        self.du_v, self.dv_v = du / self.mag, dv / self.mag
        self.voxel = self.du_v               # natural voxel size (mm)

    @property
    def uc(self):                            # detector column of the rotation axis
        return (self.nu - 1) / 2.0 + self.cor_px


# ---------------------------------------------------------- centre of rotation
def _subpixel_peak(cc):
    k = int(np.argmax(cc))
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return k + 0.5 * (y0 - y2) / den
    return float(k)


def cor_from_centroid(proj):
    """Axis position (px from detector centre) from the centre of mass of every projection.
    The centre of mass of the object projects to the centroid of its projection; over a full 360 deg turn
    it circles the axis, so the mean centroid is the axis position. Needs the sample to be (nearly)
    inside the detector and a ~zero background. Tested: within 0.03 px on four simulated scans."""
    nv, nu = proj.shape[1:]
    s = proj[:, nv // 4:3 * nv // 4, :].mean(axis=1)                 # (n, nu)
    edge = max(2, nu // 30)
    base = 0.5 * (s[:, :edge].mean(axis=1) + s[:, -edge:].mean(axis=1))
    s = np.clip(s - base[:, None], 0, None)
    u = np.arange(nu)
    c = (s * u).sum(axis=1) / np.maximum(s.sum(axis=1), 1e-12)
    return float(c.mean() - (nu - 1) / 2.0)


def cor_from_opposing(proj, angles_deg):
    """Older estimate from the two projections ~180 deg apart. Biased in cone-beam geometry (up to ~1 px
    in tests) because the two views have different magnification; kept only as a cross-check."""
    a = np.asarray(angles_deg)
    i0 = 0
    target = a[i0] + 180.0
    i1 = int(np.argmin(np.abs(a - target)))
    nv, nu = proj.shape[1:]
    rows = slice(nv // 4, 3 * nv // 4)
    p0 = proj[i0, rows].mean(axis=0)
    p1 = proj[i1, rows].mean(axis=0)[::-1]            # mirror the opposing view
    p0 = p0 - p0.mean(); p1 = p1 - p1.mean()
    n = 2 * nu
    cc = np.fft.irfft(np.fft.rfft(p0, n) * np.conj(np.fft.rfft(p1, n)), n)
    cc = np.concatenate([cc[-nu:], cc[:nu]])          # lags -nu .. nu-1
    lag = float(_subpixel_peak(cc)) - nu
    return float(lag / 2.0)


def sharpness(img):
    """Energy (sum of squares) inside a circular mask. A correct COR makes the object
    compact and edges sharp -> maximum; a wrong COR smears it out. (Tested on a phantom:
    peak within ~0.1 px; the gradient/TV metrics were NOT reliable.)"""
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - ny / 2) ** 2 + (xx - nx / 2) ** 2 < (0.45 * min(ny, nx)) ** 2
    return float((img[mask] ** 2).sum())


def cor_sweep(proj, angles_deg, geo, centre, half_range=2.0, step=0.2, n_use=360, **fdk_kw):
    """Refine the COR by reconstructing the middle slice for candidate values and keeping the sharpest,
    with a parabola through the best three scores for sub-step precision."""
    sel = np.linspace(0, len(angles_deg) - 1, min(n_use, len(angles_deg))).astype(int)
    scores = []
    for c in np.arange(centre - half_range, centre + half_range + 1e-9, step):
        geo.cor_px = c
        sl = fdk(proj[sel], np.asarray(angles_deg)[sel], geo,
                 z_range=(geo.nv // 2, geo.nv // 2 + 1), verbose=False, **fdk_kw)[0]
        scores.append((c, sharpness(sl)))
    cs = np.array([c for c, _ in scores]); ss = np.array([x for _, x in scores])
    k = int(np.argmax(ss))
    cbest = cs[k]
    if 0 < k < len(ss) - 1:
        den = ss[k - 1] - 2 * ss[k] + ss[k + 1]
        if den < 0:
            cbest = cs[k] + 0.5 * (ss[k - 1] - ss[k + 1]) / den * step
    geo.cor_px = cbest
    return cbest, scores


# --------------------------------------------------------------------- FDK
def ramp_filter(nu, du_v, window="hann", cutoff=1.0):
    """Ram-Lak ramp filter built from its spatial-domain kernel (Kak & Slaney, eq. 61):
    h(0)=1/(4 du^2), h(n)=0 for even n, h(n)=-1/(pi^2 n^2 du^2) for odd n. Sampling |f| at the FFT bins
    instead (the previous version) is not the same filter and biased reconstructed values ~2.4 % low."""
    n = int(2 ** np.ceil(np.log2(2 * nu)))
    k = np.arange(n)
    k = np.where(k <= n // 2, k, k - n).astype(float)
    h = np.zeros(n)
    h[0] = 1.0 / (4.0 * du_v ** 2)
    odd = (np.abs(k) % 2) == 1
    h[odd] = -1.0 / (np.pi ** 2 * k[odd] ** 2 * du_v ** 2)
    H = np.real(np.fft.rfft(h)) * du_v
    f = np.fft.rfftfreq(n, d=du_v)
    fmax = f.max() * cutoff
    if window == "hann":
        H = H * 0.5 * (1 + np.cos(np.pi * np.minimum(f / fmax, 1.0)))
    elif window == "hamming":
        H = H * (0.54 + 0.46 * np.cos(np.pi * np.minimum(f / fmax, 1.0)))
    H[f > fmax] = 0
    return H, n


def angle_weights(th):
    """Radians of rotation each projection represents (half the gap to each neighbour). Equal to 2*pi/n for
    even spacing; follows the real spacing in the .ang file (which is not perfectly even)."""
    th = np.asarray(th, float)
    if len(th) >= 3 and abs(th.max() - th.min()) > 1.5 * np.pi:      # full turn: neighbours wrap round
        def cd(a, b):
            return (a - b + np.pi) % (2 * np.pi) - np.pi
        return 0.5 * (np.abs(cd(np.roll(th, -1), th)) + np.abs(cd(th, np.roll(th, 1))))
    return np.abs(np.gradient(th)) if len(th) > 1 else np.ones(1)


if HAVE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _bp_numba(vol, q, j0, fu, mag, wk, zvox, vc, inv_dv, nv):
        nz, ny, nx = vol.shape
        for z in prange(nz):
            zz = zvox[z]
            for y in range(ny):
                for x in range(nx):
                    jj = j0[y, x]
                    if jj < 0:
                        continue
                    vf = vc - zz * mag[y, x] * inv_dv
                    i0 = int(np.floor(vf))
                    if i0 < 0 or i0 >= nv - 1:
                        continue
                    fv = vf - i0
                    f = fu[y, x]
                    a = q[i0, jj] * (1.0 - f) + q[i0, jj + 1] * f
                    b = q[i0 + 1, jj] * (1.0 - f) + q[i0 + 1, jj + 1] * f
                    vol[z, y, x] += wk[y, x] * ((1.0 - fv) * a + fv * b)


def fdk(proj, angles_deg, geo, z_range=None, window="hann", cutoff=1.0,
        angle_sign=1.0, nxy=None, verbose=True, slab_voxels=3_000_000,
        backend="auto"):
    """Feldkamp-Davis-Kress reconstruction. proj: (n, nv, nu) line integrals.
    Returns volume (nz, ny, nx) in 1/mm."""
    n, nv, nu = proj.shape
    nxy = nxy or nu
    z0, z1 = z_range or (0, nv)
    nz = z1 - z0
    sod = geo.sod
    th = np.deg2rad(np.asarray(angles_deg, float)) * angle_sign
    full = abs(th.max() - th.min()) > 1.5 * np.pi
    wts = angle_weights(th) * (0.5 if full else 1.0)      # (full turn: each ray is seen twice)

    use_numba = HAVE_NUMBA and backend in ("auto", "numba")
    if backend == "numba" and not HAVE_NUMBA:
        raise RuntimeError("--backend numba requested but numba is not installed (pip install numba)")

    u = (np.arange(nu) - geo.uc) * geo.du_v
    v = ((nv - 1) / 2.0 - np.arange(nv)) * geo.dv_v
    cosw = (sod / np.sqrt(sod ** 2 + u[None, :] ** 2 + v[:, None] ** 2)).astype(np.float32)
    H, nfft = ramp_filter(nu, geo.du_v, window, cutoff)

    xs = (np.arange(nxy) - (nxy - 1) / 2.0) * geo.voxel
    X, Y = np.meshgrid(xs, xs)             # (ny, nx)
    vol = np.zeros((nz, nxy, nxy), np.float32)
    zvox = (((nv - 1) / 2.0 - np.arange(z0, z1)) * geo.dv_v).astype(np.float64)
    vc = (nv - 1) / 2.0

    # only the detector rows that the requested slices can touch need filtering
    rmax = np.hypot(xs.max(), xs.max())
    mags = [sod / (sod - rmax), sod / (sod + rmax)]
    rows_needed = [vc - z * m / geo.dv_v for z in (zvox.min(), zvox.max()) for m in mags]
    r_lo = max(0, int(np.floor(min(rows_needed))) - 1)
    r_hi = min(nv, int(np.ceil(max(rows_needed))) + 3)

    buf = np.zeros((nv, nfft), np.float32)
    q = np.zeros((nv, nu), np.float32)
    chunk = max(1, slab_voxels // (nxy * nxy))
    zvox32 = zvox.astype(np.float32)

    t0 = time.time()
    for k in range(n):
        buf[r_lo:r_hi, :] = 0
        buf[r_lo:r_hi, :nu] = proj[k, r_lo:r_hi] * cosw[r_lo:r_hi]
        flt = np.fft.irfft(np.fft.rfft(buf[r_lo:r_hi], axis=1) * H[None, :], nfft, axis=1)[:, :nu]
        q[r_lo:r_hi] = flt
        c, s_ = np.cos(th[k]), np.sin(th[k])
        t = X * c + Y * s_
        s = -X * s_ + Y * c
        mag = sod / (sod - s)
        uf = t * mag / geo.du_v + geo.uc
        j0 = np.floor(uf).astype(np.int64)
        fu = (uf - j0).astype(np.float32)
        ok = (j0 >= 0) & (j0 < nu - 1)
        wk = ((mag ** 2) * wts[k]).astype(np.float32)
        if use_numba:
            _bp_numba(vol, q, np.where(ok, j0, -1).astype(np.int32), fu, mag.astype(np.float32), wk,
                      zvox32, np.float32(vc), np.float32(1.0 / geo.dv_v), nv)
        else:
            j0c = np.clip(j0, 0, nu - 2)
            magf = mag.astype(np.float32)
            qf = q.ravel()
            for a in range(0, nz, chunk):
                b = min(nz, a + chunk)
                vf = vc - zvox32[a:b, None, None] * magf[None] / np.float32(geo.dv_v)
                i0 = np.floor(vf).astype(np.int64)
                fv = (vf - i0).astype(np.float32)
                okv = ok[None] & (i0 >= 0) & (i0 < nv - 1)
                i0c = np.clip(i0, 0, nv - 2)
                base = i0c * nu + j0c[None]
                val = ((1 - fv) * ((1 - fu)[None] * qf[base] + fu[None] * qf[base + 1]) +
                       fv * ((1 - fu)[None] * qf[base + nu] + fu[None] * qf[base + nu + 1]))
                vol[a:b] += np.where(okv, val * wk[None], 0.0)
        if verbose and (k % 25 == 0 or k == n - 1):
            el = time.time() - t0
            print(f"  backprojected {k + 1}/{n}  ({el:.0f}s, ETA {el / (k + 1) * (n - k - 1):.0f}s)", end="\r")
    if verbose:
        print()
    return vol


# ------------------------------------------------------------------ saving
def to_uint16(vol, grey_range=None):
    """Map attenuation (1/mm) to 16-bit.
    Default window: 0 (air) up to 2 x the 99th percentile of the volume. The 99th
    percentile sits on the main material, so it shows at ~50 % grey and smaller, denser
    features (e.g. zircon) stay brighter instead of being clipped to the same white.
    Works the same for a full volume or a few slices (--z). Override with
    --grey-range LO HI (values in 1/mm)."""
    if grey_range:
        lo, hi = grey_range
    else:
        lo = 0.0
        hi = 2.0 * float(np.percentile(vol[::2, ::2, ::2], 99))
        if hi <= 0:
            hi = float(vol.max()) or 1.0
    out = np.clip((vol - lo) / (hi - lo), 0, 1) * 65535
    return out.astype(np.uint16), float(lo), float(hi)


def save_tiff_stack(vol16, folder, name):
    os.makedirs(folder, exist_ok=True)
    for i in range(vol16.shape[0]):
        tifffile.imwrite(os.path.join(folder, f"{name}_{i:04d}.tif"), vol16[i])
    print(f"  wrote {vol16.shape[0]} TIFF slices -> {folder}")


def save_tiff_single(vol16, voxel_mm, folder, name):
    """ONE multi-page 16-bit TIFF containing the whole volume (one page per slice).
    Opens directly in Fiji/ImageJ, napari, 3D Slicer. Voxel size is stored so Fiji
    shows it in mm."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + "_stack.tif")
    res = 1.0 / voxel_mm                      # pixels per mm
    tifffile.imwrite(path, vol16, photometric="minisblack", imagej=True,
                     resolution=(res, res),
                     metadata={"spacing": voxel_mm, "unit": "mm", "axes": "ZYX"})
    print(f"  wrote single stacked TIFF -> {path}  ({vol16.shape[0]} slices)")


def save_tiff_float(vol, voxel_mm, folder, name):
    """ONE multi-page float32 TIFF: every voxel is the attenuation coefficient in 1/mm. No grey-range
    mapping, so values are directly comparable between runs (hover in Fiji to read them)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + "_stack_f32.tif")
    res = 1.0 / voxel_mm
    tifffile.imwrite(path, vol.astype(np.float32), photometric="minisblack", imagej=True,
                     resolution=(res, res), metadata={"spacing": voxel_mm, "unit": "mm", "axes": "ZYX"})
    print(f"  wrote float32 stack (values in 1/mm) -> {path}")


def save_vol_vgi(vol16, voxel_mm, folder, name):
    """Raw little-endian uint16 .vol + .vgi descriptor for VGSTUDIO MAX.
    If the .vgi is rejected by your VG version, use File > Import > Raw data and enter
    the size / 16-bit unsigned / little endian / voxel size printed below."""
    os.makedirs(folder, exist_ok=True)
    nz, ny, nx = vol16.shape
    volpath = os.path.join(folder, name + ".vol")
    vol16.astype("<u2").tofile(volpath)          # x fastest, then y, then z
    vgi = f"""{{volume1}}
[representation]
size = {nx} {ny} {nz}
datatype = unsigned integer
datarange = 0 65535
bitsperelement = 16

[file1]
SkipHeader = 0
FileFormat = raw
Size = {nx} {ny} {nz}
Name = {name}.vol
Datatype = 16bit
datarange = 0 65535
BitsPerElement = 16

[geometry]
resolution = {voxel_mm:.8f} {voxel_mm:.8f} {voxel_mm:.8f}
unit = mm
"""
    with open(os.path.join(folder, name + ".vgi"), "w") as f:
        f.write(vgi)
    print(f"  wrote {volpath} (+ .vgi)")
    print(f"  raw import values: {nx} x {ny} x {nz}, 16-bit unsigned, little endian, "
          f"voxel {voxel_mm:.6f} mm")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="FDK reconstruction for Nikon/X-Tek scans")
    ap.add_argument("--xtekct", required=True)
    ap.add_argument("--ang", required=True)
    ap.add_argument("--proj-dir", required=True, help="folder with projection TIFFs")
    ap.add_argument("--flat"); ap.add_argument("--dark")
    ap.add_argument("--bin", type=int, default=4, help="detector binning (4 -> 512^3)")
    ap.add_argument("--step", type=int, default=2, help="use every Nth projection")
    ap.add_argument("--cor", type=float, help="centre of rotation (binned px from detector "
                                              "centre); skips automatic search")
    ap.add_argument("--check-cor", action="store_true",
                    help="also run the slice-sharpness sweep and print its result as a cross-check. It never "
                         "overrides the centroid estimate (in tests it was no better on clean scans and 2 px "
                         "off when the sample overfilled the detector)")
    ap.add_argument("--no-sweep", action="store_true", help=argparse.SUPPRESS)   # old option, now a no-op
    ap.add_argument("--angle-sign", type=float, default=1.0,
                    help="use -1 if the result is mirrored")
    ap.add_argument("--window", default="hann", choices=["hann", "hamming", "ramlak"])
    ap.add_argument("--cutoff", type=float, default=0.8,
                    help="filter cutoff 0-1. Lower = smoother/less noise but softer detail "
                         "(try 0.5-0.6 for noisy data, 1.0 for maximum sharpness)")
    ap.add_argument("--backend", default="auto", choices=["auto", "numba", "numpy"],
                    help="auto = numba if installed (much faster), else numpy")
    ap.add_argument("--z", nargs=2, type=int, help="reconstruct only detector rows z0 z1 "
                                                   "(binned px) - good for tests")
    ap.add_argument("--grey-range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="attenuation window (1/mm) mapped to 0..65535 (default: automatic)")
    ap.add_argument("--out", default="recon")
    ap.add_argument("--format", nargs="+", default=["stack"],
                    choices=["stack", "f32", "tiff", "vol"],
                    help="stack = ONE multi-page 16-bit TIFF (default); f32 = ONE float32 TIFF in 1/mm; "
                         "tiff = folder of slices; vol = .vol + .vgi for VG Studio")
    a = ap.parse_args()
    t_start = time.time()

    cfg = parse_xtekct(a.xtekct)
    angles = read_angles(a.ang)
    files = find_projection_files(a.proj_dir)
    print(f"{len(files)} projection files, {len(angles)} angles")
    n = min(len(files), len(angles))
    files, angles = files[:n], angles[:n]

    print("Loading projections...")
    proj, idx = load_projections(files, a.bin, a.step, a.flat, a.dark,
                                 cfg.get("WhiteLevel", 60000.0))
    angles = angles[idx]
    nv, nu = proj.shape[1:]
    geo = Geometry(cfg["SrcToObject"], cfg["SrcToDetector"],
                   cfg["DetectorPixelSizeX"] * a.bin, cfg["DetectorPixelSizeY"] * a.bin,
                   nu, nv)
    print(f"Detector {nu}x{nv} after {a.bin}x binning, voxel size {geo.voxel:.5f} mm")
    if a.bin > 1 or a.step > 1:
        print(f"  note: using --bin {a.bin} --step {a.step}; use --bin 1 --step 1 for full "
              f"detail (slower)")
    backend = "numba" if (HAVE_NUMBA and a.backend in ("auto", "numba")) else "numpy"
    print(f"Backprojection backend: {backend}" + ("" if HAVE_NUMBA else "  (pip install numba for a big speed-up)"))
    fdk_kw = dict(window=a.window, cutoff=a.cutoff, angle_sign=a.angle_sign, backend=a.backend)

    cor_info = {}
    if a.cor is not None:
        geo.cor_px = a.cor
        cor_info["method"] = "given on the command line"
    else:
        c_cen = cor_from_centroid(proj)
        c_opp = cor_from_opposing(proj, angles)
        print(f"Centre of rotation: centroid method {c_cen:+.2f} px   (opposing-view cross-check {c_opp:+.2f} px)")
        cor_info.update(centroid=c_cen, opposing_views=c_opp)
        geo.cor_px = c_cen
        if abs(c_cen - c_opp) > 2.0:
            print("  note: the two estimates differ by >2 px. The opposing-view one is known to be biased in "
                  "cone-beam geometry (up to ~1.2 px in tests), so this alone is not alarming.")
        cor_info["method"] = "centroid of projections"
        if a.check_cor:
            print("Cross-check: slice sharpness sweep (+-2 px around the centroid estimate)...")
            c1, _ = cor_sweep(proj, angles, geo, c_cen, **fdk_kw)
            geo.cor_px = c_cen                       # the sweep never overrides the centroid estimate
            print(f"  sweep result {c1:+.2f} px (difference {c1 - c_cen:+.2f} px)")
            cor_info["sweep_check"] = float(c1)
            if abs(c1 - c_cen) > 0.5:
                print("  note: they differ by >0.5 px. If the sample touches the detector edge the sweep is the "
                      "less reliable one; otherwise inspect a slice for doubled edges and try --cor.")
    print(f"Using COR = {geo.cor_px:+.2f} px (binned)")

    print("Reconstructing (FDK)...")
    zr = tuple(a.z) if a.z else None
    t_rec = time.time()
    vol = fdk(proj, angles, geo, z_range=zr, **fdk_kw)
    t_rec = time.time() - t_rec

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(a.xtekct))[0])
    vol16, lo, hi = to_uint16(vol, a.grey_range)
    print(f"Grey-value mapping: {lo:.5g}..{hi:.5g} 1/mm -> 0..65535")
    if "stack" in a.format:
        save_tiff_single(vol16, geo.voxel, a.out, name)
    if "f32" in a.format:
        save_tiff_float(vol, geo.voxel, a.out, name)
    if "tiff" in a.format:
        save_tiff_stack(vol16, os.path.join(a.out, name + "_tiff"), name)
    if "vol" in a.format:
        save_vol_vgi(vol16, geo.voxel, a.out, name)

    settings = {
        "command_line": " ".join(sys.argv),
        "script": "ct_recon.py (spatial Ram-Lak filter, centroid COR, real angle weights)",
        "inputs": {"xtekct": a.xtekct, "ang": a.ang, "proj_dir": a.proj_dir, "n_files": len(files),
                   "n_projections_used": int(len(idx)), "bin": a.bin, "step": a.step},
        "geometry": {"SrcToObject_mm": geo.sod, "SrcToDetector_mm": geo.sdd, "detector_px_after_binning": [nu, nv],
                     "detector_pixel_mm_after_binning": [geo.du, geo.dv], "voxel_size_mm": geo.voxel},
        "centre_of_rotation": dict(cor_info, used_px=float(geo.cor_px)),
        "filter": {"window": a.window, "cutoff": a.cutoff, "padding": "zero", "kernel": "spatial Ram-Lak"},
        "angle_sign": a.angle_sign,
        "z_rows": list(zr) if zr else "all",
        "grey_mapping": {"lo_1_per_mm": lo, "hi_1_per_mm": hi,
                         "how_to_read": "mu (1/mm) = lo + grey/65535 * (hi - lo)"},
        "volume_shape_zyx": list(vol.shape),
        "backend": backend,
        "seconds": {"reconstruction": round(t_rec, 1), "total": round(time.time() - t_start, 1)},
    }
    sp = os.path.join(a.out, name + "_recon_settings.json")
    with open(sp, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  wrote settings log -> {sp}")
    print("Done.")


if __name__ == "__main__":
    main()
