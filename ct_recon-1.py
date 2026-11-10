#!/usr/bin/env python3
"""
ct_recon.py - Cone-beam (FDK) reconstruction for Nikon / X-Tek CT scans.

Reads  : <scan>.xtekct (geometry), <scan>.ang (angles), projection TIFFs
Does   : load -> normalise -> find centre of rotation -> FDK -> save
Writes : ONE multi-page 16-bit TIFF (default), and optionally a folder of slices
         or .vol + .vgi (opens in VGSTUDIO MAX)

Example
-------
python ct_recon.py --xtekct scan.xtekct --ang scan.ang --proj-dir ./scan_folder \
       --bin 4 --step 2 --out ./recon   (add --format stack vol for VG Studio too)

Requires: numpy, scipy, tifffile   (pip install numpy scipy tifffile)
"""
import argparse, glob, os, re, sys, time
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter


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
             if not re.search(r"(flat|dark|white|black|ref)", os.path.basename(f), re.I)]
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


def cor_from_opposing(proj, angles_deg):
    """Estimate the axis position (px from detector centre) from the two projections
    ~180 deg apart. Works for scans covering 360 deg (yours: 2000 x 0.18 deg)."""
    a = np.asarray(angles_deg)
    i0 = 0
    target = a[i0] + 180.0
    i1 = int(np.argmin(np.abs(a - target)))
    nv, nu = proj.shape[1:]
    rows = slice(nv // 4, 3 * nv // 4)
    p0 = proj[i0, rows].mean(axis=0)
    p1 = proj[i1, rows].mean(axis=0)[::-1]            # mirror the opposing view
    p0 = p0 - p0.mean(); p1 = p1 - p1.mean()
    # cross-correlate via FFT (zero padded)
    n = 2 * nu
    cc = np.fft.irfft(np.fft.rfft(p0, n) * np.conj(np.fft.rfft(p1, n)), n)
    cc = np.concatenate([cc[-nu:], cc[:nu]])          # lags -nu .. nu-1
    lag = _subpixel_peak(cc) - nu                      # p0[i] ~ p1m[i - lag]
    return lag / 2.0


def sharpness(img):
    """Energy (sum of squares) inside a circular mask. A correct COR makes the object
    compact and edges sharp -> maximum; a wrong COR smears it out. (Tested on a phantom:
    peak within ~0.1 px; the gradient/TV metrics were NOT reliable.)"""
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    mask = (yy - ny / 2) ** 2 + (xx - nx / 2) ** 2 < (0.45 * min(ny, nx)) ** 2
    return float((img[mask] ** 2).sum())


def cor_sweep(proj, angles_deg, geo, centre, half_range=6.0, step=0.25, n_use=360):
    """Refine the COR by reconstructing the middle slice for candidate values
    and keeping the one with the best sharpness metric."""
    sel = np.linspace(0, len(angles_deg) - 1, min(n_use, len(angles_deg))).astype(int)
    best = None
    scores = []
    for c in np.arange(centre - half_range, centre + half_range + 1e-9, step):
        geo.cor_px = c
        sl = fdk(proj[sel], np.asarray(angles_deg)[sel], geo,
                 z_range=(geo.nv // 2, geo.nv // 2 + 1), verbose=False)[0]
        s = sharpness(sl)
        scores.append((c, s))
        if best is None or s > best[1]:
            best = (c, s)
    geo.cor_px = best[0]
    return best[0], scores


# --------------------------------------------------------------------- FDK
def ramp_filter(nu, du_v, window="hann", cutoff=1.0):
    n = int(2 ** np.ceil(np.log2(2 * nu)))
    f = np.fft.rfftfreq(n, d=du_v)
    H = np.abs(f)
    fmax = f.max() * cutoff
    if window == "hann":
        H *= 0.5 * (1 + np.cos(np.pi * np.minimum(f / fmax, 1.0)))
    elif window == "hamming":
        H *= 0.54 + 0.46 * np.cos(np.pi * np.minimum(f / fmax, 1.0))
    H[f > fmax] = 0
    return H, n


def fdk(proj, angles_deg, geo, z_range=None, window="hann", cutoff=1.0,
        angle_sign=1.0, nxy=None, verbose=True, slab_voxels=3_000_000):
    """Feldkamp-Davis-Kress reconstruction. proj: (n, nv, nu) line integrals.
    Returns volume (nz, ny, nx) in 1/mm."""
    n, nv, nu = proj.shape
    nxy = nxy or nu
    z0, z1 = z_range or (0, nv)
    nz = z1 - z0
    sod = geo.sod
    th = np.deg2rad(np.asarray(angles_deg, float)) * angle_sign
    # angular step (handles uneven spacing from .ang file): use mean spacing over 360
    dth = 2 * np.pi / n if abs(th.max() - th.min()) > 1.5 * np.pi else \
        abs(np.diff(th)).mean()
    half = 0.5 if abs(th.max() - th.min()) > 1.5 * np.pi else 1.0  # full vs short scan(approx)

    # virtual-detector coordinates
    u = (np.arange(nu) - geo.uc) * geo.du_v
    v = ((nv - 1) / 2.0 - np.arange(nv)) * geo.dv_v
    cosw = sod / np.sqrt(sod ** 2 + u[None, :] ** 2 + v[:, None] ** 2)
    H, nfft = ramp_filter(nu, geo.du_v, window, cutoff)

    xs = (np.arange(nxy) - (nxy - 1) / 2.0) * geo.voxel
    X, Y = np.meshgrid(xs, xs)             # (ny, nx)
    vol = np.zeros((nz, nxy, nxy), np.float32)

    zi_all = np.arange(z0, z1)
    zvox = ((nv - 1) / 2.0 - zi_all) * geo.dv_v      # z position (mm) of each output slice
    # (output voxel pitch in z equals detector pitch at the axis)
    chunk = max(1, slab_voxels // (nxy * nxy))

    t0 = time.time()
    for k in range(n):
        q = proj[k] * cosw
        q = np.fft.irfft(np.fft.rfft(q, nfft, axis=1) * H[None, :], nfft, axis=1)[:, :nu]
        q = q.astype(np.float32)
        c, s_ = np.cos(th[k]), np.sin(th[k])
        t = X * c + Y * s_
        s = -X * s_ + Y * c
        mag = sod / (sod - s)
        uf = t * mag / geo.du_v + geo.uc
        j0 = np.floor(uf).astype(np.int64)
        fu = (uf - j0).astype(np.float32)
        okj = (j0 >= 0) & (j0 < nu - 1)
        j0c = np.clip(j0, 0, nu - 2)
        w = (mag ** 2).astype(np.float32)
        for a in range(0, nz, chunk):
            b = min(nz, a + chunk)
            zz = zvox[a:b, None, None]
            vf = (nv - 1) / 2.0 - zz * mag[None] / geo.dv_v
            i0 = np.floor(vf).astype(np.int64)
            fv = (vf - i0).astype(np.float32)
            ok = okj[None] & (i0 >= 0) & (i0 < nv - 1)
            i0c = np.clip(i0, 0, nv - 2)
            base = i0c * nu + j0c[None]
            qf = q.ravel()
            val = ((1 - fv) * ((1 - fu)[None] * qf[base] + fu[None] * qf[base + 1]) +
                   fv * ((1 - fu)[None] * qf[base + nu] + fu[None] * qf[base + nu + 1]))
            vol[a:b] += np.where(ok, val * w[None], 0.0)
        if verbose and (k % 25 == 0 or k == n - 1):
            el = time.time() - t0
            print(f"  backprojected {k + 1}/{n}  ({el:.0f}s, ETA {el / (k + 1) * (n - k - 1):.0f}s)", end="\r")
    if verbose:
        print()
    return vol * (dth * half)


# ------------------------------------------------------------------ saving
def to_uint16(vol, lo_pct=0.2, hi_pct=99.8):
    sub = vol[::2, ::2, ::2]
    lo, hi = np.percentile(sub, [lo_pct, hi_pct])
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
    ap.add_argument("--no-sweep", action="store_true", help="skip sharpness refinement")
    ap.add_argument("--angle-sign", type=float, default=1.0,
                    help="use -1 if the result is mirrored")
    ap.add_argument("--window", default="hann")
    ap.add_argument("--z", nargs=2, type=int, help="reconstruct only detector rows z0 z1 "
                                                   "(binned px) - good for tests")
    ap.add_argument("--out", default="recon")
    ap.add_argument("--format", nargs="+", default=["stack"],
                    choices=["stack", "tiff", "vol"],
                    help="stack = ONE multi-page TIFF (default); tiff = folder of slices; "
                         "vol = .vol + .vgi for VG Studio")
    a = ap.parse_args()

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
    print(f"Detector {nu}x{nv}, voxel size {geo.voxel:.5f} mm")

    if a.cor is not None:
        geo.cor_px = a.cor
    else:
        c0 = cor_from_opposing(proj, angles)
        print(f"Centre of rotation (opposing views): {c0:+.2f} px from detector centre")
        geo.cor_px = c0
        if not a.no_sweep:
            print("Refining by slice sharpness sweep...")
            c1, _ = cor_sweep(proj, angles, geo, c0)
            print(f"Centre of rotation (refined): {c1:+.2f} px")
    print(f"Using COR = {geo.cor_px:+.2f} px (binned)")

    print("Reconstructing (FDK)...")
    zr = tuple(a.z) if a.z else None
    vol = fdk(proj, angles, geo, z_range=zr, window=a.window, angle_sign=a.angle_sign)

    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(a.xtekct))[0])
    vol16, lo, hi = to_uint16(vol)
    print(f"Grey-value mapping: {lo:.5g}..{hi:.5g} 1/mm -> 0..65535")
    if "stack" in a.format:
        save_tiff_single(vol16, geo.voxel, a.out, name)
    if "tiff" in a.format:
        save_tiff_stack(vol16, os.path.join(a.out, name + "_tiff"), name)
    if "vol" in a.format:
        save_vol_vgi(vol16, geo.voxel, a.out, name)
    print("Done.")


if __name__ == "__main__":
    main()
