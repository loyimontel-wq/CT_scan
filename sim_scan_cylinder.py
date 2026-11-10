#!/usr/bin/env python3
"""
sim_scan.py - make a fake Nikon-style scan (xtekct + ang + projection TIFFs) of a
phantom made of a cylinder or block (quartz or pyrite) with 3 zircon grains, so ct_recon.py can be tested without real data.

    python sim_scan.py --phantom quartz --shape cylinder --out ./sim_quartz
    python ct_recon.py --xtekct simscan/simscan.xtekct --ang simscan/simscan.ang \
        --proj-dir simscan --bin 1 --step 1 --out simscan/recon

The true centre-of-rotation offset is printed so you can check the script finds it.
"""
import argparse, os
import numpy as np
import tifffile

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="simscan")
ap.add_argument("--nu", type=int, default=192)
ap.add_argument("--nv", type=int, default=128)
ap.add_argument("--nproj", type=int, default=500)
ap.add_argument("--cor", type=float, default=3.7, help="true axis offset in px")
ap.add_argument("--shape", default="cylinder", choices=["cylinder", "box"],
                help="host shape: vertical cylinder (default) or rectangular block")
ap.add_argument("--phantom", default="quartz", choices=["quartz", "pyrite", "spheres"])
ap.add_argument("--noise", type=float, default=0.0, help="photon noise: 0=off (default), 1=mild, 0.2=noisy")
a = ap.parse_args()

name = os.path.basename(os.path.normpath(a.out))
os.makedirs(a.out, exist_ok=True)

# geometry copied from your real scan
sod, sdd = 69.1847839355469, 1147.123
du = dv = 0.2 * 2048 / a.nu          # keeps the same field of view as the real detector
white = 60000.0
mag = sdd / sod
voxel = du / mag
uc = (a.nu - 1) / 2 + a.cor

# Phantom = one rectangular block of host mineral + 3 zircon grains inside it.
# Attenuation is in 1/mm, rough values at ~90-100 keV effective energy
# (130 kV + 0.25 mm Cu). Good for testing, NOT calibrated:
#   quartz (SiO2, 2.65 g/cm3) ~0.045    pyrite (FeS2, 5.0 g/cm3) ~0.13    zircon ~0.20
MU = {"quartz": 0.96, "pyrite": 7.81, "zircon": 21.10}
BOX_HALF = np.array([7.0, 4.0, 5.0])        # block: 14 x 8 x 10 mm (x, y, z)
CYL_R, CYL_HALF_H = 6.0, 5.0                # cylinder: 12 mm diameter, 10 mm tall, axis = z
boxes, cylinders, spheres = [], [], []
if a.phantom in ("quartz", "pyrite"):
    host = MU[a.phantom]
    if a.shape == "box":
        boxes = [((0, 0, 0), BOX_HALF, host)]
    else:
        cylinders = [((0, 0, 0), CYL_R, CYL_HALF_H, host)]
    # three zircon grains: (x, y, z) mm, radius mm. Added ON TOP of the host,
    # so their extra attenuation is (zircon - host).
    for ctr, R in [((-3.0, 0.5, 1.5), 1.0), ((1.5, 1.5, -2.0), 0.8), ((3.5, -2.5, 2.5), 0.6)]:
        spheres.append((ctr, R, MU["zircon"] - host))
else:   # old sphere test
    spheres = [((0, 0, 0), 7.0, 0.5), ((3, -2, 0), 2.0, 1.0), ((-3.5, 3, 2), 1.2, 1.5),
               ((1, 4, -2.5), 1.0, 1.5), ((-2, -4, 1), 1.5, 1.0)]

ang = -(np.arange(a.nproj) * 360.0 / a.nproj) - 0.01     # negative, like your .ang
u = (np.arange(a.nu) - uc) * du
v = ((a.nv - 1) / 2 - np.arange(a.nv)) * dv
UU, VV = np.meshgrid(u, v)

for k, deg in enumerate(ang):
    t_ = np.deg2rad(deg)
    c, s = np.cos(t_), np.sin(t_)
    src = np.array([-sod * s * -1 * 0 + (0 * c - sod * s), (0 * s + sod * c), 0.0])
    sp = -(sdd - sod)
    D = np.stack([UU * c - sp * s, UU * s + sp * c, VV], -1)
    d = D - src
    dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
    p = np.zeros_like(UU)
    for ctr, half, mu in boxes:                      # ray / box chord length (slab method)
        lo, hi = np.array(ctr) - half, np.array(ctr) + half
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - src) / dn
            t2 = (hi - src) / dn
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        p += mu * np.clip(tmax - tmin, 0, None)
    for ctr, R_c, hh, mu in cylinders:               # ray / vertical cylinder chord length
        o = src - np.array(ctr)
        A = dn[..., 0] ** 2 + dn[..., 1] ** 2
        B = 2 * (o[0] * dn[..., 0] + o[1] * dn[..., 1])
        C = o[0] ** 2 + o[1] ** 2 - R_c ** 2
        disc = B ** 2 - 4 * A * C
        sq = np.sqrt(np.clip(disc, 0, None))
        tl1, tl2 = (-B - sq) / (2 * A), (-B + sq) / (2 * A)
        with np.errstate(divide="ignore", invalid="ignore"):
            tz1, tz2 = (-hh - o[2]) / dn[..., 2], (hh - o[2]) / dn[..., 2]
        tmin = np.maximum(tl1, np.minimum(tz1, tz2))
        tmax = np.minimum(tl2, np.maximum(tz1, tz2))
        p += mu * np.where(disc > 0, np.clip(tmax - tmin, 0, None), 0.0)
    for ctr, R, mu in spheres:
        w = np.array(ctr, float) - src
        along = (w * dn).sum(-1)
        dist2 = (w ** 2).sum() - along ** 2
        p += mu * 2 * np.sqrt(np.clip(R ** 2 - dist2, 0, None))
    I = white * np.exp(-p)
    if a.noise > 0:
        I = np.random.poisson(I * a.noise) / a.noise
    tifffile.imwrite(os.path.join(a.out, f"{name}_{k + 1:04d}.tif"),
                     np.clip(I, 0, 65535).astype(np.uint16))
    if k % 40 == 0:
        print(f"  projection {k + 1}/{a.nproj}", end="\r")
print()

with open(os.path.join(a.out, name + ".ang"), "w") as f:
    f.write("Proj\tAngle(deg)\r\n")
    for k, deg in enumerate(ang):
        f.write(f"{k + 1}:\t{deg:08.3f}\r\n")

with open(os.path.join(a.out, name + ".xtekct"), "w") as f:
    f.write(f"""[XTekCT]
Name={name}
VoxelsX={a.nu}
VoxelsY={a.nu}
VoxelsZ={a.nv}
VoxelSizeX={voxel}
VoxelSizeY={voxel}
VoxelSizeZ={voxel}
SrcToObject={sod}
SrcToDetector={sdd}
DetectorPixelsX={a.nu}
DetectorPixelsY={a.nv}
DetectorPixelSizeX={du}
DetectorPixelSizeY={dv}
WhiteLevel={white}
Projections={a.nproj}
InitialAngle=0.0
AngularStep={360.0 / a.nproj}
Units=mm
""")
print(f"Wrote {a.nproj} projections to {a.out}")
print(f"TRUE centre-of-rotation offset: {a.cor:+.2f} px  (voxel {voxel:.4f} mm)")
