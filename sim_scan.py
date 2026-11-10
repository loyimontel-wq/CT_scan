#!/usr/bin/env python3
"""
sim_scan.py - make a fake Nikon-style scan (xtekct + ang + projection TIFFs) of a
phantom made of a quartz grain (with pores and zircon inclusions), so ct_recon.py can be tested without real data.

    python sim_scan.py --out ./simscan
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
ap.add_argument("--nproj", type=int, default=240)
ap.add_argument("--cor", type=float, default=3.7, help="true axis offset in px")
ap.add_argument("--phantom", default="quartz", choices=["quartz", "spheres"])
ap.add_argument("--noise", type=float, default=0.0, help="Poisson-like noise, 0=off")
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

# phantom: (x, y, z) mm, radius mm, attenuation 1/mm.  Entries are ADDED together,
# so a negative value carves a void out of whatever it sits inside.
# Approximate linear attenuation at ~90-100 keV effective energy (130 kV + 0.25 mm Cu):
#   quartz (SiO2, 2.65 g/cm3)  ~0.045 /mm      zircon (ZrSiO4, 4.6 g/cm3) ~0.20 /mm
# These are rough textbook-style numbers, good enough for testing, not calibrated.
MU_QUARTZ, MU_ZIRCON = 0.045, 0.20
if a.phantom == "quartz":
    spheres = [((0, 0, 0), 7.0, MU_QUARTZ),             # quartz grain
               ((3, -2, 0), 1.6, -MU_QUARTZ),           # pore / void (air)
               ((-3.5, 3, 2), 1.0, -MU_QUARTZ),         # small pore
               ((1, 4, -2.5), 0.9, MU_ZIRCON - MU_QUARTZ),   # zircon inclusion
               ((-2, -4, 1), 1.2, MU_ZIRCON - MU_QUARTZ)]    # zircon inclusion
else:
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
