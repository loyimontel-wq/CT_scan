#!/usr/bin/env python3
"""
sim_scan.py - make a fake Nikon-style scan (xtekct + ang + projection TIFFs + truth.json) of a simple
phantom (cylinder or block of one mineral, with spherical grains of other minerals) so that ct_recon.py
can be tested without real data.

    python sim_scan.py --out ./sim_quartz                          # quartz cylinder, zircon/magnetite/void grains
    python sim_scan.py --phantom pyrite --inclusion zircon --out ./sim_pyrite
    python sim_scan.py --phantom-file phantom.example.json --out ./sim_custom
    python sim_scan.py --energy 44.79 --source paper_T2_MXLAC ...  # other tabulated values from materials.csv
    python sim_scan.py --source calc_eq7 --energy 61.3 ...         # computed with mu_calc.py (paper Eq. 7)

Model (unchanged from before): ONE effective energy, one mu per material, line integral p = sum(mu x chord
length in mm) along each ray from the point source to each detector pixel centre, I = WhiteLevel*exp(-p).
No spectrum, beam hardening, scatter or detector blur. Photon noise optional (--noise).

Attenuation values come from materials.csv (see its header for what each 'source' means).
The true centre-of-rotation offset and everything else needed to validate ct_recon.py is written to truth.json.
"""
import argparse, csv, json, os, sys
import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="simscan")
ap.add_argument("--nu", type=int, default=192)
ap.add_argument("--nv", type=int, default=128)
ap.add_argument("--nproj", type=int, default=500)
ap.add_argument("--cor", type=float, default=3.7, help="true axis offset in px")
ap.add_argument("--shape", default="cylinder", choices=["cylinder", "box"],
                help="host shape: vertical cylinder (default) or rectangular block")
ap.add_argument("--phantom", default="quartz",
                help="host material name from materials.csv (e.g. quartz, pyrite), or 'spheres' for the "
                     "old arbitrary-value test phantom")
ap.add_argument("--inclusion", default="mixed",
                help="material of the 3 grains: a name from materials.csv, or 'mixed' = zircon + magnetite + void")
ap.add_argument("--phantom-file", help="JSON file describing host and grains (overrides --phantom/--shape/--inclusion)")
ap.add_argument("--materials", default=os.path.join(HERE, "materials.csv"))
ap.add_argument("--energy", type=float, default=62.53,
                help="keV. Default 62.53 = the paper's tabulated energy closest to its 61.3 keV for 130 kV")
ap.add_argument("--source", default="paper_T2_NIST",
                help="which column of values to use: paper_T2_NIST (default), paper_T2_MXLAC, paper_T3_MXLAC, "
                     "calc_eq7, test_only")
ap.add_argument("--noise", type=float, default=0.0,
                help="photon noise: 0=off (default), 1=mild, 0.2=noisy. Counts = I x noise")
ap.add_argument("--seed", type=int, default=12345, help="noise seed (same seed = same noise)")
ap.add_argument("--no-self-check", action="store_true", help="skip the independent ray-marching check")
a = ap.parse_args()

name = os.path.basename(os.path.normpath(a.out))
os.makedirs(a.out, exist_ok=True)

# ------------------------------------------------------------------ geometry (copied from your real scan)
sod, sdd = 69.1847839355469, 1147.123
du = dv = 0.2 * 2048 / a.nu          # keeps the same field of view as the real 2048-px detector
white = 60000.0
mag = sdd / sod
voxel = du / mag
uc = (a.nu - 1) / 2 + a.cor


# ------------------------------------------------------------------ materials
def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(l for l in f if not l.lstrip().startswith("#")))


def get_material(name_, rows):
    """Return dict(mu_cm1, density, formula, source, energy) for the requested energy/source, or exit."""
    mine = [r for r in rows if r["material"] == name_]
    if not mine:
        sys.exit(f"ERROR: material '{name_}' is not in {a.materials}. Known: "
                 f"{', '.join(sorted({r['material'] for r in rows}))}")
    fixed = [r for r in mine if r["source"] == "fixed"]
    if fixed:
        r = fixed[0]
        return dict(mu_cm1=float(r["mu_cm1"]), density=float(r["density_g_cm3"] or 0), formula=r["formula"],
                    source="fixed", energy=None)
    if a.source == "calc_eq7":
        import mu_calc
        r = mine[0]
        try:
            mu = mu_calc.mu_linear_cm(r["formula"], float(r["density_g_cm3"]), a.energy)
        except (KeyError, ValueError) as ex:
            sys.exit(f"ERROR computing '{name_}' with mu_calc.py: {ex}")
        return dict(mu_cm1=mu, density=float(r["density_g_cm3"]), formula=r["formula"],
                    source="calc_eq7", energy=a.energy)
    if a.source == "test_only":
        hit = [r for r in mine if r["source"] == "test_only"]
    else:
        hit = [r for r in mine if r["source"] == a.source and r["energy_keV"] != ""
               and abs(float(r["energy_keV"]) - a.energy) < 1e-6]
    if not hit:
        have = sorted({(r["energy_keV"] or "any", r["source"]) for r in mine})
        sys.exit(f"ERROR: no value for '{name_}' at {a.energy} keV with source '{a.source}'.\n"
                 f"  Available for {name_}: {have}\n"
                 f"  Choose a different --energy/--source, use --source calc_eq7, or add a row to materials.csv.")
    r = hit[0]
    return dict(mu_cm1=float(r["mu_cm1"]), density=float(r["density_g_cm3"]), formula=r["formula"],
                source=r["source"], energy=float(r["energy_keV"]) if r["energy_keV"] else None)


# ------------------------------------------------------------------ phantom definition
DEFAULT_GRAINS = [((-3.0, 0.5, 1.5), 1.0), ((1.5, 1.5, -2.0), 0.8), ((3.5, -2.5, 2.5), 0.6)]
BOX_HALF = [7.0, 4.0, 5.0]           # block: 14 x 8 x 10 mm (x, y, z)
CYL_R, CYL_HALF_H = 6.0, 5.0         # cylinder: 12 mm diameter, 10 mm tall, axis = z
legacy = (a.phantom == "spheres" and not a.phantom_file)

if a.phantom_file:
    with open(a.phantom_file) as f:
        spec = json.load(f)
    host_spec = spec["host"]
    grain_specs = spec.get("grains", [])
elif not legacy:
    host_spec = ({"shape": "box", "material": a.phantom, "half_size_mm": BOX_HALF} if a.shape == "box" else
                 {"shape": "cylinder", "material": a.phantom, "radius_mm": CYL_R, "half_height_mm": CYL_HALF_H})
    incl = ["zircon", "magnetite", "void"] if a.inclusion == "mixed" else [a.inclusion] * 3
    grain_specs = [{"material": m, "centre_mm": list(c), "radius_mm": r} for (c, r), m in zip(DEFAULT_GRAINS, incl)]

boxes, cylinders, spheres = [], [], []
used = {}
if legacy:
    print("NOTE: legacy 'spheres' phantom - arbitrary attenuation values, not from the paper.")
    spheres = [((0, 0, 0), 7.0, 0.5), ((3, -2, 0), 2.0, 1.0), ((-3.5, 3, 2), 1.2, 1.5),
               ((1, 4, -2.5), 1.0, 1.5), ((-2, -4, 1), 1.5, 1.5 * 2 / 3)]
    SUGGEST_HI = 1.8
else:
    rows = read_rows(a.materials)
    names = [host_spec["material"]] + [g["material"] for g in grain_specs]
    for n_ in dict.fromkeys(names):
        used[n_] = get_material(n_, rows)
    mu_mm = {n_: used[n_]["mu_cm1"] / 10.0 for n_ in used}           # cm^-1 -> mm^-1
    host = mu_mm[host_spec["material"]]
    hc = np.array(host_spec.get("centre_mm", [0, 0, 0]), float)
    if host_spec["shape"] == "box":
        boxes = [(tuple(hc), np.array(host_spec["half_size_mm"], float), host)]
    else:
        cylinders = [(tuple(hc), host_spec["radius_mm"], host_spec["half_height_mm"], host)]
    # grains are ADDED on top of the host: extra mu = (grain - host). Exact only if each grain is fully inside the
    # host and no two grains overlap, so check that (and stop if not).
    for i, g in enumerate(grain_specs):
        c, R = np.array(g["centre_mm"], float), float(g["radius_mm"])
        d = c - hc
        if host_spec["shape"] == "box":
            h = np.array(host_spec["half_size_mm"], float)
            inside = bool(np.all(np.abs(d) + R <= h))
        else:
            inside = np.hypot(d[0], d[1]) + R <= host_spec["radius_mm"] and abs(d[2]) + R <= host_spec["half_height_mm"]
        if not inside:
            sys.exit(f"ERROR: grain {i + 1} ({g['material']}) is not fully inside the host - move or shrink it.")
        for j in range(i):
            g2 = grain_specs[j]
            if np.linalg.norm(c - np.array(g2["centre_mm"], float)) < R + g2["radius_mm"]:
                sys.exit(f"ERROR: grains {j + 1} and {i + 1} overlap - separate them.")
        spheres.append((tuple(c), R, mu_mm[g["material"]] - host))
    SUGGEST_HI = 1.2 * max(mu_mm.values())
    print(f"Attenuation values (energy {a.energy} keV, source {a.source}):")
    for n_, m_ in used.items():
        print(f"  {n_:10s} {m_['formula']:14s} density {m_['density']:5.2f} g/cm3  "
              f"mu = {m_['mu_cm1']:8.3f} cm^-1 = {m_['mu_cm1'] / 10:.4f} mm^-1   [{m_['source']}"
              f"{'' if m_['energy'] is None else ' @ %.2f keV' % m_['energy']}]")
    for g in grain_specs:
        print(f"  grain {g['material']:10s} at {tuple(g['centre_mm'])} mm, r = {g['radius_mm']} mm")

# ------------------------------------------------------------------ forward projection (unchanged approach)
ang = -(np.arange(a.nproj) * 360.0 / a.nproj) - 0.01     # negative, like your .ang
u = (np.arange(a.nu) - uc) * du
v = ((a.nv - 1) / 2 - np.arange(a.nv)) * dv
UU, VV = np.meshgrid(u, v)
rng = np.random.default_rng(a.seed)


def project(deg):
    """Noise-free line integrals p (nv, nu) for one angle, plus source position (object frame)."""
    t_ = np.deg2rad(deg)
    c, s = np.cos(t_), np.sin(t_)
    src = np.array([-sod * s, sod * c, 0.0])             # source: t = 0, s = +SOD in the rotating frame
    sp = -(sdd - sod)                                    # detector plane: s = -(SDD - SOD)
    D = np.stack([UU * c - sp * s, UU * s + sp * c, VV], -1)
    d = D - src
    dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
    p = np.zeros_like(UU)
    for ctr, half, mu in boxes:                          # ray / box chord length (slab method)
        lo, hi = np.array(ctr) - half, np.array(ctr) + half
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - src) / dn
            t2 = (hi - src) / dn
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        p += mu * np.clip(tmax - tmin, 0, None)
    for ctr, R_c, hh, mu in cylinders:                   # ray / vertical cylinder chord length
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
    for ctr, R, mu in spheres:                           # ray / sphere chord length
        w = np.array(ctr, float) - src
        along = (w * dn).sum(-1)
        dist2 = (w ** 2).sum() - along ** 2
        p += mu * 2 * np.sqrt(np.clip(R ** 2 - dist2, 0, None))
    return p, src, dn


check_ks = sorted({0, a.nproj // 3, (2 * a.nproj) // 3})
kept = {}
t_min = 1.0
pc_row, pc_col = a.nv // 2, int(np.clip(round(uc), 0, a.nu - 1))
p_center = []
for k, deg in enumerate(ang):
    p, src, dn = project(deg)
    p_center.append(float(p[pc_row, pc_col]))
    t_min = min(t_min, float(np.exp(-p).min()))
    if k in check_ks:
        kept[k] = (p.copy(), src, dn)
    I = white * np.exp(-p)
    if a.noise > 0:
        I = rng.poisson(I * a.noise) / a.noise
    tifffile.imwrite(os.path.join(a.out, f"{name}_{k + 1:04d}.tif"),
                     np.rint(np.clip(I, 0, 65535)).astype(np.uint16))
    if k % 40 == 0:
        print(f"  projection {k + 1}/{a.nproj}", end="\r")
print()

# ------------------------------------------------------------------ .ang and .xtekct (format unchanged)
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

# ------------------------------------------------------------------ independent check by ray marching
check_result = None
if not a.no_self_check and not legacy:
    def mu_at(P):
        """mu (1/mm) at points P (...,3), built directly from 'what is inside what' (grains REPLACE the host)."""
        m = np.zeros(P.shape[:-1])
        if boxes:
            ctr, half, mu = boxes[0]
            m = np.where(np.all(np.abs(P - np.array(ctr)) <= half, axis=-1), mu, m)
        else:
            ctr, R_c, hh, mu = cylinders[0]
            Q = P - np.array(ctr)
            m = np.where((Q[..., 0] ** 2 + Q[..., 1] ** 2 <= R_c ** 2) & (np.abs(Q[..., 2]) <= hh), mu, m)
        for g, (c_, R_, mu_extra) in zip(grain_specs, spheres):
            inside = ((P - np.array(c_)) ** 2).sum(-1) <= R_ ** 2
            m = np.where(inside, mu_mm[g["material"]], m)
        return m

    step = 0.005
    diffs = []
    rng2 = np.random.default_rng(0)
    for k, (p, src, dn) in kept.items():
        hit = np.argwhere(p > 0.01)
        pick = [(pc_row, pc_col)] + [tuple(hit[i]) for i in rng2.choice(len(hit), size=min(10, len(hit)), replace=False)]
        for (i, j) in pick:
            d_ = dn[i, j]
            t0 = float(-(src @ d_))                      # closest approach of the ray to the origin
            ts = t0 + np.arange(-20, 20, step)
            pts = src[None, :] + ts[:, None] * d_[None, :]
            diffs.append(abs(float((mu_at(pts) * step).sum()) - float(p[i, j])))
    check_result = dict(rays=len(diffs), max_abs_dp=max(diffs), mean_abs_dp=float(np.mean(diffs)),
                        passed=bool(max(diffs) < 0.02))
    print(f"Self-check (analytic chords vs independent ray marching, {len(diffs)} rays): "
          f"max |dp| = {max(diffs):.4f}, mean = {np.mean(diffs):.4f}  ->  "
          f"{'PASS' if check_result['passed'] else 'WARNING: differs by more than 0.02 - investigate'}")

# ------------------------------------------------------------------ hand-check numbers and truth.json
print(f"Wrote {a.nproj} projections to {a.out}")
hand = None
if cylinders:
    p_host = cylinders[0][3] * 2 * cylinders[0][1]
    I_exp = white * np.exp(-p_host)
    hand = dict(pixel_row=pc_row, pixel_col=pc_col, host_only_p=p_host, transmission=float(np.exp(-p_host)),
                expected_grey_value=float(I_exp), analytic_p_min=min(p_center), analytic_p_max=max(p_center))
    print(f"Hand check (pixel row {pc_row}, col {pc_col}, the ray through the rotation axis): "
          f"p = mu x diameter = {p_host:.4f}, transmission {100 * np.exp(-p_host):.2f} %, "
          f"expected grey value ~{I_exp:.0f} in every projection (noise-free)")
print(f"Lowest transmission anywhere: {100 * t_min:.3f} %"
      + ("  <- very low: expect noisy/photon-starved rays if you use --noise" if t_min < 0.005 else ""))
if not legacy:
    print(f"Suggested reconstruction flag so every material is visible:  --grey-range 0 {SUGGEST_HI:.3f}")
print(f"TRUE centre-of-rotation offset: {a.cor:+.2f} px  (voxel {voxel:.4f} mm)")

truth = {
    "generated_by": "sim_scan.py (version with materials.csv / truth.json)",
    "command_line": " ".join(sys.argv),
    "geometry": {"SrcToObject_mm": sod, "SrcToDetector_mm": sdd, "detector_pixels": [a.nu, a.nv],
                 "detector_pixel_mm": du, "real_detector_pixels": [2048, 2048], "real_detector_pixel_mm": 0.2,
                 "note": "detector is a coarser stand-in for the real one, same field of view",
                 "voxel_size_mm": voxel, "true_cor_px": a.cor,
                 "cor_definition": "rotation axis projects at column (nu-1)/2 + cor_px (same as ct_recon's cor_px)"},
    "acquisition": {"n_projections": a.nproj, "angles_deg": "-(k*360/N) - 0.01, k = 0..N-1", "white_level": white,
                    "noise_scale": a.noise, "seed": a.seed,
                    "lowest_transmission": t_min},
    "attenuation": {"energy_keV": a.energy, "source": a.source,
                    "materials": {k: dict(m_, mu_mm1=m_["mu_cm1"] / 10.0) for k, m_ in used.items()},
                    "caveats": ["single effective energy, no beam hardening, scatter or spectrum",
                                "values tied to the listed densities",
                                "paper Table 2 values are calculated, not measured"]},
    "phantom": ({"legacy_spheres": True} if legacy else {"host": host_spec, "grains": grain_specs}),
    "self_check": check_result,
    "hand_check": hand,
}
if not legacy:
    n_ = a.nu
    truth["expected_in_reconstruction"] = {
        "note": "for ct_recon.py run with --bin 1 --step 1 and no --z. Indices are 0-based [z_slice, row, col]; "
                "Fiji shows slices 1-based. With --bin b divide positions by b. Values are mu in 1/mm "
                "(before the grey-range mapping).",
        "host_mu_mm1": mu_mm[host_spec["material"]],
        "grains": [{"material": g["material"], "mu_mm1": mu_mm[g["material"]],
                    "index_zyx": [round((a.nv - 1) / 2 - g["centre_mm"][2] / voxel, 1),
                                  round((n_ - 1) / 2 + g["centre_mm"][1] / voxel, 1),
                                  round((n_ - 1) / 2 + g["centre_mm"][0] / voxel, 1)],
                    "radius_voxels": round(g["radius_mm"] / voxel, 1)} for g in grain_specs]}
with open(os.path.join(a.out, "truth.json"), "w") as f:
    json.dump(truth, f, indent=2)
print(f"Wrote {os.path.join(a.out, 'truth.json')}")
