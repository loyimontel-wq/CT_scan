#!/usr/bin/env python3
"""
mu_calc.py - linear attenuation coefficient of a mineral from its chemical formula.

Method (MXLAC paper, Eq. 6 and 7):  mu/rho = sum_i w_i (mu/rho)_i ;   mu = (mu/rho) * density
  w_i       = mass fraction of element i, from the formula and standard atomic masses
  (mu/rho)_i = NIST element value (nist_elements.csv), interpolated linearly in log(E)-log(mu/rho)
Difference from the paper: the authors fitted polynomials to the NIST data; this uses log-log interpolation.
Valid 30-150 keV only, for elements in nist_elements.csv, single (effective) energy, no beam hardening.

    python mu_calc.py SiO2 2.65 62.53          one mineral
    python mu_calc.py --verify                 compare with the paper's Table 2 values in materials.csv
    python mu_calc.py --add zircon ZrSiO4 4.71 61.3   append a calc_eq7 row to materials.csv
"""
import csv, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Standard atomic weights (IUPAC, rounded). Not from the paper.
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "O": 15.999, "F": 18.998, "Mg": 24.305, "Al": 26.982,
               "Si": 28.085, "P": 30.974, "S": 32.06, "Ca": 40.078, "Cr": 51.996, "Fe": 55.845,
               "Zn": 65.38, "Zr": 91.224, "Ba": 137.327}


def parse_formula(formula):
    f = formula.replace(" ", "")
    pos = 0

    def number():
        nonlocal pos
        m = re.match(r"\d+\.?\d*", f[pos:])
        if m:
            pos += m.end()
            return float(m.group())
        return 1.0

    def group():
        nonlocal pos
        counts = {}
        while pos < len(f):
            ch = f[pos]
            if ch == "(":
                pos += 1
                sub = group()
                mult = number()
                for k, v in sub.items():
                    counts[k] = counts.get(k, 0) + v * mult
            elif ch == ")":
                pos += 1
                return counts
            else:
                m = re.match(r"[A-Z][a-z]?", f[pos:])
                if not m:
                    raise ValueError(f"cannot parse formula '{formula}' at '{f[pos:]}' "
                                     f"(hydrates written with a dot are not supported)")
                pos += m.end()
                counts[m.group()] = counts.get(m.group(), 0) + number()
        return counts

    return group()


def mass_fractions(formula):
    counts = parse_formula(formula)
    missing = [e for e in counts if e not in ATOMIC_MASS]
    if missing:
        raise KeyError(f"no atomic mass for {missing} - add to ATOMIC_MASS in mu_calc.py")
    tot = sum(ATOMIC_MASS[e] * n for e, n in counts.items())
    return {e: ATOMIC_MASS[e] * n / tot for e, n in counts.items()}


_elements = None


def load_elements(path=None):
    global _elements
    if _elements is None:
        path = path or os.path.join(HERE, "nist_elements.csv")
        rows = csv.DictReader(l for l in open(path) if not l.lstrip().startswith("#"))
        d = {}
        for r in rows:
            d.setdefault(r["symbol"], []).append((float(r["energy_keV"]), float(r["mu_rho_cm2_g"])))
        _elements = {k: (np.array(sorted(v))[:, 0], np.array(sorted(v))[:, 1]) for k, v in d.items()}
    return _elements


def mu_rho_element(sym, energy_keV):
    els = load_elements()
    if sym not in els:
        raise KeyError(f"element {sym} is not in nist_elements.csv (loaded: {', '.join(els)}). "
                       f"Add its NIST rows - see the header of that file.")
    E, M = els[sym]
    if not (E[0] <= energy_keV <= E[-1]):
        raise ValueError(f"energy {energy_keV} keV outside the supported {E[0]:.0f}-{E[-1]:.0f} keV")
    return float(np.exp(np.interp(np.log(energy_keV), np.log(E), np.log(M))))


def mu_linear_cm(formula, density, energy_keV):
    """Linear attenuation coefficient in cm^-1."""
    w = mass_fractions(formula)
    mu_rho = sum(w[e] * mu_rho_element(e, energy_keV) for e in w)
    return mu_rho * density


def _read_materials(path):
    return list(csv.DictReader(l for l in open(path) if not l.lstrip().startswith("#")))


def verify(path):
    rows = _read_materials(path)
    print(f"{'mineral':10s} {'keV':>6s} {'paper NIST':>10s} {'paper MXLAC':>11s} {'calc (eq7)':>10s} {'calc vs NIST':>12s}")
    worst = 0.0
    seen = set()
    for r in rows:
        if r["source"] != "paper_T2_NIST":
            continue
        key = (r["material"], r["energy_keV"])
        if key in seen:
            continue
        seen.add(key)
        mx = [x for x in rows if x["material"] == r["material"] and x["energy_keV"] == r["energy_keV"]
              and x["source"] == "paper_T2_MXLAC"]
        try:
            c = mu_linear_cm(r["formula"], float(r["density_g_cm3"]), float(r["energy_keV"]))
        except (KeyError, ValueError) as ex:
            m = re.search(r"element (\w+)", str(ex))
            print(f"{r['material']:10s} {float(r['energy_keV']):6.2f}   skipped (needs NIST data for {m.group(1) if m else '?'})")
            continue
        n = float(r["mu_cm1"])
        err = 100 * (c - n) / n
        worst = max(worst, abs(err))
        print(f"{r['material']:10s} {float(r['energy_keV']):6.2f} {n:10.3f} {float(mx[0]['mu_cm1']) if mx else float('nan'):11.3f} "
              f"{c:10.3f} {err:+11.1f}%")
    print(f"largest |difference| vs the paper's NIST column: {worst:.1f}%  (the paper reports MXLAC-vs-NIST up to ~5%)")


def main(argv):
    if len(argv) >= 2 and argv[1] == "--verify":
        verify(os.path.join(HERE, "materials.csv")); return
    if len(argv) == 6 and argv[1] == "--add":
        name, formula, dens, en = argv[2], argv[3], float(argv[4]), float(argv[5])
        mu = mu_linear_cm(formula, dens, en)
        with open(os.path.join(HERE, "materials.csv"), "a", newline="") as f:
            csv.writer(f).writerow([name, formula, dens, en, f"{mu:.4f}", "calc_eq7", "computed by mu_calc.py"])
        print(f"added {name}: {mu:.4f} cm^-1 at {en} keV"); return
    if len(argv) == 4:
        formula, dens, en = argv[1], float(argv[2]), float(argv[3])
        w = mass_fractions(formula)
        print(f"{formula}  density {dens} g/cm3  energy {en} keV")
        for e, wi in w.items():
            print(f"  {e:2s} weight fraction {wi:.4f}   (mu/rho)_i = {mu_rho_element(e, en):.4f} cm2/g")
        mu = mu_linear_cm(formula, dens, en)
        print(f"  mu = {mu:.4f} cm^-1  = {mu / 10:.5f} mm^-1"); return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
