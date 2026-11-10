#!/usr/bin/env python3
"""
spectrum.py - a deliberately simple tungsten-target X-ray spectrum model for sim_scan.py --spectrum.

NOT measured data and NOT from the MXLAC paper (the paper shows measured spectra only as a figure, and gives
only the resulting effective energies). What is modelled:
  continuum        Kramers' law: photons per keV  ~  (kV - E) / E
  characteristic   tungsten K lines (Ka1 59.32, Ka2 57.98, Kb1 67.24, Kb2 69.07 keV; approximate relative
                   strengths), present only above the W K-edge (69.525 keV), strength ~ (kV/69.525 - 1)^1.63
  filtration       exp(-mu/rho * rho * t) for aluminium and copper (NIST data in nist_elements.csv):
                   an 'inherent' aluminium-equivalent thickness (fitted, see below) plus any added Al / Cu filter
Not modelled: target self-absorption, anode angle / heel effect, tube window details, detector response,
scatter. Energy range 10 keV to kV (attenuation data end at 150 keV, so kV <= 150).

Calibration against the paper: the paper gives the effective energy (count-weighted mean photon energy of its
measured spectra, its Eq. 2) for each tube voltage (its Fig. 4 table). Two parameters were chosen using the
60-150 kV values: inherent filtration 7.5 mm Al-equivalent (fitted) and line strength char_scale 1.0 (chosen so
the K lines are ~10 % of photons at 140 kV, roughly what the paper's Fig. 1 shows by eye). Result: differences
of -1.5 to +2.1 keV, rms 1.25 keV (+0.9 keV at 130 kV); run `python spectrum.py` for the table. The residuals
have a trend (the model rises a little faster with kV than the paper's table). 7.5 mm Al is a fitted nuisance
parameter that makes the model reproduce the paper's fairly hard effective energies; it is NOT a measurement of
any real tube. The paper's effective energies also do not include added filters (its Table 1 uses the same
45.5 keV for every filter), so an added filter here shifts the beam further than the paper's table would say.
"""
import sys
import numpy as np
import mu_calc

W_K_EDGE = 69.525
K_LINES = [(59.318, 1.0), (57.981, 0.58), (67.244, 0.21), (69.067, 0.07)]   # (keV, relative weight), approximate
RHO = {"Al": 2.699, "Cu": 8.96}                                                # g/cm3, standard values
DEFAULT_INHERENT_AL_MM = 7.5
DEFAULT_CHAR_SCALE = 1.0
MAX_KV = 150.0
# MXLAC paper, Fig. 4 reference table: tube voltage (kV) -> effective energy (keV). Only the part up to 150 kV.
PAPER_EFFECTIVE_KEV = {60: 41.70, 70: 45.50, 80: 49.10, 90: 52.30, 100: 55.10, 110: 57.40,
                       120: 59.50, 130: 61.30, 140: 63.10, 150: 64.80}


def spectrum(kv, inherent_al_mm=DEFAULT_INHERENT_AL_MM, char_scale=DEFAULT_CHAR_SCALE,
             filter_cu_mm=0.0, filter_al_mm=0.0, emin=10.0, dE=1.0):
    """Relative photon numbers N(E) per energy bin (bin centres E, width dE), after all filtration."""
    if kv > MAX_KV:
        raise ValueError(f"kV {kv} > {MAX_KV}: the attenuation data in nist_elements.csv end at 150 keV")
    E = np.arange(emin + dE / 2, kv, dE)
    N = (kv - E) / E * dE
    if kv > W_K_EDGE and char_scale > 0:
        n_k = char_scale * 10.0 * (kv / W_K_EDGE - 1.0) ** 1.63
        tot = sum(w for _, w in K_LINES)
        for e, w in K_LINES:
            if e < kv:
                N[int(np.argmin(np.abs(E - e)))] += n_k * w / tot
    mu_al = mu_calc.mu_rho_array("Al", E) * RHO["Al"] / 10.0      # 1/mm
    mu_cu = mu_calc.mu_rho_array("Cu", E) * RHO["Cu"] / 10.0
    T = np.exp(-mu_al * (inherent_al_mm + filter_al_mm) - mu_cu * filter_cu_mm)
    return E, N * T


def mean_energy_counts(E, N):
    """Count-weighted mean photon energy - the paper's 'effective energy' definition (its Eq. 2)."""
    return float((N * E).sum() / N.sum())


def detector_weights(E, N):
    """Weights of an energy-integrating detector (signal ~ photon energy). Sum to 1."""
    w = N * E
    return w / w.sum()


def mean_energy_weighted(E, N):
    """Energy-weighted mean photon energy = mean energy seen by an energy-integrating detector."""
    w = detector_weights(E, N)
    return float((w * E).sum())


def calibration_table():
    print("kV    paper (Fig.4)   model (count-weighted mean)   difference")
    errs = []
    for kv, ref in PAPER_EFFECTIVE_KEV.items():
        E, N = spectrum(kv)
        m = mean_energy_counts(E, N)
        errs.append(m - ref)
        print(f"{kv:3d}   {ref:8.2f} keV      {m:8.2f} keV               {m - ref:+.2f} keV")
    print(f"rms difference {np.sqrt(np.mean(np.square(errs))):.2f} keV  "
          f"(inherent filtration {DEFAULT_INHERENT_AL_MM} mm Al-equivalent, char_scale {DEFAULT_CHAR_SCALE}; "
          f"fitted, see the docstring)")


if __name__ == "__main__":
    calibration_table()
    print()
    for cu in (0.0, 0.25):
        E, N = spectrum(130, filter_cu_mm=cu)
        print(f"130 kV, added Cu {cu:4.2f} mm: count-weighted mean {mean_energy_counts(E, N):.1f} keV, "
              f"energy-weighted mean {mean_energy_weighted(E, N):.1f} keV")
