"""
Where does the closed-form readout reference stop being trustworthy as the
fringes tilt?

The committed RCWA grid (experiments/rcwa_crosscheck.py, run_e7_grid) checks
the closed-form Kogelnik reference against RCWA at slant 0 and 20 degrees and
nothing in between, so it cannot say where the deviation sets in. The paired
gain, meanwhile, collapses between 5 and 7.5 degrees (tier S8), and the two
facts cannot be compared without intermediate angles. This script fills in
2.5, 5, 7.5, 10 and 15 degrees on the same machinery.

Two K values are those of the S8 slant sweep (1.9635 and 3.9270 rad/um) so the
deviation can be read at exactly the angles where the gain was measured; 6 and
12 rad/um are carried over from the committed grid for continuity.

Both incidence conventions are swept, because they are not interchangeable
here. The committed grid uses Bragg incidence, but SlabBPM reads the slab out
with a normally incident plane wave, so "normal" is the geometry that matches
the design loop and "bragg" is the one that matches the committed grid.

Scope, stated here so the manuscript cannot overstate it: this compares
UNSLANTED Kogelnik (which has no slant term) against RCWA on a genuinely
slanted grating, which is the same comparison run_e7_grid makes. It therefore
bounds the error of the CLOSED-FORM reference as the slant grows, most of which
is Bragg detuning that the closed form omits by construction. It does NOT
measure the error of the split-step beam propagation used inside the design
loop, which applies its own per-slice lateral shear and does model the
geometry. A direct beam-propagation-against-RCWA comparison is a separate
experiment and is not done here.

Usage: python experiments/rcwa_slant_sweep.py   (CPU, about 2 minutes)
Writes results/applied_optics/rcwa_slant_sweep.json
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch

from holomedia import kogelnik_de

try:
    import torcwa  # noqa: F401
except ImportError:
    print("torcwa not installed (`pip install torcwa`); skipping slant sweep.")
    sys.exit(0)

from rcwa_crosscheck import (THICKNESS_UM, WAVELENGTH_UM, bragg_angle_deg,
                             rcwa_de_general)

# The two S8 K values, plus two from the committed grid for continuity.
K_VALUES = [1.9635, 3.9270, 6.0, 12.0]
S8_K_VALUES = [1.9635, 3.9270]
DN_VALUES = [1.0e-3, 3.5e-3, 6.0e-3]
SLANTS_DEG = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
POLARIZATIONS = ["ss", "pp"]
# "normal" matches SlabBPM's normally incident readout (the design loop);
# "bragg" matches the committed run_e7_grid comparison.
INCIDENCES = ["normal", "bragg"]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "applied_optics")
OUT_PATH = os.path.join(OUT_DIR, "rcwa_slant_sweep.json")


def main() -> int:
    n_cases = (len(K_VALUES) * len(DN_VALUES) * len(SLANTS_DEG)
               * len(POLARIZATIONS) * len(INCIDENCES))
    print(f"slant sweep: {n_cases} cases "
          f"({len(K_VALUES)} K x {len(DN_VALUES)} dn x {len(SLANTS_DEG)} slants "
          f"x {len(POLARIZATIONS)} polarizations x {len(INCIDENCES)} incidences)")

    cases, i = [], 0
    for K in K_VALUES:
        tB = bragg_angle_deg(K)
        theta_B_rad = math.radians(tB)
        for dn in DN_VALUES:
            for incidence in INCIDENCES:
                # The closed form carries no slant term either way; at normal
                # incidence it is additionally detuned by the full Bragg angle,
                # exactly as run_e7_grid does for its "unslanted_normal" row.
                if incidence == "normal":
                    dtheta = torch.tensor(0.0 - theta_B_rad)
                    eta_kog = float(kogelnik_de(torch.tensor(dn), THICKNESS_UM,
                                                WAVELENGTH_UM, theta_B=theta_B_rad,
                                                dtheta=dtheta))
                else:
                    eta_kog = float(kogelnik_de(torch.tensor(dn), THICKNESS_UM,
                                                WAVELENGTH_UM, theta_B=theta_B_rad))
                for slant in SLANTS_DEG:
                    for pol in POLARIZATIONS:
                        i += 1
                        rec = dict(K=K, dn=dn, slant_deg=slant, polarization=pol,
                                   incidence=incidence, theta_B_deg=tB,
                                   kogelnik=eta_kog)
                        try:
                            de0, de1 = rcwa_de_general(K, dn, tB, polarization=pol,
                                                       slant_deg=slant,
                                                       incidence=incidence)
                            rec.update(rcwa_t0=de0, rcwa_t1=de1,
                                       abs_deviation=abs(eta_kog - de1))
                        except Exception as e:
                            rec.update(error=str(e))
                        cases.append(rec)
                        if i % 50 == 0 or i == n_cases:
                            print(f"  {i}/{n_cases} done", flush=True)

    valid = [c for c in cases if c.get("abs_deviation") is not None]

    def _median(vals):
        v = sorted(vals)
        mid = len(v) // 2
        return v[mid] if len(v) % 2 else 0.5 * (v[mid - 1] + v[mid])

    def summarize(subset):
        vals = [c["abs_deviation"] for c in subset]
        if not vals:
            return dict(n=0)
        t1 = [c["rcwa_t1"] for c in subset]
        t0 = [c["rcwa_t0"] for c in subset]
        return dict(n=len(vals), mean=sum(vals) / len(vals),
                    median=_median(vals), max=max(vals), min=min(vals),
                    # the mechanism evidence: where the diffracted order goes
                    median_rcwa_first_order=_median(t1),
                    median_rcwa_zero_order=_median(t0),
                    median_kogelnik=_median([c["kogelnik"] for c in subset]))

    def sweep_for(incidence, K_filter=None):
        sel = [c for c in valid if c["incidence"] == incidence
               and (K_filter is None or c["K"] in K_filter)]
        by_slant = {f"{s:g}": summarize([c for c in sel if c["slant_deg"] == s])
                    for s in SLANTS_DEG}
        base = by_slant["0"].get("median")
        # Smallest TESTED slant at which the median deviation first exceeds
        # twice the unslanted median. Reported as a grid point, never
        # interpolated: the grid is coarse and a crossing between two
        # angles is not resolved.
        onset = None
        if base is not None:
            for s in SLANTS_DEG[1:]:
                m = by_slant[f"{s:g}"].get("median")
                if m is not None and m > 2.0 * base:
                    onset = s
                    break
        return dict(by_slant=by_slant, unslanted_median=base,
                    median_doubling_onset_deg=onset)

    by_incidence = {inc: sweep_for(inc) for inc in INCIDENCES}
    by_incidence_s8_K_only = {inc: sweep_for(inc, K_filter=S8_K_VALUES)
                              for inc in INCIDENCES}

    out = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        K_values=K_VALUES, s8_K_values=S8_K_VALUES, dn_values=DN_VALUES,
        slants_deg=SLANTS_DEG, polarizations=POLARIZATIONS,
        incidences=INCIDENCES,
        thickness_um=THICKNESS_UM, wavelength_um=WAVELENGTH_UM,
        n_cases=n_cases, n_valid=len(valid),
        by_incidence=by_incidence,
        by_incidence_s8_K_only=by_incidence_s8_K_only,
        max_abs_deviation=max((c["abs_deviation"] for c in valid), default=None),
        cases=cases,
        caveat="Unslanted Kogelnik (no slant term) against RCWA on a slanted "
               "grating, as in run_e7_grid. Bounds the closed-form reference, "
               "not the split-step beam propagation used in the design loop. "
               "'normal' is the incidence SlabBPM actually uses.",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.normpath(OUT_PATH)}")

    for inc in INCIDENCES:
        rec = by_incidence[inc]
        print(f"\n  incidence = {inc}"
              f"{'   (matches the design loop readout)' if inc == 'normal' else ''}")
        print("  slant   median|Kog-RCWA|   max      RCWA 1st order   RCWA 0th order")
        for s in SLANTS_DEG:
            a = rec["by_slant"][f"{s:g}"]
            print(f"  {s:5.1f}      {a.get('median', float('nan')):.4f}"
                  f"          {a.get('max', float('nan')):.4f}"
                  f"        {a.get('median_rcwa_first_order', float('nan')):.4f}"
                  f"           {a.get('median_rcwa_zero_order', float('nan')):.4f}")
        print(f"  median first exceeds 2x the unslanted median at: "
              f"{rec['median_doubling_onset_deg'] if rec['median_doubling_onset_deg'] is not None else 'no tested angle'} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
