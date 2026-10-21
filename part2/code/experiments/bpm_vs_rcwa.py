"""
Does the propagator inside the design loop agree with RCWA?

Every previous check in this repository compares the CLOSED-FORM Kogelnik
reference against RCWA (`experiments/rcwa_crosscheck.py` and
`experiments/rcwa_slant_sweep.py`). Neither touches `holomedia.diffraction.
SlabBPM`, the split-step beam propagation that actually sits inside the
optimization loop and produces every reconstruction the paper scores. The
manuscript has had to say so, and that caveat is load-bearing: it is why the
validated envelope is restricted to unslanted gratings. This script closes
that gap by comparing SlabBPM with RCWA directly, on the same sinusoidal
gratings, with the conventions matched explicitly.

CONVENTIONS, stated because a silent mismatch here would put a confidently
wrong number in a submitted paper.

  Grating.     dn(x) = dn * cos(K x), extruded through T = 30 um and sheared
               laterally by z*tan(phi) at depth z, which is exactly what both
               SlabBPM.forward and rcwa_order_powers do for a slant.

  Incidence.   Two cases, and they are not interchangeable.
               "normal" is what SlabBPM.forward uses by default (incident=None
               gives a uniform field), so it is the geometry the design loop
               actually reads out in, and it is the primary result here.
               "bragg" illuminates both engines at the symmetric two-wave
               Bragg angle. For the BPM that means handing forward() an
               incident plane wave exp(i*k_x*x) with k_x = +K/2 inside the
               medium, which is the same relation rcwa_crosscheck.
               bragg_angle_deg encodes: sin(theta_B) = K*lambda/(4*pi*n0)
               gives k0*n0*sin(theta_B) = K/2. RCWA refracts that to air by
               Snell and enters at -theta_air.

  Commensurability. At Bragg incidence the incident transverse frequency is
               exactly K/2, so over the periodic window L = n_x*dx it is an
               integer number of cycles whenever L/(2*Lambda) is an integer.
               Both tested K satisfy this (L/Lambda = 16 and 32), so the
               incident wave and every order land exactly on an FFT bin and
               nothing leaks. The script asserts this rather than assuming it.

  Order index. The field is taken immediately AFTER the slab and before the
               free-space step to the reconstruction plane, and Fourier
               transformed. Order m sits at bin b_inc + m*(L/Lambda), where
               b_inc is the incident bin (0 for normal, L/(2*Lambda) for
               Bragg). RCWA's m is the same m: rcwa_de_general documents that
               for an unslanted transmission grating the Bragg-matched order
               is m=+1, and the BPM shift by +K lands on the same wave.

  Normalization. SlabBPM models no interfaces, so it has no Fresnel loss,
               while RCWA has air on both sides of an n0 = 1.5 slab and
               therefore loses a few percent per surface. Comparing raw
               efficiencies would charge the BPM for physics it does not claim
               to model. The comparison quantity is instead the first-order
               FRACTION among the three low orders,

                   eta1 = P(+1) / [P(-1) + P(0) + P(+1)],

               computed identically from both engines. That cancels any
               common transmission factor and isolates how the grating
               distributes power between the orders, which is what determines
               the reconstruction in this paraxial regime. Raw per-order
               values are written to the output file too, so the
               normalization can be audited.

ANCHOR. Before sweeping anything, `anchor_against_kogelnik()` checks the BPM
extraction at unslanted Bragg incidence against the closed-form Kogelnik
efficiency, in the regime where the committed RCWA grid already shows Kogelnik
is accurate. If the conventions above were wrong, that check fails and the
sweep does not run.

Usage: python experiments/bpm_vs_rcwa.py   (CPU, a few minutes)
Writes results/applied_optics/bpm_vs_rcwa.json
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
from holomedia.diffraction import SlabBPM

try:
    import torcwa  # noqa: F401
except ImportError:
    print("torcwa not installed (`pip install torcwa`); skipping.")
    sys.exit(0)

from rcwa_crosscheck import (N0, THICKNESS_UM, WAVELENGTH_UM, bragg_angle_deg,
                             rcwa_order_powers)

torch.set_default_dtype(torch.float64)

# Readout grid, matching the main campaign (Table 1).
N_X, DX, N_Z = 1024, 0.05, 128
WINDOW_UM = N_X * DX

# The two spatial frequencies of the S8 slant sweep, so the deviation can be
# read at exactly the angles where the paired gain was measured. They are
# built from the integer period count rather than written as rounded decimals:
# the campaign's K_nominal values are themselves 2*pi*n/window, and rounding
# them to 1.9635 and 3.9270 would make the grating incommensurate with the
# periodic window by a few parts in 10^6, which is enough to leak power out of
# the order bins.
PERIODS_PER_WINDOW = [16, 32]
K_VALUES = [2 * math.pi * n / WINDOW_UM for n in PERIODS_PER_WINDOW]
DN_VALUES = [1.0e-3, 3.5e-3, 6.0e-3]
SLANTS_DEG = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
INCIDENCES = ["normal", "bragg"]
ORDERS = (-1, 0, 1)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "applied_optics")
OUT_PATH = os.path.join(OUT_DIR, "bpm_vs_rcwa.json")


def _orders_per_window(K_um: float) -> float:
    """L / Lambda: how many grating periods fit the periodic window, which is
    also the FFT bin spacing between adjacent diffraction orders."""
    return WINDOW_UM * K_um / (2 * math.pi)


def bpm_order_powers(K_um: float, dn: float, slant_deg: float,
                     incidence: str) -> dict:
    """Transmitted power per diffraction order from SlabBPM, as {m: P(m)}.

    Takes the field immediately after the slab (the slab loop of
    SlabBPM.forward, without its final free-space step), Fourier transforms it
    and reads the bins the orders land on. See the module docstring for the
    bin arithmetic and why no free-space step is applied.
    """
    per_window = _orders_per_window(K_um)
    n_per = round(per_window)
    if abs(per_window - n_per) > 1e-6:
        raise ValueError(f"K={K_um} gives {per_window} periods per window; "
                         f"the grating must be commensurate with the window")

    x = torch.arange(N_X, dtype=torch.float64) * DX
    dn_profile = dn * torch.cos(K_um * x)

    bpm = SlabBPM(N_X, DX, WAVELENGTH_UM, THICKNESS_UM, n_z=N_Z, n0=N0)

    if incidence == "normal":
        incident = torch.ones(N_X, dtype=torch.complex128)
        b_inc = 0
    elif incidence == "bragg":
        # |k_x| = k0*n0*sin(theta_B) = K/2, the relation
        # rcwa_crosscheck.bragg_angle_deg encodes. The SIGN is set by RCWA's
        # convention, which rcwa_de_general documents: the incident wave
        # enters at -theta_air, so k_x = -K/2, and the Bragg-matched order is
        # then m=+1, landing at -K/2 + K = +K/2. Illuminating at +K/2 instead
        # would put the matched partner at m=-1 and leave m=+1 far off Bragg,
        # which is a silent factor-of-twenty error rather than a crash.
        k_x = -K_um / 2.0
        cycles = k_x * WINDOW_UM / (2 * math.pi)
        if abs(cycles - round(cycles)) > 1e-6:
            raise ValueError(f"Bragg tilt at K={K_um} is {cycles} cycles over "
                             f"the window; not commensurate, so the FFT would "
                             f"leak")
        incident = torch.exp(1j * k_x * x).to(torch.complex128)
        b_inc = int(round(cycles))
    else:
        raise ValueError(f"unknown incidence {incidence!r}")

    # Replay SlabBPM's slab loop, stopping before its free-space propagation:
    # order powers are defined on the field leaving the slab.
    E = incident.clone()
    tan_phi = math.tan(math.radians(slant_deg))
    fx = torch.fft.fftfreq(N_X, d=DX)
    dn_hat = torch.fft.fft(dn_profile.to(torch.complex128))
    for iz in range(N_Z):
        z = (iz + 0.5) * bpm.dz
        dn_z = torch.fft.ifft(dn_hat * torch.exp(-2j * math.pi * fx * tan_phi * z)).real
        E = E * torch.exp(1j * bpm.k0 * dn_z * bpm.dz)
        E = torch.fft.ifft(torch.fft.fft(E) * bpm.H_slab)

    spectrum = torch.fft.fft(E)
    total = float((spectrum.abs() ** 2).sum())
    return {int(m): float(spectrum[(b_inc + m * n_per) % N_X].abs() ** 2) / total
            for m in ORDERS}


def first_order_fraction(powers: dict) -> float:
    denom = sum(powers[m] for m in ORDERS)
    return powers[1] / denom if denom > 0 else float("nan")


# Kogelnik's two-wave theory is a small-signal, high-Q result, and this
# repository's own RCWA grid already shows where it stops being a usable
# reference: run_e7_grid's finding note records the deviation as "concentrated
# at high dn (>=6e-3)". The anchor therefore gates only on the modulations
# where the closed form is trustworthy. The strongest case is still computed
# and printed, as a diagnostic, but a disagreement there is Kogelnik's and
# must not be allowed to veto the sweep.
ANCHOR_DN_VALUES = [1.0e-3, 3.5e-3]


def anchor_against_kogelnik(tol: float = 0.06) -> bool:
    """At unslanted Bragg incidence the BPM's first-order fraction must track
    the closed-form Kogelnik efficiency wherever the closed form is itself
    reliable. If it does not, the conventions in the module docstring are
    wrong somewhere, and nothing downstream can be trusted.

    Note what this does and does not establish. It fixes the bookkeeping: the
    incidence sign, which FFT bin each order lands in, and the normalization.
    It is not independent evidence about the propagator, since Kogelnik and
    the BPM share the scalar paraxial approximation. The sweep against RCWA is
    what tests the physics.
    """
    print("anchor: BPM first-order fraction vs Kogelnik, unslanted, Bragg "
          f"incidence (gated on dn <= {max(ANCHOR_DN_VALUES):.1e}, "
          f"tolerance {tol})")
    worst, ok = 0.0, True
    for K in K_VALUES:
        for dn in DN_VALUES:
            theta_B = math.radians(bragg_angle_deg(K))
            eta_kog = float(kogelnik_de(torch.tensor(dn), THICKNESS_UM,
                                        WAVELENGTH_UM, theta_B=theta_B))
            powers = bpm_order_powers(K, dn, 0.0, "bragg")
            eta_bpm = first_order_fraction(powers)
            dev = abs(eta_bpm - eta_kog)
            gated = dn in ANCHOR_DN_VALUES
            if gated:
                worst = max(worst, dev)
                ok &= dev <= tol
                flag = "" if dev <= tol else "   <-- OUT OF TOLERANCE"
            else:
                flag = "   (diagnostic only, Kogelnik unreliable at this dn)"
            print(f"  K={K:7.4f} dn={dn:.1e}  Kogelnik={eta_kog:.4f}  "
                  f"BPM={eta_bpm:.4f}  |dev|={dev:.4f}{flag}")
    print(f"  worst gated deviation {worst:.4f}\n")
    return ok


def main() -> int:
    anchored = anchor_against_kogelnik()
    if not anchored:
        print("ANCHOR FAILED. The BPM order extraction does not reproduce the\n"
              "closed form where the closed form is known to be good, so the\n"
              "conventions are wrong somewhere. Not writing a result file:\n"
              "a number from a miscalibrated comparison is worse than none.")
        return 1

    n_cases = (len(K_VALUES) * len(DN_VALUES) * len(SLANTS_DEG)
               * len(INCIDENCES))
    print(f"sweep: {n_cases} cases "
          f"({len(K_VALUES)} K x {len(DN_VALUES)} dn x {len(SLANTS_DEG)} "
          f"slants x {len(INCIDENCES)} incidences)")

    cases, i = [], 0
    for incidence in INCIDENCES:
        for K in K_VALUES:
            tB = bragg_angle_deg(K)
            for dn in DN_VALUES:
                for slant in SLANTS_DEG:
                    i += 1
                    rec = dict(K=K, dn=dn, slant_deg=slant, incidence=incidence,
                               theta_B_deg=tB)
                    try:
                        bp = bpm_order_powers(K, dn, slant, incidence)
                        rp = rcwa_order_powers(K, dn, tB, orders=ORDERS,
                                               polarization="ss",
                                               slant_deg=slant,
                                               incidence=incidence)
                        eb, er = first_order_fraction(bp), first_order_fraction(rp)
                        rec.update(
                            bpm_orders={str(k): v for k, v in bp.items()},
                            rcwa_orders={str(k): v for k, v in rp.items()},
                            bpm_eta1=eb, rcwa_eta1=er, abs_deviation=abs(eb - er))
                    except Exception as e:
                        rec.update(error=str(e))
                    cases.append(rec)
                    if i % 20 == 0 or i == n_cases:
                        print(f"  {i}/{n_cases} done", flush=True)

    valid = [c for c in cases if c.get("abs_deviation") is not None]

    def _median(v):
        v = sorted(v)
        m = len(v) // 2
        return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])

    def summarize(subset):
        d = [c["abs_deviation"] for c in subset]
        if not d:
            return dict(n=0)
        return dict(n=len(d), mean=sum(d) / len(d), median=_median(d),
                    max=max(d), min=min(d),
                    median_bpm_eta1=_median([c["bpm_eta1"] for c in subset]),
                    median_rcwa_eta1=_median([c["rcwa_eta1"] for c in subset]))

    by_incidence = {
        inc: dict(
            overall=summarize([c for c in valid if c["incidence"] == inc]),
            by_slant={f"{s:g}": summarize([c for c in valid
                                           if c["incidence"] == inc
                                           and c["slant_deg"] == s])
                      for s in SLANTS_DEG},
            unslanted=summarize([c for c in valid if c["incidence"] == inc
                                 and c["slant_deg"] == 0.0]),
            slanted=summarize([c for c in valid if c["incidence"] == inc
                               and c["slant_deg"] > 0.0]),
        )
        for inc in INCIDENCES
    }

    out = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        anchor_passed=anchored,
        K_values=K_VALUES, dn_values=DN_VALUES, slants_deg=SLANTS_DEG,
        incidences=INCIDENCES, orders=list(ORDERS),
        n_x=N_X, dx=DX, n_z=N_Z, window_um=WINDOW_UM,
        thickness_um=THICKNESS_UM, wavelength_um=WAVELENGTH_UM, n0=N0,
        n_cases=n_cases, n_valid=len(valid),
        by_incidence=by_incidence,
        max_abs_deviation=max((c["abs_deviation"] for c in valid), default=None),
        cases=cases,
        quantity="first-order fraction P(+1)/[P(-1)+P(0)+P(+1)], computed the "
                 "same way from both engines so that the Fresnel losses RCWA "
                 "models and SlabBPM does not cannot enter the comparison",
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.normpath(OUT_PATH)}")

    for inc in INCIDENCES:
        rec = by_incidence[inc]
        tag = "   (the design loop's geometry)" if inc == "normal" else ""
        print(f"\n  incidence = {inc}{tag}")
        print("  slant   median |BPM - RCWA|    max      median BPM   median RCWA")
        for s in SLANTS_DEG:
            a = rec["by_slant"][f"{s:g}"]
            print(f"  {s:5.1f}      {a.get('median', float('nan')):.4f}"
                  f"            {a.get('max', float('nan')):.4f}"
                  f"      {a.get('median_bpm_eta1', float('nan')):.4f}"
                  f"       {a.get('median_rcwa_eta1', float('nan')):.4f}")
        u, sl = rec["unslanted"], rec["slanted"]
        print(f"  unslanted: median {u.get('median', float('nan')):.4f}, "
              f"max {u.get('max', float('nan')):.4f}   |   "
              f"slanted: median {sl.get('median', float('nan')):.4f}, "
              f"max {sl.get('max', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
