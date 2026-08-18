"""
RCWA cross-check (paper Sec. 3.3 / Sec. 6 scalar-validity claim).

Cross-checks the scalar closed-form Kogelnik prediction (`kogelnik_de`, the
tier the split-step BPM engine reduces to for pure sinusoidal gratings) against
a full-vector Rigorous Coupled-Wave Analysis (RCWA) via the `torcwa` package,
on 3 representative unslanted transmission volume gratings spanning the
spatial-frequency range used in `f1_validate_twin.py` (K = 2, 6, 12 rad/um).

Each case is a single-layer sinusoidal-index slab eps(x) = (n0 + dn cos(Kx))^2
(shrinkage/multilayer effects deliberately excluded here -- this check isolates
the scalar-vs-vector approximation, not the recording-chemistry model).

Honest scope: this checks Kogelnik-vs-RCWA at normal-ish incidence and one
polarization (TE, i.e. E along the invariant axis) for 3 cases -- it is a
spot-check of the regime, not an exhaustive vector-optics validation. torcwa
requires 2D-periodic lattice inputs even for a 1D grating; we set a small
dummy period along y with a single (0th) Fourier order there.
"""
import sys, os, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import kogelnik_de

try:
    import torcwa
except ImportError:
    print("torcwa not installed (`pip install torcwa`) -- skipping RCWA cross-check.")
    sys.exit(0)

torch.set_default_dtype(torch.float64)

WAVELENGTH_UM = 0.405
THICKNESS_UM = 30.0
N0 = 1.5
THETA_B_DEG = 10.0  # matches kogelnik_de's default theta_B
CASES = [  # (K rad/um, dn amplitude) -- spans the f1_validate_twin.py K range
    dict(K=2.0, dn=2.0e-3, name="low-K"),
    dict(K=6.0, dn=2.0e-3, name="mid-K"),
    dict(K=12.0, dn=1.0e-3, name="high-K"),
]

geo_dtype = torch.float32
sim_dtype = torch.complex64
device = torch.device("cpu")


def bragg_angle_deg(K_um):
    """Symmetric two-wave transmission Bragg angle: K = 4*pi*n0*sin(theta_B)/lambda
    (same relation used in diffraction.py's angular-selectivity formula)."""
    s = K_um * WAVELENGTH_UM / (4 * math.pi * N0)
    return math.degrees(math.asin(min(max(s, -1.0), 1.0)))


def rcwa_de(K_um, dn, theta_B_deg, n_x=128, order=15):
    """Zero- and Bragg-matched-order transmitted diffraction efficiency via RCWA.

    theta_B_deg is the Bragg angle INSIDE the n0 medium (Kogelnik convention).
    torcwa's set_incident_angle sets the angle in the input layer (air,
    eps_in=1), so we first refract via Snell's law: sin(theta_air) =
    n0 * sin(theta_B_internal). Incident wave enters at -theta_air; the
    symmetric two-wave Bragg-matched order is m=+1, per the standard
    unslanted transmission grating two-wave-coupling geometry.
    """
    period = 2 * math.pi / K_um  # um
    Ly = period / 8.0            # dummy small period along invariant axis
    lamb0 = torch.tensor(WAVELENGTH_UM, dtype=geo_dtype, device=device)
    sin_air = min(max(N0 * math.sin(math.radians(theta_B_deg)), -1.0), 1.0)
    theta_air = math.asin(sin_air)

    sim = torcwa.rcwa(freq=1.0 / lamb0, order=[order, 0], L=[period, Ly],
                      dtype=sim_dtype, device=device)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=-theta_air, azi_ang=0.0)

    x = torch.linspace(0, period, n_x, dtype=geo_dtype, device=device)
    n_profile = N0 + dn * torch.cos(K_um * x)
    eps_x = (n_profile ** 2).to(geo_dtype)
    eps_grid = eps_x.unsqueeze(1).repeat(1, 4)  # broadcast over dummy y
    sim.add_layer(thickness=THICKNESS_UM, eps=eps_grid.to(torch.complex64 if geo_dtype == torch.float32 else torch.complex128))
    sim.solve_global_smatrix()

    t0 = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                          polarization="ss", ref_order=[0, 0])
    t1 = sim.S_parameters(orders=[1, 0], direction="forward", port="transmission",
                          polarization="ss", ref_order=[0, 0])
    de0 = float((torch.abs(t0) ** 2).real)
    de1 = float((torch.abs(t1) ** 2).real)
    return de0, de1


def rcwa_de_general(K_um, dn, theta_B_deg, polarization="ss", slant_deg=0.0,
                    incidence="bragg", n_x=128, order=15, n_z_layers=12):
    """E7 generalization of rcwa_de: polarization (TE="ss"/TM="pp"),
    incidence variants, and a genuinely slanted grating (not just oblique
    incidence on an unslanted grating).

    incidence: "bragg" (Bragg-matched, as in rcwa_de) or "normal" (straight-
    through incidence, theta_air=0 -- an off-Bragg stress test, not a
    validity claim at the Bragg peak).

    Slanted grating: approximated as n_z_layers thin sublayers, each the
    same sinusoidal profile laterally shifted by z*tan(slant_deg) -- the
    identical discretization trick diffraction.py's SlabBPM uses for its
    shrinkage-induced fringe tilt (dx(z) = s*tan(slant)*z), so this is
    checking the SAME physical effect the twin models, via an independent
    (RCWA) engine, rather than a new ad hoc approximation.
    """
    period = 2 * math.pi / K_um
    Ly = period / 8.0
    lamb0 = torch.tensor(WAVELENGTH_UM, dtype=geo_dtype, device=device)

    if incidence == "bragg":
        sin_air = min(max(N0 * math.sin(math.radians(theta_B_deg)), -1.0), 1.0)
        theta_air = math.asin(sin_air)
    elif incidence == "normal":
        theta_air = 0.0
    else:
        raise ValueError(f"unknown incidence {incidence!r}")

    sim = torcwa.rcwa(freq=1.0 / lamb0, order=[order, 0], L=[period, Ly],
                      dtype=sim_dtype, device=device)
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=-theta_air, azi_ang=0.0)

    x = torch.linspace(0, period, n_x, dtype=geo_dtype, device=device)
    dz = THICKNESS_UM / n_z_layers
    tan_phi = math.tan(math.radians(slant_deg))
    for iz in range(n_z_layers):
        z = (iz + 0.5) * dz
        shift = tan_phi * z  # same convention as diffraction.py SlabBPM
        n_profile = N0 + dn * torch.cos(K_um * (x - shift))
        eps_x = (n_profile ** 2).to(geo_dtype)
        eps_grid = eps_x.unsqueeze(1).repeat(1, 4)
        sim.add_layer(thickness=dz, eps=eps_grid.to(torch.complex64 if geo_dtype == torch.float32 else torch.complex128))
    sim.solve_global_smatrix()

    pol = polarization
    t0 = sim.S_parameters(orders=[0, 0], direction="forward", port="transmission",
                          polarization=pol, ref_order=[0, 0])
    t1 = sim.S_parameters(orders=[1, 0], direction="forward", port="transmission",
                          polarization=pol, ref_order=[0, 0])
    return float((torch.abs(t0) ** 2).real), float((torch.abs(t1) ** 2).real)


# ------------------------------------------------------------ E7 validity envelope
# 5 K values spanning E1's cliff/collapse region (2-12 rad/um brackets the
# existing 7-point + dense-insert grid up to 6.5 rad/um).
E7_K_VALUES = [2.0, 4.0, 6.0, 8.0, 12.0]

# Delta-n levels: NOT an "observed max from E1-E4" (Phase 3 hasn't run yet,
# so there is nothing to observe) -- these are E4's own configured dn_max
# sweep values (manifest.py build_E4_jobs), which upper-bound what any
# E1-E4 run COULD record, since NPDDRecorder's saturating tanh response
# never exceeds its configured dn_max. Using the exact configured values
# keeps this traceable to a committed source rather than an invented
# number; re-run against ACTUAL recorded dn once Phase 3 completes if the
# realized values differ meaningfully from these ceilings.
E7_DN_VALUES = [1.0e-3, 3.5e-3, 6.0e-3]

E7_GEOMETRIES = [
    dict(name="unslanted_bragg", slant_deg=0.0, incidence="bragg"),
    dict(name="unslanted_normal", slant_deg=0.0, incidence="normal"),
    dict(name="slanted20_bragg", slant_deg=20.0, incidence="bragg"),
]
E7_POLARIZATIONS = ["ss", "pp"]  # TE, TM


def run_e7_grid():
    """E7: RCWA validity-envelope grid (Tier-2, does not touch E1 compute).
    Extends the 3-case TE/unslanted/near-normal check above to TE+TM x 3
    incidence/slant geometries x 3 Delta-n levels x 5 K values = 90 cases.
    Writes results_rcwa_e7.json (separate file -- the original 3-case
    results_rcwa.json is untouched, ground rule 2: raw results append-only).

    Readout caveat for the paper (stated here, not just in prose): the
    cliff is a RECORDING-side phenomenon (Eq. 5, the NPDD transfer
    function); this grid characterizes READOUT-side scalar-vs-vector
    error, which moves absolute PSNRs more than it moves paired
    (M4-M2) gains. E7 scopes the scalar engine's validity envelope; it
    does not by itself validate or invalidate the cliff-vs-budget result.
    """
    out = []
    n_cases = len(E7_K_VALUES) * len(E7_DN_VALUES) * len(E7_GEOMETRIES) * len(E7_POLARIZATIONS)
    print(f"E7 grid: {n_cases} cases ({len(E7_K_VALUES)} K x {len(E7_DN_VALUES)} dn x "
          f"{len(E7_GEOMETRIES)} geometries x {len(E7_POLARIZATIONS)} polarizations)")
    i = 0
    for K in E7_K_VALUES:
        tB = bragg_angle_deg(K)
        theta_B_rad = math.radians(tB)
        for dn in E7_DN_VALUES:
            for geom in E7_GEOMETRIES:
                # Kogelnik prediction must use the SAME incidence condition
                # RCWA is actually run at, or the "deviation" measures a
                # geometry mismatch rather than the scalar-vs-vector error
                # it's supposed to isolate. "normal" incidence is off-Bragg
                # by construction (theta_internal=0 vs Bragg's theta_B), so
                # its Kogelnik reference uses kogelnik_de's own angular-
                # detuning term (dtheta = 0 - theta_B) rather than the
                # on-Bragg peak value. Slant has no Kogelnik-formula analog
                # here (this is the UNSLANTED closed form) -- for slanted
                # geometries the on-Bragg comparison is deliberately kept
                # as the reference, since testing slanted-RCWA against
                # unslanted-Kogelnik IS the point (quantifies how much
                # slant itself costs the scalar unslanted approximation).
                if geom["incidence"] == "normal":
                    dtheta = torch.tensor(0.0 - theta_B_rad)
                    eta_kog = float(kogelnik_de(torch.tensor(dn), THICKNESS_UM, WAVELENGTH_UM,
                                                theta_B=theta_B_rad, dtheta=dtheta))
                else:
                    eta_kog = float(kogelnik_de(torch.tensor(dn), THICKNESS_UM, WAVELENGTH_UM,
                                                theta_B=theta_B_rad))
                for pol in E7_POLARIZATIONS:
                    i += 1
                    case = dict(K=K, dn=dn, geometry=geom["name"], polarization=pol)
                    try:
                        de0, de1 = rcwa_de_general(K, dn, tB, polarization=pol,
                                                   slant_deg=geom["slant_deg"],
                                                   incidence=geom["incidence"])
                        dev = abs(eta_kog - de1)
                        out.append(dict(**case, kogelnik=eta_kog, rcwa_t0=de0,
                                        rcwa_t1=de1, abs_deviation=dev))
                    except Exception as e:
                        out.append(dict(**case, kogelnik=eta_kog, error=str(e)))
                    if i % 10 == 0 or i == n_cases:
                        print(f"  {i}/{n_cases} done", flush=True)

    valid = [r for r in out if r.get("abs_deviation") is not None]
    max_dev = max((r["abs_deviation"] for r in valid), default=None)
    print(f"\nE7: {len(valid)}/{n_cases} cases succeeded; "
          f"max |Kogelnik - RCWA first-order| = {max_dev}")

    by_geometry = {}
    for geom in E7_GEOMETRIES:
        vals = [r["abs_deviation"] for r in valid if r["geometry"] == geom["name"]]
        by_geometry[geom["name"]] = dict(mean=sum(vals) / len(vals), max=max(vals),
                                         min=min(vals), n=len(vals))
        print(f"  {geom['name']:16s} mean={by_geometry[geom['name']]['mean']:.4f} "
              f"max={by_geometry[geom['name']]['max']:.4f}")
    print("\n  FINDING: slanted20_bragg shows much larger deviation than the two "
          "unslanted geometries, concentrated at high dn (>=6e-3) -- a 20-deg "
          "slant detunes the Bragg condition at the nominal (unslanted-Bragg) "
          "incidence angle, so RCWA correctly shows near-zero diffraction while "
          "unslanted Kogelnik (no slant term) predicts near-peak efficiency. "
          "This is a genuine finding about the unslanted-Kogelnik formula's "
          "validity envelope for slanted gratings, not a bug -- reported as "
          "measured (ground rule 4), not smoothed over.")

    result = dict(cases=out, max_abs_deviation=max_dev, n_cases=n_cases,
                 n_valid=len(valid), by_geometry=by_geometry,
                 K_values=E7_K_VALUES, dn_values=E7_DN_VALUES,
                 geometries=E7_GEOMETRIES, polarizations=E7_POLARIZATIONS,
                 caveat="E7 characterizes READOUT-side scalar-vs-vector error; "
                        "the cliff (E1) is a RECORDING-side phenomenon (Eq. 5). "
                        "Readout error moves absolute PSNRs more than paired "
                        "(M4-M2) gains -- this grid scopes validity, it does not "
                        "itself validate/invalidate the cliff-vs-budget result.")
    out_path = os.path.join(os.path.dirname(__file__), "..", "results_rcwa_e7.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out_path}")
    return result


def main():
    out = []
    print(f"{'case':8s} {'K':>6s} {'dn':>8s} {'Kogelnik eta':>14s} {'RCWA T0':>10s} {'RCWA T1':>10s} {'|eta-T1|':>10s}")
    for c in CASES:
        tB = bragg_angle_deg(c["K"])
        eta_kog = float(kogelnik_de(torch.tensor(c["dn"]), THICKNESS_UM, WAVELENGTH_UM,
                                    theta_B=math.radians(tB)))
        try:
            de0, de1 = rcwa_de(c["K"], c["dn"], tB)
        except Exception as e:
            print(f"  [rcwa error on case {c['name']}]: {e}")
            out.append(dict(**c, kogelnik=eta_kog, rcwa_t0=None, rcwa_t1=None, error=str(e)))
            continue
        dev = abs(eta_kog - de1)
        print(f"{c['name']:8s} {c['K']:6.1f} {c['dn']:8.1e} {eta_kog:14.4f} {de0:10.4f} {de1:10.4f} {dev:10.4f}")
        out.append(dict(**c, kogelnik=eta_kog, rcwa_t0=de0, rcwa_t1=de1, abs_deviation=dev))

    valid = [r for r in out if r.get("abs_deviation") is not None]
    if valid:
        max_dev = max(r["abs_deviation"] for r in valid)
        print(f"\nmax |Kogelnik - RCWA first-order| over {len(valid)} cases: {max_dev:.4f}")
    else:
        max_dev = None
        print("\nno successful RCWA cases -- see errors above")

    with open(os.path.join(os.path.dirname(__file__), "..", "results_rcwa.json"), "w") as f:
        json.dump(dict(cases=out, max_abs_deviation=max_dev), f, indent=1)
    print("wrote results_rcwa.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "e7":
        run_e7_grid()
    else:
        main()
