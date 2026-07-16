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
    main()
