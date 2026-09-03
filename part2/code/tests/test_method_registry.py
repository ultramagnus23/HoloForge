"""Phase 2 method-registry tests: contrast_project (Phase 1.4 constraint
mechanism) and M3 (linear_precomp), the previously-missing baseline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import torch
from holomedia import NPDDRecorder, MediumParams, SlabBPM, linear_precomp, media_in_the_loop
from holomedia.optimize import contrast_project

torch.set_default_dtype(torch.float64)


def test_contrast_project_hits_cap():
    torch.manual_seed(0)
    E = torch.rand(256) * 0.2
    E[10] = 5.0
    E[50] = 8.0
    for budget in [1.0, 2.5]:
        for cap in [2.0, 4.0, 8.0]:
            Ep = contrast_project(E, budget, cap)
            achieved = (Ep.max() / Ep.mean()).item()
            assert abs(achieved - cap) < 1e-4, f"budget={budget} cap={cap}: achieved {achieved}"
            assert abs(Ep.mean().item() - budget) < 1e-8
    print("contrast_project OK: hits cap to <1e-4, mean exact")


def test_contrast_project_none_is_dose_project_only():
    torch.manual_seed(1)
    E = torch.rand(128) + 0.01
    from holomedia.optimize import dose_project
    a = contrast_project(E, 1.0, None)
    b = dose_project(E, 1.0)
    assert torch.equal(a, b)
    print("contrast_project(cap=None) == dose_project OK")


def _setup(n_x=128, sigma=0.08, D0=0.1):
    dx = 51.2 / n_x
    p = MediumParams(sigma=sigma, D0=D0)
    rec = NPDDRecorder(n_x, dx, t_total=8, n_steps=30, params=p)
    bpm = SlabBPM(n_x, dx, 0.405, p.thickness, n_z=8, n0=p.n0)
    return rec, bpm


def test_linear_precomp_reduces_to_target_when_H_is_one():
    # sigma=0, D0=0 => Ghat(K)=1, D0*K^2/F0=0 => H(K)=1 for all K => boost=1
    rec, bpm = _setup(sigma=0.0, D0=0.0)
    x = torch.arange(rec.n_x)
    target = ((x // 16) % 2).double() + 0.1  # >0 everywhere, avoids clip-to-0 confound

    dose_budget = 1.0
    E, recon = linear_precomp(target, rec, bpm, dose_budget=dose_budget, contrast_cap=None)
    expected = target * (dose_budget / target.mean())  # pure dose rescaling, no reshaping
    max_diff = (E - expected).abs().max().item()
    assert max_diff < 1e-8, f"H==1 should leave target unmodified (up to dose rescale): diff={max_diff}"
    print("linear_precomp H==1 reduces to target OK, max diff =", max_diff)


def test_linear_precomp_satisfies_constraints_exactly():
    rec, bpm = _setup(sigma=0.08, D0=0.1)  # realistic H(K), nontrivial boost
    x = torch.arange(rec.n_x)
    target = ((x // 8) % 2).double()

    for cap in [None, 2.0, 4.0]:
        E, recon = linear_precomp(target, rec, bpm, dose_budget=1.0, contrast_cap=cap)
        assert (E >= -1e-9).all(), f"cap={cap}: E has negative entries"
        assert abs(E.mean().item() - 1.0) < 1e-8, f"cap={cap}: mean(E) != dose_budget"
        if cap is not None:
            achieved = (E.max() / E.mean()).item()
            assert achieved <= cap + 1e-4, f"cap={cap}: achieved contrast {achieved} exceeds cap"
        assert torch.isfinite(recon).all()
    print("linear_precomp constraint satisfaction OK (E>=0, mean exact, contrast<=cap)")


def test_history_last_iteration_is_accurate_stop_point():
    """Regression test: history[-1][0] must equal the TRUE last iteration
    run, whether or not converge_tol triggered early. Before this fix,
    history only ever recorded log_every-aligned checkpoints, so a full
    n_iters run's last entry was almost never at n_iters-1 -- making it
    indistinguishable from a genuine early stop downstream
    (experiments/methods.py's early_stop_reason inference)."""
    rec, bpm = _setup(n_x=64, sigma=0.08, D0=0.1)
    x = torch.arange(rec.n_x)
    target = ((x // 8) % 2).double()
    n_iters, log_every = 60, 5

    # no early stop possible -> last entry must be exactly n_iters-1
    _, _, hist_full = media_in_the_loop(target, rec, bpm, n_iters=n_iters, seed=0,
                                        log_every=log_every, verbose=False,
                                        converge_tol=None)
    assert hist_full[-1][0] == n_iters - 1, \
        f"full run should end at iter {n_iters-1}, got {hist_full[-1][0]}"

    # deliberately loose tol -> must stop well before n_iters-1
    _, _, hist_early = media_in_the_loop(target, rec, bpm, n_iters=n_iters, seed=0,
                                         log_every=log_every, verbose=False,
                                         converge_tol=0.5, patience=3)
    assert hist_early[-1][0] < n_iters - 1, \
        f"loose-tol run should stop early, got last iter {hist_early[-1][0]}"
    print(f"history accuracy OK: full run ends at {hist_full[-1][0]}, "
          f"early-stop run ends at {hist_early[-1][0]}")


def test_oracle_respects_medium_saturation():
    """Regression test for the unphysical-oracle bug: oracle_ideal built
    its index profile as dn_max*(E - E.mean()) with no saturation clamp,
    so a contrast_cap of B produced index modulation up to (B-1)*dn_max --
    measured at exactly 7.00x dn_max for B=8. dn_max IS the medium's
    saturation index, so that 'oracle' was an unachievable bound and every
    headroom-to-oracle number was measured against it. Both oracles must
    now stay within the physically reachable index range."""
    from holomedia import oracle_ideal, oracle_unconstrained
    n_x = 128
    med = MediumParams()
    rec = NPDDRecorder(n_x, 0.05, params=med)
    bpm = SlabBPM(n_x, 0.05, 0.405, med.thickness, n_z=8, n0=med.n0)
    target = torch.zeros(n_x); target[40:60] = 1.0

    for cap in (2.0, 4.0, 8.0):
        E, _ = oracle_ideal(target, rec, bpm, n_iters=20, dose_budget=1.0,
                            seed=0, contrast_cap=cap)
        dn_lin = med.dn_max * (E - E.mean())
        dn = med.dn_max * torch.tanh(dn_lin / med.dn_max)
        ratio = float(dn.abs().max()) / med.dn_max
        assert ratio <= 1.0 + 1e-6, (
            f"ORC at cap={cap}x reached {ratio:.2f}x dn_max -- oracle is "
            f"recording beyond the medium's saturation index")

    dn_u, _ = oracle_unconstrained(target, rec, bpm, n_iters=20, seed=0)
    assert float(dn_u.abs().max()) / med.dn_max <= 1.0 + 1e-6
    print("oracle saturation OK: both oracles within 1.0x dn_max (ORC was 7.00x)")


def test_unconstrained_oracle_is_at_least_the_constrained_one():
    """Ordering invariant: M5b (oracle_unconstrained) optimizes over a
    strictly LOOSER feasible set than M5a (oracle_ideal) -- same dn_max
    saturation bound, but no E>=0 / dose / contrast constraints -- so it
    can never score worse. It did, by 3.8 dB (6.76 vs 10.55), because it
    optimized a raw index-valued variable of scale dn_max=3.5e-3 while
    inheriting the lr=5e-2 shared by methods whose variables are O(1):
    every Adam step moved it ~14x dn_max, saturating tanh instead of
    descending. Now reparameterized to a dimensionless u with
    dn = dn_max*tanh(u). A violation here means the M5a/M5b gap
    decomposition is invalid again."""
    from holomedia import oracle_ideal, oracle_unconstrained, psnr_si
    n_x = 128
    med = MediumParams()
    rec = NPDDRecorder(n_x, 0.05, params=med)
    bpm = SlabBPM(n_x, 0.05, 0.405, med.thickness, n_z=8, n0=med.n0)
    target = torch.zeros(n_x); target[40:60] = 1.0

    _, recon_c = oracle_ideal(target, rec, bpm, n_iters=300, lr=5e-2,
                              dose_budget=1.0, seed=0, contrast_cap=8.0)
    _, recon_u = oracle_unconstrained(target, rec, bpm, n_iters=300, lr=5e-2, seed=0)
    p_c, p_u = psnr_si(recon_c, target), psnr_si(recon_u, target)
    assert p_u >= p_c - 0.5, (
        f"unconstrained oracle ({p_u:.2f} dB) scored below the constrained one "
        f"({p_c:.2f} dB) -- impossible for a looser feasible set; the M5a/M5b "
        f"decomposition is invalid")
    print(f"oracle ordering OK: ORU {p_u:.2f} dB >= ORC {p_c:.2f} dB")


def test_loss_and_metric_are_the_same_objective():
    """Regression test for the objective/metric mismatch: optimizers
    minimized a SUM-normalized MSE while the reported PSNR was
    MAX-normalized, so 'optimized better' and 'scored better' could
    disagree. psnr_si must be exactly the monotone transform of the
    si_mse the optimizers now minimize, and must be invariant to
    rescaling the reconstruction."""
    from holomedia import si_mse, psnr_si
    torch.manual_seed(0)
    b = torch.rand(256).abs() + 0.1
    a = (b + 0.1 * torch.rand(256)).abs()

    # psnr_si == -10 log10(si_mse), exactly
    assert abs(psnr_si(a, b) - float(-10.0 * torch.log10(si_mse(a, b) + 1e-12))) < 1e-9

    # scale invariance: the whole point -- brightness is a readout gain,
    # not a property of the design, so it must not change the score.
    for s in (0.01, 0.5, 3.0, 100.0):
        assert abs(psnr_si(a * s, b) - psnr_si(a, b)) < 1e-4, \
            f"psnr_si changed under rescaling by {s}"

    # better reconstruction => strictly better score (monotone, sane)
    worse = (b + 0.5 * torch.rand(256)).abs()
    assert psnr_si(a, b) > psnr_si(worse, b)
    print("objective/metric alignment OK: psnr_si == -10log10(si_mse), scale-invariant")




# ============================================================ SAT surrogate
def test_saturation_only_twin_matches_npdd_in_zero_transport_limit():
    """SaturationOnlyTwin is not an arbitrary sigmoid: it is the exact
    closed-form solution of the NPDD system once every transport term is
    switched off. Pin that, because it is the entire justification for
    calling SAT-vs-MIL an ablation of the transport physics rather than a
    comparison against some other model that happens to saturate.

    n_steps is set high on the reference so the residual is the IMEX
    integrator's own time-discretization error, not the surrogate's.
    """
    from holomedia import SaturationOnlyTwin
    p = MediumParams(D0=0.0, sigma=0.0, k_bleach=0.0)
    full = NPDDRecorder(256, 0.05, t_total=10.0, n_steps=4000, params=p)
    sat = SaturationOnlyTwin(256, 0.05, t_total=10.0, params=p)
    torch.manual_seed(0)
    E = torch.rand(256) * 2.0
    a, b = full(E), sat(E)
    rel = float((a - b).abs().max() / b.abs().max())
    assert rel < 1e-3, f"zero-transport limit mismatch: {rel}"
    print(f"SaturationOnlyTwin == NPDD in zero-transport limit OK ({rel:.2e} rel)")


def test_uncalibrated_surrogate_is_saturated_at_working_dose():
    """The documented reason SAT calibrates before optimizing.

    At the default medium, a = kappa * t_total = 20, so the exact
    zero-transport map has already run to completion at mean dose 1 and
    its slope there is ~0. This test pins the diagnosis so nobody
    "simplifies" the calibration away and quietly turns the baseline into
    a straw man that loses on optimizer conditioning rather than on
    modeling power.
    """
    from holomedia import SaturationOnlyTwin
    from holomedia.npdd import fit_saturation_only
    p = MediumParams()
    rec = NPDDRecorder(256, 0.2, params=p)
    uncal = SaturationOnlyTwin(256, 0.2, params=p)          # a = kappa*t = 20
    a_eff, nrmse = fit_saturation_only(rec, n_samples=16)
    cal = SaturationOnlyTwin(256, 0.2, params=p, a_eff=a_eff)

    def slope(model):
        E = torch.ones(256, requires_grad=True)
        model(E).sum().backward()
        return float(E.grad.abs().mean())

    assert slope(cal) > 1000 * slope(uncal), (
        f"calibration should restore usable gradient: "
        f"{slope(cal):.3e} vs {slope(uncal):.3e}")
    assert a_eff < p.kappa * rec.t_total, "fit should reduce the sensitivity"
    assert nrmse < 0.25, f"pointwise fit unexpectedly poor: {nrmse}"
    print(f"SAT calibration OK: a_eff={a_eff:.2f} (vs 20 uncalibrated), "
          f"NRMSE={nrmse:.3f}, slope x{slope(cal)/slope(uncal):.1e}")


def test_sat_gain_falls_between_bsgd_and_mil():
    """SAT has strictly less modeling power than MIL (a pointwise
    saturating map vs. the full transport PDE) and strictly more than
    BSGD (which models no medium at all), so its score on the real twin
    should land between them. If it does not, the harness is wrong --
    e.g. SAT accidentally optimizing or being evaluated against the wrong
    twin -- which is exactly the failure this guards.

    Compared with a tolerance rather than exactly: all three are Adam
    runs on a non-convex objective, so a few hundredths of a dB of
    crossover is optimizer noise, not a broken ordering.
    """
    from experiments.methods import run_method
    n_x, dx = 256, 0.2
    p = MediumParams()
    rec = NPDDRecorder(n_x, dx, params=p)
    bpm = SlabBPM(n_x, dx, 0.405, p.thickness, n_z=32, n0=p.n0,
                  dtype=torch.complex128)
    x = torch.arange(n_x)
    target = ((x // 8) % 2).double()

    scores = {m: run_method(m, target, rec, bpm, seed=0, n_iters=300,
                            contrast_cap=2.0)["psnr"]
              for m in ("BSGD", "SAT", "MIL")}
    tol = 0.05
    assert scores["SAT"] >= scores["BSGD"] - tol, scores
    assert scores["SAT"] <= scores["MIL"] + tol, scores
    print("SAT between BSGD and MIL OK: "
          + ", ".join(f"{k}={v:.3f}dB" for k, v in scores.items()))


def test_sat_is_evaluated_on_the_real_twin_not_the_surrogate():
    """SAT designs against the cheap model but must be SCORED on the real
    one -- that asymmetry is the whole experiment. Verified structurally:
    re-running the returned exposure through the real recorder must
    reproduce the returned reconstruction exactly, and the surrogate's
    own reconstruction must differ from it (otherwise the surrogate is
    silently standing in for the twin at evaluation time and the
    comparison is meaningless).
    """
    from holomedia import sat_sgd, SaturationOnlyTwin
    from holomedia.optimize import sat_fit_cache_clear
    sat_fit_cache_clear()
    n_x, dx = 128, 0.4
    p = MediumParams()
    rec = NPDDRecorder(n_x, dx, params=p)
    bpm = SlabBPM(n_x, dx, 0.405, p.thickness, n_z=16, n0=p.n0,
                  dtype=torch.complex128)
    x = torch.arange(n_x)
    target = ((x // 8) % 2).double()
    E, recon, _ = sat_sgd(target, rec, bpm, n_iters=30, contrast_cap=2.0,
                          fit_samples=8)

    with torch.no_grad():
        real = bpm(rec(E), shrinkage=rec.p.shrinkage)
        surr_model = SaturationOnlyTwin(n_x, dx, params=p,
                                        a_eff=sat_sgd.last_a_eff)
        surr = bpm(surr_model(E), shrinkage=p.shrinkage)
    assert torch.allclose(recon, real), "SAT must report the REAL twin's readout"
    assert not torch.allclose(recon, surr, atol=1e-9), \
        "surrogate and real twin readouts are identical -- evaluation is not real"
    print("SAT evaluated on the real twin OK")


if __name__ == "__main__":
    test_contrast_project_hits_cap()
    test_contrast_project_none_is_dose_project_only()
    test_linear_precomp_reduces_to_target_when_H_is_one()
    test_linear_precomp_satisfies_constraints_exactly()
    test_history_last_iteration_is_accurate_stop_point()
    test_oracle_respects_medium_saturation()
    test_unconstrained_oracle_is_at_least_the_constrained_one()
    test_loss_and_metric_are_the_same_objective()
    test_saturation_only_twin_matches_npdd_in_zero_transport_limit()
    test_uncalibrated_surrogate_is_saturated_at_working_dose()
    test_sat_gain_falls_between_bsgd_and_mil()
    test_sat_is_evaluated_on_the_real_twin_not_the_surrogate()
    print("PASSED")
