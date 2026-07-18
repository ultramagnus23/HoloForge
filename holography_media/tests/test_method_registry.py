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


if __name__ == "__main__":
    test_contrast_project_hits_cap()
    test_contrast_project_none_is_dose_project_only()
    test_linear_precomp_reduces_to_target_when_H_is_one()
    test_linear_precomp_satisfies_constraints_exactly()
    test_history_last_iteration_is_accurate_stop_point()
    print("PASSED")
