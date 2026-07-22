"""Phase 4 analysis tests.

Unit tests for the pure-math helpers (mean_std_median_ci95, paired_gain,
the two K* estimators) against known synthetic values, plus an end-to-end
test that runs a HANDFUL of real (tiny CPU-scale) E1-shaped jobs -- not
the full 924-job manifest, which would be too slow for a test -- through
run_manifest.run_job and then through the full aggregate.py pipeline
(headroom_closure, e1_gain_curve) to verify it produces sane output on
real data, not just mocked structures.
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

torch.set_default_dtype(torch.float64)

import aggregate as agg
from manifest import _job, DEFAULT_MEDIUM, K_from_period
import run_manifest as rm


def test_mean_std_median_ci95():
    r = agg.mean_std_median_ci95([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r["n"] == 5
    assert abs(r["mean"] - 3.0) < 1e-9
    assert abs(r["median"] - 3.0) < 1e-9
    assert r["ci95_lo"] < 3.0 < r["ci95_hi"]

    single = agg.mean_std_median_ci95([7.0])
    assert single["mean"] == 7.0 and single["ci95_lo"] == single["ci95_hi"] == 7.0

    empty = agg.mean_std_median_ci95([])
    assert empty["n"] == 0 and empty["mean"] is None
    print("mean_std_median_ci95 OK")


def test_paired_gain():
    rows_a = [dict(seed=0, psnr=5.0), dict(seed=1, psnr=6.0), dict(seed=2, psnr=7.0)]
    rows_b = [dict(seed=0, psnr=3.0), dict(seed=1, psnr=3.0)]  # seed 2 missing
    pairs = agg.paired_gain(rows_a, rows_b, key="psnr")
    assert pairs == [(0, 2.0), (1, 3.0)], pairs  # seed 2 correctly dropped (unmatched)
    print("paired_gain OK:", pairs)


def test_zero_crossing_estimator():
    # clean crossing between K=3 (gain +1) and K=4 (gain -1) -> K*=3.5
    curve = [(1, 2.0), (2, 1.5), (3, 1.0), (4, -1.0), (5, -1.2)]
    kstar = agg.find_zero_crossing_K(curve)
    assert abs(kstar - 3.5) < 1e-9, kstar

    # no crossing (always positive)
    assert agg.find_zero_crossing_K([(1, 1.0), (2, 2.0)]) is None
    print("find_zero_crossing_K OK:", kstar)


def test_ci_includes_zero_estimator():
    # K=3's CI includes 0, and gain stays <=0.25 for K=3,4,5 -> K*=3
    curve = [(1, 2.0, 1.5, 2.5), (2, 1.0, 0.5, 1.5),
            (3, 0.1, -0.2, 0.4), (4, 0.05, -0.3, 0.2), (5, -0.1, -0.4, 0.1)]
    kstar = agg.find_ci_includes_zero_K(curve)
    assert kstar == 3, kstar

    # K=3's CI includes 0 but K=4 exceeds threshold again -> not K=3, no valid K
    curve2 = [(1, 2.0, 1.5, 2.5), (2, 0.1, -0.2, 0.4), (3, 2.0, 1.8, 2.2)]
    kstar2 = agg.find_ci_includes_zero_K(curve2)
    assert kstar2 is None, kstar2
    print("find_ci_includes_zero_K OK:", kstar)


def _make_e1_job(K, budget, method_id, seed, n_x=48, n_iters=4):
    dx = 51.2 / n_x
    period_px = max(2, round(2 * 3.141592653589793 / (K * dx)))
    config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                 converge_tol=None, contrast_cap=budget, dose_budget=1.0,
                 medium=DEFAULT_MEDIUM, target=dict(kind="bars", period_px=period_px),
                 K_nominal=K)
    return _job("E1", method_id, seed, config)


def test_end_to_end_headroom_closure_on_real_tiny_data():
    tmp = tempfile.mkdtemp(prefix="agg_test_")
    try:
        rm.set_results_root(tmp)
        device = rm.get_device()
        commit = rm.git_commit_hash()

        # 3 K values, 1 budget, M2+M4, 2 seeds -- small but real, exercises
        # the full pipeline (paired gains, both K* estimators, headroom closure)
        Ks = [2.0, 5.0, 9.0]
        budget = 4.0
        for K in Ks:
            for method_id in ["M2", "M4"]:
                for seed in [0, 1]:
                    job = _make_e1_job(K, budget, method_id, seed)
                    result = rm.run_job(job, device, commit)
                    path = rm.result_path(job["experiment_id"], job["method_id"],
                                          job["config_hash"], job["seed"])
                    rm.atomic_write_json(path, result)

        out = agg.build_paper_numbers(results_root=tmp)
        assert out["n_result_files"] == 3 * 2 * 2  # 3 K x 2 methods x 2 seeds
        assert "E1" in out["experiments_present"]

        closure = out["e1_headroom_closure"]
        row = next(r for r in closure if r["budget"] == budget)
        assert row.get("status") != "no_data", row
        assert row["n_K_points"] == 3
        assert row["measured_contrast_C"] is not None
        assert row["predicted_Kc_from_measured_C"] is not None
        # at least one K* estimator should return a number or None consistently
        # (not crash) -- with only 3 K points a real crossing may or may not exist
        print("headroom closure row:", {k: v for k, v in row.items() if k != "gain_curve"})

        sub_cliff = out["sub_cliff_non_monotonicity"]
        assert sub_cliff["status"] in ("blocked_pending_phase3", "answered_from_real_data")
        print("sub_cliff status:", sub_cliff["status"])
        print("end-to-end headroom closure on real tiny E1-shaped data OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_mean_std_median_ci95()
    test_paired_gain()
    test_zero_crossing_estimator()
    test_ci_includes_zero_estimator()
    test_end_to_end_headroom_closure_on_real_tiny_data()
    print("PASSED")
