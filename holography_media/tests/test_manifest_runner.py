"""Phase 1.1/1.2 manifest runner tests.

Run at tiny CPU scale (n_x=256, n_iters=5) against a temp results dir --
never touches the real results/ tree.
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

torch.set_default_dtype(torch.float64)

from manifest import build_E1_jobs, build_E3_jobs, config_hash
import run_manifest as rm


def test_config_hash_is_method_independent_but_path_is_not():
    """Regression test for a real bug caught during Phase 1 development:
    config_hash is computed from `config` alone (method_id is NOT part of
    the config), so two jobs differing only in method_id share a
    config_hash by design (that's fine -- it lets Phase 4 group same-config
    rows across methods). But result_path MUST still disambiguate by
    method_id, or a second method's job silently collides with and is
    skipped as "already done" once the first method's file exists."""
    jobs = build_E3_jobs(n_x=64, n_iters=3, seeds=[0])
    m2 = next(j for j in jobs if j["method_id"] == "M2")
    m4 = next(j for j in jobs if j["method_id"] == "M4" and j["config"] == m2["config"])
    assert m2["config_hash"] == m4["config_hash"], \
        "same config should hash the same regardless of method"
    p2 = rm.result_path(m2["experiment_id"], m2["method_id"], m2["config_hash"], m2["seed"])
    p4 = rm.result_path(m4["experiment_id"], m4["method_id"], m4["config_hash"], m4["seed"])
    assert p2 != p4, "different methods with the same config must NOT collide on file path"
    print("config_hash/result_path collision guard OK:", p2, "!=", p4)


def test_manifest_end_to_end_and_resume():
    tmp = tempfile.mkdtemp(prefix="manifest_test_")
    try:
        rm.set_results_root(tmp)
        # match run_manifest's own internal job construction exactly (it
        # does not currently accept a seeds override, only n_x/n_iters/
        # converge_tol) so this test's expectation can't silently drift
        # from what the runner actually builds.
        jobs = build_E3_jobs(n_x=64, n_iters=3, converge_tol=1e-4)
        assert len(jobs) == 2 * 4 * 5  # 2 methods x 4 shrinkage values x 5 default seeds

        rm.run_manifest("E3", max_minutes=None, n_x=64, n_iters=3, converge_tol=1e-4)

        n_files = sum(len(files) for _, _, files in os.walk(tmp))
        assert n_files == len(jobs), f"expected {len(jobs)} result files, found {n_files}"

        # resume: rerun should do nothing (all already done)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rm.run_manifest("E3", max_minutes=None, n_x=64, n_iters=3, converge_tol=1e-4)
        assert f"{len(jobs)} already done" in buf.getvalue(), buf.getvalue()

        print(f"manifest end-to-end OK: {n_files} result files, resume is idempotent")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_schema_fields_present():
    tmp = tempfile.mkdtemp(prefix="manifest_test_schema_")
    try:
        rm.set_results_root(tmp)
        device = rm.get_device()
        commit = rm.git_commit_hash()
        jobs = build_E1_jobs(n_x=128, n_iters=3, seeds=[0])
        iterative = next(j for j in jobs if j["method_id"] == "M4")
        closed_form = next(j for j in jobs if j["method_id"] == "M3")

        required = {"git_commit", "experiment_id", "method_id", "seed", "config",
                   "config_hash", "device", "loss_curve", "iterations_run",
                   "early_stop_reason", "wall_s", "peak_mem_mb", "psnr",
                   "diffraction_efficiency", "contrast"}
        for job in (iterative, closed_form):
            result = rm.run_job(job, device, commit)
            missing = required - set(result.keys())
            assert not missing, f"{job['method_id']}: missing schema fields {missing}"
        print("schema fields present for both iterative and closed-form methods OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_config_hash_is_method_independent_but_path_is_not()
    test_manifest_end_to_end_and_resume()
    test_schema_fields_present()
    print("PASSED")
