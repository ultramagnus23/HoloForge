"""Phase 1.1/1.2 manifest runner tests.

Run at tiny CPU scale (n_x=256, n_iters=5) against a temp results dir --
never touches the real results/ tree.
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

torch.set_default_dtype(torch.float64)

from manifest import build_M1_jobs, build_S1_jobs, config_hash
import run_manifest as rm


def test_config_hash_is_method_independent_but_path_is_not():
    """Regression test for a real bug caught during Phase 1 development:
    config_hash is computed from `config` alone (method_id is NOT part of
    the config), so two jobs differing only in method_id share a
    config_hash by design (that's fine -- it lets Phase 4 group same-config
    rows across methods). But result_path MUST still disambiguate by
    method_id, or a second method's job silently collides with and is
    skipped as "already done" once the first method's file exists."""
    jobs = build_S1_jobs(n_x=64, n_iters=3, seeds=[0])
    m2 = next(j for j in jobs if j["method_id"] == "BSGD")
    m4 = next(j for j in jobs if j["method_id"] == "MIL" and j["config"] == m2["config"])
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
        jobs = build_S1_jobs(n_x=64, n_iters=3, converge_tol=1e-4)
        assert len(jobs) == 3 * 5 * 2 * 3  # 3 K points x 5 ablation conditions x 2 methods x 3 default seeds

        rm.run_manifest("S1", max_minutes=None, n_x=64, n_iters=3, converge_tol=1e-4)

        n_files = sum(len(files) for _, _, files in os.walk(tmp))
        assert n_files == len(jobs), f"expected {len(jobs)} result files, found {n_files}"

        # resume: rerun should do nothing (all already done)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rm.run_manifest("S1", max_minutes=None, n_x=64, n_iters=3, converge_tol=1e-4)
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
        jobs = build_M1_jobs(n_x=128, n_iters=3, seeds=[0])
        iterative = next(j for j in jobs if j["method_id"] == "MIL")
        closed_form = next(j for j in jobs if j["method_id"] == "LPC")

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


def test_run_manifest_returns_completion_status():
    """Regression test for the Colab notebook resume bug: run_manifest()
    used to return None in both the timed-out and fully-complete cases,
    so nothing calling it (e.g. the notebook's auto-resume loop) could
    tell whether to loop again. Now returns a status dict."""
    tmp = tempfile.mkdtemp(prefix="manifest_status_test_")
    try:
        rm.set_results_root(tmp)
        # S1 default job count: 3 K points x 5 ablation conditions x 2 methods x 3 default seeds = 90
        cut_short = rm.run_manifest("S1", max_minutes=0, n_x=32, n_iters=3)
        assert cut_short == dict(complete=False, n_run=0, n_done_already=0, n_done=0,
                                 n_total=90, n_remaining=90, last_attempted_job_id=None), cut_short

        full = rm.run_manifest("S1", max_minutes=None, n_x=32, n_iters=3)
        assert full["complete"] is True and full["n_run"] == 90 and full["n_remaining"] == 0
        assert full["n_done"] == 90

        resumed = rm.run_manifest("S1", max_minutes=None, n_x=32, n_iters=3)
        assert resumed["complete"] is True and resumed["n_run"] == 0
        assert resumed["n_done_already"] == 90 and resumed["n_done"] == 90
        print("run_manifest completion-status dict OK:", cut_short, full, resumed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_atomic_write_json_verifies_readable():
    """Spec Sec. 1.5: a job is marked done only after its output is
    verified readable, not just written."""
    tmp = tempfile.mkdtemp(prefix="write_test_")
    try:
        p = os.path.join(tmp, "a", "b.json")
        rm.atomic_write_json(p, dict(x=1, y="hello"))
        import json
        assert json.load(open(p)) == dict(x=1, y="hello")
        assert not os.path.exists(p + ".tmp")
        print("atomic_write_json write+verify OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clean_partial_files_removes_stray_tmp_only():
    """Spec Sec. 1.5: partial (.tmp) files from an interrupted write are
    deleted on startup; real result files are left alone."""
    tmp = tempfile.mkdtemp(prefix="cleanup_test_")
    try:
        real = os.path.join(tmp, "a", "real.json")
        rm.atomic_write_json(real, dict(ok=True))
        stray = os.path.join(tmp, "a", "stray.json.tmp")
        with open(stray, "w") as f:
            f.write("partial, interrupted mid-write")
        n = rm.clean_partial_files(tmp)
        assert n == 1
        assert not os.path.exists(stray)
        assert os.path.exists(real)
        print("clean_partial_files OK: removed stray .tmp, real file untouched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_deterministic_rerun_matches():
    """Spec Sec. 1.6 explicit requirement: run one job twice with the same
    seed, confirm bitwise-identical (or tolerance-identical) output. Real
    check against run_job (not a mock), covering both seed_job_rng
    (global RNG determinism) and torch.use_deterministic_algorithms."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    from manifest import build_S1_jobs
    jobs = build_S1_jobs(n_x=48, n_iters=5, seeds=[0])
    job = jobs[0]
    device = rm.get_device()
    commit = rm.git_commit_hash()

    r1 = rm.run_job(job, device, commit)
    r2 = rm.run_job(job, device, commit)
    assert r1["psnr"] == r2["psnr"], (r1["psnr"], r2["psnr"])
    assert r1["diffraction_efficiency"] == r2["diffraction_efficiency"]
    assert r1["loss_curve"] == r2["loss_curve"]
    print(f"deterministic rerun OK: psnr {r1['psnr']} == {r2['psnr']} (bitwise)")


def test_job_id_and_seed_job_rng():
    job = dict(experiment_id="M1", config_hash="abc123", method_id="MIL", seed=2)
    assert rm.job_id(job) == "M1_abc123_MIL_seed2"
    # seeding must be deterministic given the same job identity
    rm.seed_job_rng(job)
    a = torch.rand(3).tolist()
    rm.seed_job_rng(job)
    b = torch.rand(3).tolist()
    assert a == b
    # and different for a different job identity
    rm.seed_job_rng(dict(job, seed=3))
    c = torch.rand(3).tolist()
    assert a != c
    print("job_id/seed_job_rng OK")


def test_run_manifest_until_complete_happy_path():
    tmp = tempfile.mkdtemp(prefix="stall_test_")
    try:
        rm.set_results_root(tmp)
        status = rm.run_manifest_until_complete("S1", chunk_minutes=60, n_x=32, n_iters=3)
        assert status["complete"] and status["n_done"] == status["n_total"] == 90
        print("run_manifest_until_complete happy path OK:", status["n_done"], "jobs")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stall_detection_raises_after_two_stuck_chunks():
    """Spec Sec. 1.3: two consecutive chunks with no increase in n_done
    must raise, naming the manifest and the stuck job. Can't actually
    hang a job in a test, so this monkeypatches run_manifest to return a
    fixed, non-advancing status -- a deterministic simulation of the
    real stall condition, not a mock of the detection logic itself."""
    calls = []

    def fake_stuck(name, max_minutes, n_x=1024, n_iters=800, converge_tol=1e-4):
        calls.append(1)
        return dict(complete=False, n_run=0, n_done_already=5, n_done=5,
                   n_total=40, n_remaining=35, last_attempted_job_id="M1_abc_MIL_seed0")

    orig = rm.run_manifest
    rm.run_manifest = fake_stuck
    try:
        try:
            rm.run_manifest_until_complete("M1", chunk_minutes=1)
            assert False, "expected ManifestStallError"
        except rm.ManifestStallError as e:
            assert "M1" in str(e) and "M1_abc_MIL_seed0" in str(e)
            assert len(calls) == 3  # 1st call sets baseline, 2nd+3rd are the 2 stuck chunks
            print("stall detection OK, raised after", len(calls), "chunks:", e)
    finally:
        rm.run_manifest = orig


def test_probe_exit_code_reflects_gate1():
    """The Colab notebook's Gate-1 check now reads probe()'s return value
    directly (in-process) rather than parsing printed text or an exit
    code -- but the CLI's exit code (used by anyone running run_manifest.py
    from a shell) must also reflect Gate 1 correctly: 0 under budget, 2
    over. Exercised via subprocess against the real CLI entrypoint."""
    import subprocess
    # --allow-cpu: this test's purpose is the Gate-1 exit code, not the
    # hard GPU assertion (which is exercised separately and correctly
    # rejects a no-GPU CLI call by design -- this dev/CI environment has
    # no GPU at all).
    env_ok = subprocess.run(
        [sys.executable, "-m", "experiments.run_manifest", "--manifest", "S1",
         "--probe", "--n-x", "32", "--n-iters", "3", "--allow-cpu"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True)
    assert env_ok.returncode == 0, (env_ok.returncode, env_ok.stdout, env_ok.stderr)
    print("probe CLI exit code under budget OK: 0")


def test_cli_rejects_no_gpu_without_allow_cpu():
    """The hard GPU assertion (spec Sec. 1.2) must actually reject a
    no-GPU environment when --allow-cpu is not passed -- this dev
    environment has no GPU, so this is a real (not mocked) check."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "experiments.run_manifest", "--manifest", "S1",
         "--probe", "--n-x", "32", "--n-iters", "3"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True, text=True)
    assert result.returncode != 0, "expected the hard GPU assertion to fail without a GPU"
    assert "No GPU allocated" in result.stderr, result.stderr
    print("CLI correctly rejects no-GPU run without --allow-cpu")


if __name__ == "__main__":
    test_config_hash_is_method_independent_but_path_is_not()
    test_manifest_end_to_end_and_resume()
    test_schema_fields_present()
    test_run_manifest_returns_completion_status()
    test_atomic_write_json_verifies_readable()
    test_clean_partial_files_removes_stray_tmp_only()
    test_deterministic_rerun_matches()
    test_job_id_and_seed_job_rng()
    test_run_manifest_until_complete_happy_path()
    test_stall_detection_raises_after_two_stuck_chunks()
    test_probe_exit_code_reflects_gate1()
    test_cli_rejects_no_gpu_without_allow_cpu()
    print("PASSED")
