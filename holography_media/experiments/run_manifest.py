"""
Phase 1.1/1.2/1.4: resumable manifest runner + unified result schema + probe mode.

Usage:
    python -m experiments.run_manifest --manifest E1 --max-minutes 170
    python -m experiments.run_manifest --manifest E1 --probe
    python -m experiments.run_manifest --manifest all --max-minutes 170

Resume semantics: a job is "done" iff its result file
results/{experiment_id}/{config_hash}/seed{N}.json already exists on disk.
Starting the same manifest again skips every done job and picks up where
it left off -- worst case on a Colab session death is losing the one job
that was in flight (it's written only after it fully completes, atomically
via write-to-tmp-then-rename).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from holomedia import NPDDRecorder, MediumParams, SlabBPM

from manifest import BUILDERS, build_all_jobs, PAPER_SEEDS
from methods import run_method

HERE = os.path.dirname(__file__)
RESULTS_ROOT = os.path.join(HERE, "..", "results")

# Precision policy (spec Sec. 1.1): float32 by default for the production
# manifest pipeline, justified by evidence, not assumed. Measured on a
# representative config (media_in_the_loop, n_x=256, n_iters=100): float32
# vs float64 PSNR difference was 7.5e-7 dB, ~5000x smaller than the
# seed-to-seed std (3.8e-3 dB, 3 seeds) -- float32 changes the answer by
# far less than seed noise already does, so it's defensible per the
# spec's own criterion. This does NOT touch holomedia's library-wide
# defaults (still float64, unchanged) or any of the already-run legacy
# experiment scripts (run_prelim.py etc., which pin float64 explicitly or
# via NPDDRecorder's constructor default) -- only NEW jobs run through
# this manifest pipeline are affected, so no already-committed result's
# reproducibility is put at risk by this change.
DTYPE = torch.float32
CDTYPE = torch.complex64


def set_results_root(path: str) -> None:
    """Override the results directory -- used by tests/smoke runs so they
    don't write into the real results/ tree that Phase 3's GPU runs use."""
    global RESULTS_ROOT
    RESULTS_ROOT = path

# 40 T4-hour Gate-1 threshold, per master prompt Phase 1.4.
GATE1_HOURS = 40.0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[run_manifest] WARNING: no CUDA device -- running on CPU. "
          "Do not report these numbers as GPU-scale.")
    return torch.device("cpu")


def device_name(device) -> str:
    """Real hardware name (e.g. 'Tesla T4'), not just 'cuda'/'cpu' -- so
    runs from different hardware are distinguishable after the fact
    (spec Sec. 1.2). Recorded into every result JSON's device_name field."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    import platform
    return f"cpu:{platform.processor() or platform.machine() or 'unknown'}"


def assert_gpu_and_report() -> torch.device:
    """Hard assertion at the top of a real run (spec Sec. 1.2): no silent
    CPU fallback for actual science runs. This is deliberately NOT called
    by get_device() itself (which stays CPU-fallback-friendly for tests
    and local development, matching the require_gpu=False/True pattern
    already used in experiments/_gpu_common.py) -- only main()'s real
    entrypoint calls this, so `pytest`-style direct calls to run_manifest()
    /probe() from tests are unaffected."""
    assert torch.cuda.is_available(), "No GPU allocated -- stop and fix the runtime."
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"[run_manifest] GPU: {torch.cuda.get_device_name(device)}, "
         f"{props.total_memory / 1e9:.1f} GB")
    return device


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def build_target(spec: dict, n_x: int, device, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    kind = spec["kind"]
    if kind == "bars":
        period_px = spec["period_px"]
        x = torch.arange(n_x, device=device)
        # was `.double()` unconditionally -- silently forced float64
        # regardless of what dtype the recorder/bpm for this job actually
        # used, while the "spots" branch below defaulted to whatever
        # torch's ambient global dtype happened to be. Both branches now
        # honor the job's own dtype explicitly and consistently.
        return ((x // (period_px // 2)) % 2).to(dtype)
    elif kind == "spots":
        g = torch.zeros(n_x, device=device, dtype=dtype)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(spec.get("seed", 7))
        n_spots = spec.get("n_spots", 5)
        for _ in range(n_spots):
            c = int(torch.randint(n_x // 8, 7 * n_x // 8, (1,), generator=gen))
            w = int(torch.randint(12, 48, (1,), generator=gen))
            amp = float(torch.rand(1, generator=gen)) + 0.3
            g[max(0, c - w):min(n_x, c + w)] = amp
        return g
    elif kind == "image_slice":
        raise NotImplementedError(
            "image_slice targets require an actual image asset -- see "
            "build_E5_jobs's docstring. Not fabricated by this runner.")
    else:
        raise ValueError(f"unknown target kind {kind!r}")


def result_path(experiment_id: str, method_id: str, config_hash: str, seed: int) -> str:
    # method_id MUST be part of the path: config_hash is computed from
    # `config` alone (not method_id), so two jobs that differ only in
    # method (e.g. M2 vs M4 on the identical target/medium/seed) share a
    # config_hash. Without method_id in the path they'd collide on the
    # same file -- caught by the manifest smoke test (M4 jobs silently
    # skipped as "already done" when only M2 had actually run).
    return os.path.join(RESULTS_ROOT, experiment_id, config_hash, f"{method_id}_seed{seed}.json")


def atomic_write_json(path: str, data: dict) -> None:
    """Spec Sec. 1.5: serialize to a LOCAL temp file first (fast, no FUSE
    latency in the JSON-encoding step -- irrelevant for the small JSONs
    this pipeline writes, but avoids doing incremental writes directly
    against a network-backed mount either way), copy that single already-
    complete file to the final path (possibly Drive-backed), atomically
    rename into place, then VERIFY it round-trips through json.load before
    returning -- a job is not "done" until its output is confirmed
    readable, not just written. Raises (job is NOT marked done, caller's
    atomic_write_json call fails loudly) rather than silently leaving a
    corrupt result file for a later resume to trust."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    local_tmp = os.path.join(
        tempfile.gettempdir(),
        f"holoforge_{os.getpid()}_{time.time_ns()}.json.tmp")
    with open(local_tmp, "w") as f:
        json.dump(data, f, indent=1)
    final_tmp = path + ".tmp"
    shutil.copyfile(local_tmp, final_tmp)
    os.remove(local_tmp)
    os.replace(final_tmp, path)  # atomic on POSIX and Windows (same filesystem)
    with open(path) as f:
        json.load(f)  # verify-readable; raises on truncated/corrupt output


def clean_partial_files(results_root: str) -> int:
    """Spec Sec. 1.5: delete leftover *.tmp files on startup -- these are
    the local_tmp/final_tmp artifacts of a write that was interrupted
    (process killed) mid-copy/rename, before atomic_write_json's final
    os.replace. A stray .tmp is never itself a valid result (result_path
    never points at a .tmp), so it can only be debris; leaving it around
    risks confusing a later manual inspection into thinking a job is
    further along than it is. Returns the count removed."""
    if not os.path.isdir(results_root):
        return 0
    n = 0
    for root, _dirs, files in os.walk(results_root):
        for fn in files:
            if fn.endswith(".tmp"):
                os.remove(os.path.join(root, fn))
                n += 1
    return n


def job_id(job: dict) -> str:
    return f"{job['experiment_id']}_{job['config_hash']}_{job['method_id']}_seed{job['seed']}"


def seed_job_rng(job: dict) -> None:
    """Deterministically seed torch/numpy/python random from the job's
    full identity (spec Sec. 1.6). Defensive, not the primary reproducibility
    mechanism: every stochastic init in holomedia.optimize already uses its
    own local torch.Generator keyed on `seed` specifically (so the 5 paper
    seeds are reproducible AND meaningfully different from each other, not
    collapsed to one hash) -- this additionally seeds GLOBAL RNG state so
    any code path that uses it directly (rather than a passed generator)
    is also deterministic per job, without perturbing what `seed` itself
    means to the methods that already handle it correctly."""
    jid = job_id(job)
    job_seed = int(hashlib.sha256(jid.encode()).hexdigest()[:8], 16)
    torch.manual_seed(job_seed)
    np.random.seed(job_seed % (2**32))
    random.seed(job_seed)


def run_job(job: dict, device, commit: str, dtype: torch.dtype = DTYPE) -> dict:
    seed_job_rng(job)
    cfg = job["config"]
    n_x, dx = cfg["n_x"], cfg["dx"]
    medium = MediumParams(**cfg["medium"])
    n_steps = cfg.get("n_steps", 300)
    n_z = cfg.get("n_z", 32)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    rec = NPDDRecorder(n_x, dx, t_total=10.0, n_steps=n_steps, params=medium,
                       dtype=dtype).to(device)
    bpm = SlabBPM(n_x, dx, cfg["lam_um"], medium.thickness, n_z=n_z, n0=medium.n0,
                 dtype=cdtype).to(device)
    target = build_target(cfg["target"], n_x, device, dtype=dtype)

    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    result = run_method(job["method_id"], target, rec, bpm, seed=job["seed"],
                        n_iters=cfg["n_iters"], dose_budget=cfg["dose_budget"],
                        contrast_cap=cfg.get("contrast_cap"),
                        converge_tol=cfg.get("converge_tol"))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.time() - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6
                  if device.type == "cuda" else None)

    # downsample loss curve to <=200 points (Phase 1.2 schema requirement)
    hist = result.pop("loss_history")
    if len(hist) > 200:
        stride = len(hist) // 200 + 1
        hist = hist[::stride]

    return dict(
        git_commit=commit, experiment_id=job["experiment_id"],
        method_id=job["method_id"], seed=job["seed"], config=cfg,
        config_hash=job["config_hash"], device=str(device),
        device_name=device_name(device), dtype=str(dtype),
        loss_curve=hist, iterations_run=result["iterations_run"],
        early_stop_reason=result["early_stop_reason"], wall_s=wall_s,
        peak_mem_mb=peak_mem_mb, psnr=result["psnr"],
        diffraction_efficiency=result["diffraction_efficiency"],
        contrast=result["contrast"],
    )


def run_manifest(name: str, max_minutes: float | None, n_x=1024, n_iters=800,
                 converge_tol=1e-4):
    device = get_device()
    commit = git_commit_hash()
    if name == "all":
        jobs = build_all_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    else:
        jobs = BUILDERS[name](n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)

    t_start = time.time()
    n_done_already = n_run = 0
    last_attempted_job_id = None
    for job in jobs:
        path = result_path(job["experiment_id"], job["method_id"], job["config_hash"], job["seed"])
        if os.path.exists(path):
            n_done_already += 1
            continue
        if max_minutes is not None and (time.time() - t_start) / 60.0 >= max_minutes:
            n_remaining = len(jobs) - n_run - n_done_already
            print(f"[run_manifest] --max-minutes={max_minutes} reached, "
                  f"exiting cleanly before starting a new job "
                  f"({n_run} run this session, {n_done_already} already done, "
                  f"{n_remaining} remaining).")
            return dict(complete=False, n_run=n_run, n_done_already=n_done_already,
                       n_done=n_run + n_done_already, n_total=len(jobs),
                       n_remaining=n_remaining, last_attempted_job_id=last_attempted_job_id)
        last_attempted_job_id = job_id(job)
        print(f"[run_manifest] {job['experiment_id']}/{job['method_id']}/"
              f"seed{job['seed']}/{job['config_hash']} ...", flush=True)
        try:
            result = run_job(job, device, commit)
        except NotImplementedError as e:
            print(f"  SKIPPED (not runnable yet): {e}")
            continue
        atomic_write_json(path, result)
        n_run += 1
        # Heartbeat (spec Sec. 1.3): timestamp, job ID, duration, peak VRAM
        # -- one line per completed job, so a Colab session's scrollback
        # alone is enough to tell when/how long each job took without
        # needing to open the result JSONs.
        vram = f"{result['peak_mem_mb']:.0f}MB" if result["peak_mem_mb"] is not None else "n/a"
        print(f"  [heartbeat] {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"job={last_attempted_job_id} duration={result['wall_s']:.1f}s "
              f"peak_vram={vram}")
        print(f"  done: {result['wall_s']:.1f}s, psnr={result['psnr']:.2f}dB, "
              f"iters={result['iterations_run']}/{job['config']['n_iters']}")

    print(f"[run_manifest] manifest {name!r} complete: {n_run} run this "
          f"session, {n_done_already} already done, {len(jobs)} total.")
    return dict(complete=True, n_run=n_run, n_done_already=n_done_already,
               n_done=n_run + n_done_already, n_total=len(jobs), n_remaining=0,
               last_attempted_job_id=last_attempted_job_id)


class ManifestStallError(RuntimeError):
    """Raised when two consecutive chunks make zero progress on a
    manifest (spec Sec. 1.3) -- a single job is stuck taking longer than
    the chunk budget. Looping forever would silently burn GPU-hours; this
    surfaces it immediately, naming the manifest and the stuck job."""


def run_manifest_until_complete(name: str, chunk_minutes: float = 60,
                                n_x=1024, n_iters=800, converge_tol=1e-4,
                                max_stall_chunks: int = 2):
    """Repeatedly calls run_manifest in chunk_minutes-sized chunks until
    it reports complete. Extracted as a real, importable, testable
    function (rather than only living as notebook-cell source, which
    can't be unit tested) so the stall-detection logic itself has a
    regression test. The Colab notebook's auto-resume cell calls this
    directly per manifest."""
    clean_partial_files(RESULTS_ROOT)  # spec Sec. 1.5: on startup, before any work
    last_n_done = None
    stall_count = 0
    while True:
        status = run_manifest(name, max_minutes=chunk_minutes, n_x=n_x,
                              n_iters=n_iters, converge_tol=converge_tol)
        if status["complete"]:
            return status
        if last_n_done is not None and status["n_done"] <= last_n_done:
            stall_count += 1
            if stall_count >= max_stall_chunks:
                raise ManifestStallError(
                    f"STALL DETECTED in manifest {name!r}: {max_stall_chunks} "
                    f"consecutive {chunk_minutes}-minute chunks made zero progress "
                    f"(stuck on job {status['last_attempted_job_id']!r}). A single "
                    f"job is taking longer than {chunk_minutes} min -- investigate "
                    f"that job's config before re-running (e.g. it may need a "
                    f"smaller n_x/n_iters, or there's a real bug).")
        else:
            stall_count = 0
        last_n_done = status["n_done"]


def probe(name: str, n_x=1024, n_iters=800, converge_tol=1e-4):
    """Run exactly one representative job per experiment (Phase 1.4). Prints
    a T4-hour extrapolation and the Gate-1 threshold check -- does NOT
    decide anything; the user reviews this table and picks a reduction
    option if it's over budget."""
    device = get_device()
    commit = git_commit_hash()
    names = list(BUILDERS.keys()) if name == "all" else [name]
    rows = []
    for exp_name in names:
        jobs = BUILDERS[exp_name](n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
        # pick a representative iterative job (not a closed-form M1/M3, so
        # the timing reflects the actual compute-heavy path)
        rep = next((j for j in jobs if j["method_id"] in ("M2", "M4", "M5a", "M5b")), jobs[0])
        print(f"[probe] {exp_name}: running representative job "
              f"({rep['method_id']}, {len(jobs)} total jobs in this manifest) ...")
        result = run_job(rep, device, commit)
        per_job_s = result["wall_s"]
        total_s = per_job_s * len(jobs)
        rows.append(dict(experiment=exp_name, n_jobs=len(jobs),
                         per_job_s=per_job_s, total_hours=total_s / 3600.0))
        print(f"  representative job: {per_job_s:.1f}s -> "
              f"{len(jobs)} jobs -> {total_s/3600.0:.2f}h extrapolated")

    grand_total = sum(r["total_hours"] for r in rows)
    print("\n" + "=" * 60)
    print(f"{'experiment':12s} {'n_jobs':>8s} {'per_job_s':>10s} {'total_hours':>12s}")
    for r in rows:
        print(f"{r['experiment']:12s} {r['n_jobs']:8d} {r['per_job_s']:10.1f} {r['total_hours']:12.2f}")
    print("-" * 60)
    print(f"{'TOTAL':12s} {'':8s} {'':10s} {grand_total:12.2f}")
    print("=" * 60)
    if grand_total > GATE1_HOURS:
        e1_row = next((r for r in rows if r["experiment"] == "E1"), None)
        e1_hours = e1_row["total_hours"] if e1_row else 0.0
        print(f"\nGATE 1: FAILED -- projected {grand_total:.1f}h exceeds the "
              f"{GATE1_HOURS:.0f}h threshold. Per the master prompt, this requires "
              f"a decision, not a unilateral reduction:\n"
              f"  (a) 3 seeds instead of 5 for E1 (would cut E1's ~{e1_hours:.1f}h by ~40%)\n"
              f"  (b) coarser K/sweep grids\n"
              f"  (c) drop E6 (wavelength rerun; already have a single-seed supplement result)\n"
              f"Report this table back and choose a combination before running 'full'.")
    else:
        print(f"\nGATE 1: PASSED -- projected {grand_total:.1f}h is within the "
              f"{GATE1_HOURS:.0f}h threshold -- no reduction needed.")
    return grand_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, choices=list(BUILDERS.keys()) + ["all"])
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--n-x", type=int, default=1024)
    ap.add_argument("--n-iters", type=int, default=800)
    ap.add_argument("--converge-tol", type=float, default=1e-4)
    ap.add_argument("--results-dir", type=str, default=None,
                    help="override results/ output directory (for smoke testing)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="skip the hard GPU assertion (local dev/smoke-testing "
                         "only -- never use for an actual science run)")
    args = ap.parse_args()

    # spec Sec. 1.6: deterministic algorithms where feasible. Verified
    # (tests/test_manifest_runner.py::test_deterministic_rerun_matches)
    # this holds for every op the current pipeline actually uses (FFT,
    # elementwise, Adam) on CPU. Not yet verified on CUDA specifically --
    # some cuBLAS/cuDNN ops lack deterministic kernels and would raise
    # RuntimeError under strict mode; warn_only=True degrades to a
    # warning instead of a hard failure if that happens on a real GPU run,
    # so a first Colab run surfaces the problem via a printed warning
    # rather than crashing the whole manifest.
    torch.use_deterministic_algorithms(True, warn_only=True)

    if args.results_dir:
        set_results_root(args.results_dir)

    n_cleaned = clean_partial_files(RESULTS_ROOT)  # spec Sec. 1.5
    if n_cleaned:
        print(f"[run_manifest] cleaned {n_cleaned} leftover .tmp file(s) from a "
             f"previously-interrupted write.")

    if args.allow_cpu:
        print("[run_manifest] --allow-cpu set: skipping the hard GPU assertion. "
              "Do NOT report whatever this run produces as GPU-scale.")
    else:
        assert_gpu_and_report()

    if args.probe:
        grand_total = probe(args.manifest, n_x=args.n_x, n_iters=args.n_iters,
                           converge_tol=args.converge_tol)
        # Exit code distinguishes Gate-1 pass/fail so calling scripts (the
        # Colab one-click cell) can act on it without parsing printed text.
        # 0 = under budget, 2 = over budget (needs your reduction decision),
        # matching the master prompt's "do not decide unilaterally."
        sys.exit(2 if grand_total > GATE1_HOURS else 0)
    else:
        run_manifest(args.manifest, args.max_minutes, n_x=args.n_x,
                    n_iters=args.n_iters, converge_tol=args.converge_tol)


if __name__ == "__main__":
    main()
