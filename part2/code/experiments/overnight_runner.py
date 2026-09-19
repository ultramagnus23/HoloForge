"""
Overnight unattended wrapper around run_manifest.py.

Two things run_manifest.py doesn't do on its own, which an unattended
overnight session needs:
  1. A wall-clock budget: stop cleanly before morning instead of running
     indefinitely (or, with --max-minutes alone, stopping only after that
     one chunk -- there's no built-in "keep going until N hours have
     elapsed, across restarts").
  2. Crash tolerance: an uncaught exception (e.g. a CUDA device drop) kills
     run_manifest.py's process outright. Resume is safe by construction
     (a job is "done" iff its result file exists on disk, written
     atomically) -- so the right response to a crash is "start a fresh
     process, it'll skip everything already written," not "stop for the
     night." This wrapper does that automatically, capped at
     --max-consecutive-crashes so a persistently broken GPU/config doesn't
     spin all night doing nothing.

Each chunk runs as its own subprocess (not an in-process retry loop) so a
CUDA fault takes down only that subprocess, never this wrapper.

Multiple manifests can be queued in one session with --manifests (comma-
separated, priority order): the wall-clock budget is shared across all of
them, moving to the next manifest as soon as the current one reports
complete, so a session that finishes manifest A early spends the rest of
the budget on B instead of idling.

Usage:
    python -m experiments.overnight_runner --manifests S1,S2 --hours 10
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from manifest import BUILDERS
from run_manifest import result_path, RESULTS_ROOT, apply_shard

HERE = os.path.dirname(__file__)


def jobs_remaining(manifest_name: str, n_x: int, n_iters: int, converge_tol: float,
                   shard: tuple[int, int] | None = None) -> tuple[int, int]:
    """(n_done, n_total) via the same filesystem check run_manifest.py's
    resume logic uses -- no CUDA/GPU needed, safe to call from the wrapper
    itself to decide whether to bother launching another chunk."""
    jobs = apply_shard(BUILDERS[manifest_name](n_x=n_x, n_iters=n_iters, converge_tol=converge_tol), shard)
    n_done = sum(
        1 for j in jobs
        if os.path.exists(result_path(j["experiment_id"], j["method_id"], j["config_hash"], j["seed"]))
    )
    return n_done, len(jobs)


def run_one_manifest(manifest: str, deadline: float, chunk_minutes: float,
                     max_consecutive_crashes: int, n_x: int, n_iters: int,
                     converge_tol: float, shard: tuple[int, int] | None = None) -> bool:
    """Drive one manifest with crash-tolerant chunked restarts until it's
    complete or the shared deadline is hit. Returns True iff it completed
    (so the caller knows whether to move on to the next manifest or stop)."""
    n_done, n_total = jobs_remaining(manifest, n_x, n_iters, converge_tol, shard)
    print(f"[overnight] manifest={manifest!r}: {n_done}/{n_total} jobs already done",
          flush=True)
    if n_done >= n_total:
        print(f"[overnight] {manifest!r} already complete.", flush=True)
        return True

    attempt = 0
    consecutive_crashes = 0
    while time.time() < deadline:
        n_done, n_total = jobs_remaining(manifest, n_x, n_iters, converge_tol, shard)
        if n_done >= n_total:
            print(f"[overnight] manifest {manifest!r} complete "
                  f"({n_done}/{n_total}).", flush=True)
            return True

        remaining_min = (deadline - time.time()) / 60.0
        chunk = min(chunk_minutes, remaining_min)
        if chunk <= 0.5:
            break
        attempt += 1
        print(f"\n[overnight] {manifest} attempt {attempt}: {n_done}/{n_total} done, "
              f"chunk={chunk:.1f}min, {remaining_min:.1f}min left in shared budget",
              flush=True)
        cmd = [sys.executable, "-m", "experiments.run_manifest",
               "--manifest", manifest, "--max-minutes", str(chunk),
               "--n-x", str(n_x), "--n-iters", str(n_iters),
               "--converge-tol", str(converge_tol)]
        if shard is not None:
            cmd += ["--shard", f"{shard[0]}/{shard[1]}"]
        proc = subprocess.run(cmd, cwd=os.path.join(HERE, ".."))
        if proc.returncode == 0:
            consecutive_crashes = 0
        else:
            consecutive_crashes += 1
            print(f"[overnight] {manifest} chunk exited with code {proc.returncode} "
                  f"(crash #{consecutive_crashes} in a row) -- retrying, "
                  f"resume will skip everything already written.", flush=True)
            if consecutive_crashes >= max_consecutive_crashes:
                print(f"[overnight] {consecutive_crashes} consecutive crashes on "
                      f"{manifest!r} -- this looks persistent, not a transient "
                      f"GPU drop. Stopping so it doesn't spin all night for nothing.",
                      flush=True)
                sys.exit(1)
            time.sleep(15)  # let the device settle before retrying

    n_done, n_total = jobs_remaining(manifest, n_x, n_iters, converge_tol, shard)
    print(f"[overnight] {manifest} at budget cutoff: {n_done}/{n_total} done.", flush=True)
    return n_done >= n_total


def _prevent_idle_sleep() -> None:
    """Windows: ask the OS not to idle-sleep while this (long-lived) process
    runs. Per-thread execution state -- released automatically when the
    process exits, and it changes no system power setting. No-op elsewhere.
    Does NOT stop a closed lid or a manual Sleep."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    _prevent_idle_sleep()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True,
                    help="comma-separated manifest names, in priority order "
                         "(e.g. 'S1,S2') -- the wall-clock budget is shared "
                         "across all of them")
    ap.add_argument("--hours", type=float, required=True,
                    help="wall-clock budget for this overnight session")
    ap.add_argument("--chunk-minutes", type=float, default=15.0,
                    help="max-minutes passed to each run_manifest.py chunk "
                         "-- smaller means faster crash detection, larger "
                         "means less subprocess-restart overhead")
    ap.add_argument("--max-consecutive-crashes", type=int, default=5)
    ap.add_argument("--n-x", type=int, default=1024)
    ap.add_argument("--n-iters", type=int, default=800)
    ap.add_argument("--converge-tol", type=float, default=1e-4)
    ap.add_argument("--shard", type=str, default=None,
                    help="'i/N': this worker takes every N-th job, offset i, of each "
                         "manifest (launch N workers, one per shard, to share the GPU)")
    args = ap.parse_args()
    shard = None
    if args.shard is not None:
        _i, _n = args.shard.split("/")
        shard = (int(_i), int(_n))

    manifests = args.manifests.split(",")
    for m in manifests:
        assert m in BUILDERS, f"unknown manifest {m!r}, choices: {list(BUILDERS.keys())}"

    deadline = time.time() + args.hours * 3600
    print(f"[overnight] queue={manifests}, shared budget={args.hours:.1f}h, "
          f"n_x={args.n_x}", flush=True)

    for manifest in manifests:
        if time.time() >= deadline:
            print(f"[overnight] budget exhausted before reaching {manifest!r}.", flush=True)
            break
        run_one_manifest(manifest, deadline, args.chunk_minutes,
                         args.max_consecutive_crashes, args.n_x,
                         args.n_iters, args.converge_tol, shard)

    print(f"\n[overnight] session done. Status:", flush=True)
    for m in manifests:
        n_done, n_total = jobs_remaining(m, args.n_x, args.n_iters, args.converge_tol, shard)
        print(f"  {m}: {n_done}/{n_total}", flush=True)
    print("[overnight] rerun the same command tomorrow night to pick up "
          "where this left off (already-done jobs are skipped).", flush=True)


if __name__ == "__main__":
    main()
