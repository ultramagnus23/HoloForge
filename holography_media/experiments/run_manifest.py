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
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from holomedia import NPDDRecorder, MediumParams, SlabBPM

from manifest import BUILDERS, build_all_jobs, PAPER_SEEDS
from methods import run_method

HERE = os.path.dirname(__file__)
RESULTS_ROOT = os.path.join(HERE, "..", "results")


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


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def build_target(spec: dict, n_x: int, device) -> torch.Tensor:
    kind = spec["kind"]
    if kind == "bars":
        period_px = spec["period_px"]
        x = torch.arange(n_x, device=device)
        return ((x // (period_px // 2)) % 2).double()
    elif kind == "spots":
        g = torch.zeros(n_x, device=device)
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)  # atomic on POSIX and Windows (same filesystem)


def run_job(job: dict, device, commit: str) -> dict:
    cfg = job["config"]
    n_x, dx = cfg["n_x"], cfg["dx"]
    medium = MediumParams(**cfg["medium"])
    n_steps = cfg.get("n_steps", 300)
    n_z = cfg.get("n_z", 32)

    rec = NPDDRecorder(n_x, dx, t_total=10.0, n_steps=n_steps, params=medium).to(device)
    bpm = SlabBPM(n_x, dx, cfg["lam_um"], medium.thickness, n_z=n_z, n0=medium.n0).to(device)
    target = build_target(cfg["target"], n_x, device)

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
                       n_total=len(jobs), n_remaining=n_remaining)
        print(f"[run_manifest] {job['experiment_id']}/{job['method_id']}/"
              f"seed{job['seed']}/{job['config_hash']} ...", flush=True)
        try:
            result = run_job(job, device, commit)
        except NotImplementedError as e:
            print(f"  SKIPPED (not runnable yet): {e}")
            continue
        atomic_write_json(path, result)
        n_run += 1
        print(f"  done: {result['wall_s']:.1f}s, psnr={result['psnr']:.2f}dB, "
              f"iters={result['iterations_run']}/{job['config']['n_iters']}")

    print(f"[run_manifest] manifest {name!r} complete: {n_run} run this "
          f"session, {n_done_already} already done, {len(jobs)} total.")
    return dict(complete=True, n_run=n_run, n_done_already=n_done_already,
               n_total=len(jobs), n_remaining=0)


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
    args = ap.parse_args()

    if args.results_dir:
        set_results_root(args.results_dir)

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
