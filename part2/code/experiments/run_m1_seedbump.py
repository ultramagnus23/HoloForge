"""
Seed bump for the M1 headline grid (Phase 3 Tier-1 item 1): raises the
iterative methods (BSGD/MIL/ORC/ORU) from 3 seeds to 8, re-using
run_manifest.py's exact execution path (same result schema, same
results/M1/ directory, same resumability) so the extra 5 seeds' worth of
jobs merge transparently with the existing 3-seed data -- nothing about
the aggregation pipeline (analysis/aggregate.py) needs to change, it just
sees more result files per config.

GS and LPC are excluded from the seed bump: they're closed-form, seed-
independent (manifest.py's build_M1_jobs already only emits seed 0 for
them), so extra seeds there are meaningless.

Usage: python -m experiments.run_m1_seedbump [--max-minutes N]
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from manifest import build_M1_jobs, PAPER_SEEDS
from run_manifest import (get_device, git_commit_hash, run_job, atomic_write_json,
                          result_path, apply_shard)

SEED_BUMP = list(range(8))  # 0-7; 0,1,2 already exist and are skipped by the resume check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--n-x", type=int, default=1024)
    ap.add_argument("--n-iters", type=int, default=800)
    ap.add_argument("--shard", type=str, default=None)
    args = ap.parse_args()

    shard = None
    if args.shard:
        i, n = args.shard.split("/")
        shard = (int(i), int(n))

    jobs = build_M1_jobs(n_x=args.n_x, n_iters=args.n_iters, seeds=SEED_BUMP)
    jobs = apply_shard(jobs, shard)

    device = get_device()
    commit = git_commit_hash()
    t_start = time.time()
    n_done_already = n_run = 0
    for job in jobs:
        path = result_path(job["experiment_id"], job["method_id"], job["config_hash"], job["seed"])
        if os.path.exists(path):
            n_done_already += 1
            continue
        if args.max_minutes is not None and (time.time() - t_start) / 60.0 >= args.max_minutes:
            print(f"[seedbump] --max-minutes={args.max_minutes} reached: "
                 f"{n_run} run, {n_done_already} already done, "
                 f"{len(jobs) - n_run - n_done_already} remaining.")
            return
        print(f"[seedbump] {job['experiment_id']}/{job['method_id']}/seed{job['seed']}/"
             f"{job['config_hash']} ...", flush=True)
        try:
            result = run_job(job, device, commit)
        except NotImplementedError as e:
            print(f"  SKIPPED: {e}")
            continue
        atomic_write_json(path, result)
        n_run += 1
        print(f"  done: {result['wall_s']:.1f}s, psnr={result['psnr']:.2f}dB", flush=True)

    print(f"[seedbump] complete: {n_run} run this session, {n_done_already} "
         f"already done, {len(jobs)} total (seeds {SEED_BUMP}).")


if __name__ == "__main__":
    main()
