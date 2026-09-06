"""
S6: JOINT twin-miscalibration Monte Carlo runner (WP5, Applied Optics
revision).

    python -m experiments.run_s6_joint_mismatch                  # full grid
    python -m experiments.run_s6_joint_mismatch --allow-cpu       # local smoke

WHAT THIS ADDS OVER S3
-----------------------
S3 (experiments/run_s3_mismatch.py) already builds the design/evaluate
split this reuses: optimize E* ONCE at theta_nominal, evaluate that SAME
fixed exposure at some theta_prime, with no re-optimization. But S3 wrongs
exactly one NPDD parameter at a time (D0, sigma, kappa, dn_max), one at a
time in isolation. A real twin-calibration error is never that clean --
every parameter is uncertain simultaneously. S6 draws all four parameters
independently per Monte Carlo trial (manifest.build_S6_joint_conditions)
and evaluates the SAME S3 exposures against each joint draw.

COST
----
Zero new optimization: this reuses S3's already-cached designs
(results/S3/_designs/*.pt -- see run_s3_mismatch.design_one/design_path,
imported directly). Every S6 evaluation is a single forward NPDD+BPM
pass, same cost class as S3's own evaluation stage.

OUTPUT
------
Standard Phase-1.2 manifest schema under experiment_id "S6", so
analysis/aggregate.py loads it with no special-casing beyond the
dedicated s6_joint_mismatch_summary function (paired gain is grouped by
draw_id, not by (param, pct) the way S3's is).
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from manifest import build_S3_designs, build_S6_joint_conditions, s6_result_config, config_hash
import run_manifest
from run_manifest import (DTYPE, atomic_write_json, git_commit_hash, get_device,
                          device_name, assert_gpu_and_report, result_path)
from run_s3_mismatch import design_one, evaluate


def run(n_x=1024, n_iters=800, converge_tol=1e-4, seeds=None, device=None,
       dtype=DTYPE, n_draws=None, pct_range=None):
    device = device if device is not None else get_device()
    commit = git_commit_hash()
    designs = build_S3_designs(n_x=n_x, n_iters=n_iters,
                              converge_tol=converge_tol, seeds=seeds)
    kwargs = {}
    if n_draws is not None:
        kwargs["n_draws"] = n_draws
    if pct_range is not None:
        kwargs["pct_range"] = pct_range
    conditions = build_S6_joint_conditions(**kwargs)
    print(f"[S6] {len(designs)} designs (reused from S3) x {len(conditions)} "
         f"joint draws = {len(designs) * len(conditions)} evaluations", flush=True)

    n_written = n_skipped = 0
    for job in designs:
        cfg = job["config"]
        E = None
        for cond in conditions:
            eval_cfg = s6_result_config(cfg, cond)
            h = config_hash(eval_cfg)
            path = result_path("S6", job["method_id"], h, job["seed"])
            if os.path.exists(path):
                n_skipped += 1
                continue
            if E is None:
                E = design_one(job, device, dtype=dtype)  # loads S3's cached .pt
            t0 = time.time()
            scored = evaluate(E, cfg, cond["medium"], device, dtype=dtype)
            atomic_write_json(path, dict(
                git_commit=commit, experiment_id="S6",
                method_id=job["method_id"], seed=job["seed"], config=eval_cfg,
                config_hash=h, device=str(device), device_name=device_name(device),
                dtype=str(dtype), loss_curve=[], iterations_run=cfg["n_iters"],
                early_stop_reason="evaluation_only", wall_s=time.time() - t0,
                peak_mem_mb=None, **scored))
            n_written += 1
    print(f"[S6] complete: {n_written} evaluations written, "
         f"{n_skipped} already done.", flush=True)
    return dict(n_written=n_written, n_skipped=n_skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-x", type=int, default=1024)
    ap.add_argument("--n-iters", type=int, default=800)
    ap.add_argument("--converge-tol", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-draws", type=int, default=None)
    ap.add_argument("--pct-range", type=float, default=None)
    ap.add_argument("--results-dir", type=str, default=None,
                    help="override results/ output directory (smoke testing)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="skip the hard GPU assertion (local dev only)")
    args = ap.parse_args()

    if args.results_dir:
        run_manifest.set_results_root(args.results_dir)

    device = get_device() if args.allow_cpu else assert_gpu_and_report()
    run(n_x=args.n_x, n_iters=args.n_iters, converge_tol=args.converge_tol,
       seeds=args.seeds if args.seeds else None, device=device,
       n_draws=args.n_draws, pct_range=args.pct_range)


if __name__ == "__main__":
    main()
