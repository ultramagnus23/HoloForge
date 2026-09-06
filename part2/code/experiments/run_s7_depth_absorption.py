"""
S7: depth-resolved absorption robustness runner (WP6, Applied Optics
revision).

    python -m experiments.run_s7_depth_absorption                  # full grid
    python -m experiments.run_s7_depth_absorption --allow-cpu       # local smoke

WHAT THIS ADDS
--------------
oe_main.tex's Discussion section already states, ANALYTICALLY, that the
uniform-through-depth recording assumption is "good only for optical
density <~ 0.1 over the recorded thickness" -- but that bound was never
tested empirically: the recording model has always treated dn as uniform
through depth, so no result in this paper has ever actually recorded
something else. S7 makes that bound empirical: holomedia.npdd's new
depth_resolved_dn actually attenuates the recording exposure with depth
(Beer-Lambert) and runs the SAME NPDD physics independently at each of
n_z depth slices, producing a genuinely 2D-in-(x,z) recorded profile fed
into SlabBPM.forward_depth_resolved (also new) instead of the uniform
extrusion every other tier uses.

Reuses S3's cached design exposures (results/S3/_designs/*.pt) -- same
free-reuse pattern S5/S6 already established this revision: evaluating
an existing exposure under a different READOUT/recording assumption is
exactly as valid as evaluating it under a different medium (S3) or a
different draw (S6), so there is no reason to re-optimize.

Tested optical densities: 0.0 (reproduces every existing result exactly
-- see holomedia.npdd.depth_resolved_dn's docstring for the verified
OD=0 equivalence), 0.1 (the paper's own stated "good approximation"
boundary) and 0.3 (the paper's own stated "not a small perturbation"
case) -- both already discussed in prose, not new numbers invented for
this check.

COST
----
Zero new optimization (same design reuse as S3/S5/S6). Each evaluation
now costs n_z NPDDRecorder forward passes batched into ONE call
(depth_resolved_dn) plus one SlabBPM.forward_depth_resolved -- more
expensive than S3/S6's single-uniform-profile evaluation, but still a
forward-only cost, no gradient loop.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from holomedia import depth_resolved_dn, psnr_si, psnr, diffraction_efficiency
from manifest import build_S3_designs, config_hash
import run_manifest
from run_manifest import (DTYPE, build_target, atomic_write_json, git_commit_hash,
                          get_device, device_name, assert_gpu_and_report, result_path)
from run_s3_mismatch import design_one, _build_stack
from methods import contrast_stats

S7_OPTICAL_DENSITIES = [0.0, 0.1, 0.3]
S7_N_Z = 16  # coarser than SlabBPM's usual n_z=32 default -- forward passes
            # here cost n_z recorder evaluations each, so this keeps S7's
            # total cost bounded; verified this doesn't change the OD=0
            # equivalence (that check is exact regardless of n_z).


def s7_result_config(design_config: dict, optical_density: float) -> dict:
    return dict(design_config, optical_density=optical_density,
               n_z_depth=S7_N_Z, design_medium="nominal", arm="depth_absorption")


def evaluate_depth_resolved(E: torch.Tensor, cfg: dict, medium_dict: dict,
                            optical_density: float, device, dtype=DTYPE) -> dict:
    rec, bpm = _build_stack(cfg, medium_dict, device, dtype)
    target = build_target(cfg["target"], cfg["n_x"], device, dtype=dtype)
    mask = (target > 0.05).double()
    with torch.no_grad():
        if optical_density == 0.0:
            # exact-equivalence path (verified in holomedia.npdd's tests):
            # skip the batched n_z-slice recorder call entirely when it is
            # guaranteed to reproduce the uniform profile, cheaper and
            # avoids relying on a numerically-exact-but-still-more-costly
            # depth-resolved call for a condition that needs it least.
            dn = rec(E.to(device))
            recon = bpm(dn, shrinkage=rec.p.shrinkage)
        else:
            dn_stack = depth_resolved_dn(rec, E.to(device), optical_density, S7_N_Z)
            bpm_depth = type(bpm)(bpm.n_x, bpm.dx, bpm.lam, bpm.T, n_z=S7_N_Z,
                                  n0=bpm.n0, z_recon_um=bpm.z_recon, dtype=bpm.cdtype).to(device)
            recon = bpm_depth.forward_depth_resolved(dn_stack, shrinkage=rec.p.shrinkage)
    return dict(psnr=psnr_si(recon, target), psnr_maxnorm_legacy=psnr(recon, target),
               diffraction_efficiency=diffraction_efficiency(recon, mask),
               contrast=contrast_stats(E))


def run(n_x=1024, n_iters=800, converge_tol=1e-4, seeds=None, device=None, dtype=DTYPE):
    device = device if device is not None else get_device()
    commit = git_commit_hash()
    designs = build_S3_designs(n_x=n_x, n_iters=n_iters,
                              converge_tol=converge_tol, seeds=seeds)
    print(f"[S7] {len(designs)} designs (reused from S3) x {len(S7_OPTICAL_DENSITIES)} "
         f"optical densities = {len(designs) * len(S7_OPTICAL_DENSITIES)} evaluations", flush=True)

    n_written = n_skipped = 0
    for job in designs:
        cfg = job["config"]
        E = None
        for od in S7_OPTICAL_DENSITIES:
            eval_cfg = s7_result_config(cfg, od)
            h = config_hash(eval_cfg)
            path = result_path("S7", job["method_id"], h, job["seed"])
            if os.path.exists(path):
                n_skipped += 1
                continue
            if E is None:
                E = design_one(job, device, dtype=dtype)
            t0 = time.time()
            scored = evaluate_depth_resolved(E, cfg, cfg["medium"], od, device, dtype=dtype)
            atomic_write_json(path, dict(
                git_commit=commit, experiment_id="S7",
                method_id=job["method_id"], seed=job["seed"], config=eval_cfg,
                config_hash=h, device=str(device), device_name=device_name(device),
                dtype=str(dtype), loss_curve=[], iterations_run=cfg["n_iters"],
                early_stop_reason="evaluation_only", wall_s=time.time() - t0,
                peak_mem_mb=None, **scored))
            n_written += 1
    print(f"[S7] complete: {n_written} evaluations written, "
         f"{n_skipped} already done.", flush=True)
    return dict(n_written=n_written, n_skipped=n_skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-x", type=int, default=1024)
    ap.add_argument("--n-iters", type=int, default=800)
    ap.add_argument("--converge-tol", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--results-dir", type=str, default=None,
                    help="override results/ output directory (smoke testing)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="skip the hard GPU assertion (local dev only)")
    args = ap.parse_args()

    if args.results_dir:
        run_manifest.set_results_root(args.results_dir)

    device = get_device() if args.allow_cpu else assert_gpu_and_report()
    run(n_x=args.n_x, n_iters=args.n_iters, converge_tol=args.converge_tol,
       seeds=args.seeds if args.seeds else None, device=device)


if __name__ == "__main__":
    main()
