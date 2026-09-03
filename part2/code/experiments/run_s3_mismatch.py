"""
S3: twin-MISCALIBRATION robustness runner.

    python -m experiments.run_s3_mismatch                  # full grid
    python -m experiments.run_s3_mismatch --allow-cpu      # local smoke
    python -m experiments.run_s3_mismatch --n-x 256 --n-iters 50 --seeds 0

WHAT QUESTION THIS ANSWERS, AND WHY S2 DOES NOT ANSWER IT
---------------------------------------------------------
The paper claims MIL's advantage is "not sensitive to getting any single
NPDD parameter wrong". S2 cannot support that claim: it perturbs the
medium and then optimizes AND evaluates BOTH arms on the perturbed medium,
so the perturbation is common to both arms -- exactly the condition under
which Sec. A's paired-comparison argument says a systematic twin error
cancels. Under S2 nothing is ever *wrong*; it is merely *different*.

S3 breaks the arms apart in time, which is what "miscalibration" actually
means for a write-once medium:

    design time   optimize E* ONCE against the twin at theta_nominal
    record time   evaluate that SAME, already-fixed E* on a twin at
                  theta_prime != theta_nominal

There is no re-optimization at theta_prime. The exposure never learns it
was designed for the wrong medium.

WHY THE BSGD CONTROL NEEDS NO SPECIAL HANDLING
----------------------------------------------
media_blind_sgd optimizes against dn = c_lin * (E - mean E) with
c_lin = dn_max, under a scale-invariant objective. A uniform rescale of
c_lin leaves si_mse exactly unchanged, and no other medium parameter
enters its loop at all -- so its designed exposure is theta-independent by
construction. "Designed at theta_nominal" and "designed at theta_prime"
are the same tensor for BSGD, which is what makes it the honest paired
baseline at every perturbation: it is the arm that cannot be
miscalibrated, because it was never calibrated.

COST
----
The design stage is the only optimization: 2 methods x 3 K x n_seeds.
Everything else is a single forward NPDD pass per (design, condition), so
the 28-condition evaluation grid costs forward passes, not optimizations.

OUTPUT
------
Standard Phase-1.2 manifest schema, so analysis/aggregate.py loads and
pairs these with no special-casing (see manifest.s3_result_config): the
EVALUATION medium goes in config["medium"], with mismatch_param /
mismatch_pct / design_medium alongside it. Both arms share those fields,
so MIL and BSGD land in the same pairing group at each theta_prime.

Resumability: designed exposures are checkpointed to
results/S3/_designs/, and each evaluation writes its own content-hashed
JSON -- rerunning the same command skips everything already on disk.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
                       media_in_the_loop, media_blind_sgd,
                       psnr_si, psnr, diffraction_efficiency)

from manifest import (build_S3_conditions, build_S3_designs, s3_result_config,
                      config_hash)
from methods import contrast_stats
import run_manifest
from run_manifest import (DTYPE, build_target, atomic_write_json,
                          git_commit_hash, get_device, device_name,
                          assert_gpu_and_report, result_path, seed_job_rng)


def design_path(job: dict) -> str:
    # run_manifest.RESULTS_ROOT is read through the module (not imported by
    # value) so set_results_root() from a test or a --results-dir run is
    # actually honored here too.
    return os.path.join(run_manifest.RESULTS_ROOT, "S3", "_designs",
                        job["config_hash"],
                        f"{job['method_id']}_seed{job['seed']}.pt")


def _build_stack(cfg: dict, medium_dict: dict, device, dtype):
    """Recorder + BPM for one medium. Everything that depends on the medium
    is rebuilt from scratch (NPDDRecorder caches sigma inside G_hat, SlabBPM
    caches thickness and n0), so a perturbed theta_prime really is a
    different twin rather than the nominal one with a field poked."""
    medium = MediumParams(**medium_dict)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    rec = NPDDRecorder(cfg["n_x"], cfg["dx"], t_total=10.0,
                       n_steps=cfg.get("n_steps", 300), params=medium,
                       dtype=dtype).to(device)
    bpm = SlabBPM(cfg["n_x"], cfg["dx"], cfg["lam_um"], medium.thickness,
                  n_z=cfg.get("n_z", 32), n0=medium.n0, dtype=cdtype).to(device)
    return rec, bpm


def design_one(job: dict, device, dtype=DTYPE) -> torch.Tensor:
    """Optimize one exposure at theta_nominal. Cached to disk."""
    path = design_path(job)
    if os.path.exists(path):
        return torch.load(path, map_location=device, weights_only=True)

    seed_job_rng(job)
    cfg = job["config"]
    rec, bpm = _build_stack(cfg, cfg["medium"], device, dtype)
    target = build_target(cfg["target"], cfg["n_x"], device, dtype=dtype)

    t0 = time.time()
    if job["method_id"] == "MIL":
        E, _, _ = media_in_the_loop(target, rec, bpm, n_iters=cfg["n_iters"],
                                    dose_budget=cfg["dose_budget"], seed=job["seed"],
                                    verbose=False, converge_tol=cfg.get("converge_tol"),
                                    contrast_cap=cfg.get("contrast_cap"))
    elif job["method_id"] == "BSGD":
        E, _, _ = media_blind_sgd(target, rec, bpm, n_iters=cfg["n_iters"],
                                  dose_budget=cfg["dose_budget"], seed=job["seed"],
                                  contrast_cap=cfg.get("contrast_cap"))
    else:
        raise ValueError(
            f"S3 design stage covers MIL/BSGD only, got {job['method_id']!r}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(E.detach().cpu(), tmp)
    os.replace(tmp, path)
    print(f"  [design] {job['method_id']}/seed{job['seed']}/"
          f"K={cfg['K_nominal']:.3f} optimized in {time.time() - t0:.1f}s",
          flush=True)
    return E.detach().to(device)


def evaluate(E: torch.Tensor, cfg: dict, medium_dict: dict, device,
             dtype=DTYPE) -> dict:
    """Score an already-fixed exposure on a twin at some medium."""
    rec, bpm = _build_stack(cfg, medium_dict, device, dtype)
    target = build_target(cfg["target"], cfg["n_x"], device, dtype=dtype)
    mask = (target > 0.05).double()
    with torch.no_grad():
        recon = bpm(rec(E.to(device)), shrinkage=rec.p.shrinkage)
    return dict(psnr=psnr_si(recon, target),
                psnr_maxnorm_legacy=psnr(recon, target),
                diffraction_efficiency=diffraction_efficiency(recon, mask),
                contrast=contrast_stats(E))


def run(n_x=1024, n_iters=800, converge_tol=1e-4, seeds=None, device=None,
        dtype=DTYPE):
    device = device if device is not None else get_device()
    commit = git_commit_hash()
    designs = build_S3_designs(n_x=n_x, n_iters=n_iters,
                               converge_tol=converge_tol, seeds=seeds)
    conditions = build_S3_conditions(n_x=n_x)
    print(f"[S3] {len(designs)} designs x {len(conditions)} mismatch conditions "
          f"= {len(designs) * len(conditions)} evaluations", flush=True)

    n_written = n_skipped = 0
    for job in designs:
        cfg = job["config"]
        # Designed lazily: if every evaluation for this design is already on
        # disk, the (expensive) optimization is never re-run on resume.
        E = None
        for cond in conditions:
            eval_cfg = s3_result_config(cfg, cond)
            h = config_hash(eval_cfg)
            path = result_path("S3", job["method_id"], h, job["seed"])
            if os.path.exists(path):
                n_skipped += 1
                continue
            if E is None:
                E = design_one(job, device, dtype=dtype)
            t0 = time.time()
            scored = evaluate(E, cfg, cond["medium"], device, dtype=dtype)
            atomic_write_json(path, dict(
                git_commit=commit, experiment_id="S3",
                method_id=job["method_id"], seed=job["seed"], config=eval_cfg,
                config_hash=h, device=str(device), device_name=device_name(device),
                dtype=str(dtype), loss_curve=[], iterations_run=cfg["n_iters"],
                early_stop_reason="evaluation_only", wall_s=time.time() - t0,
                peak_mem_mb=None, **scored))
            n_written += 1
    print(f"[S3] complete: {n_written} evaluations written, "
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
