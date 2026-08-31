"""
Runner for the bounded 2D study (experiments/manifest_2d.py).

Usage:
    python -m experiments.run_2d --calibrate          # 1 job, timing only
    python -m experiments.run_2d                       # full grid, resumable
    python -m experiments.run_2d --max-minutes 170      # time-boxed

Resume semantics match run_manifest.py: a job is done iff
results/M1_2D/{config_hash}/{method_id}_seed{seed}.json already exists.
Each result is written atomically (tmp file + rename) after it completes,
so an interrupted run loses at most the one job in flight.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from holomedia import MediumParams, contrast_project, dose_project, si_mse, psnr_si, diffraction_efficiency
from holomedia.npdd3d import NPDDRecorder3D
from holomedia.diffraction3d import SlabBPM3D

from manifest_2d import build_2d_jobs, TARGET_FNS

HERE = os.path.dirname(__file__)
RESULTS_ROOT = os.path.join(HERE, "..", "results", "M1_2D")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _seeded_init_theta_2d(n: int, device, dtype, seed: int, eps: float = 1e-2):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    theta = eps * torch.randn(n, n, generator=g, dtype=dtype)
    return theta.to(device).requires_grad_(True)


def _build_recorder_bpm(config: dict):
    medium = MediumParams(**config["medium"])
    rec = NPDDRecorder3D(n_x=config["n"], n_y=config["n"], dx=config["dx"], dy=config["dx"],
                         n_steps=config["n_steps"], params=medium, dtype=DTYPE).to(DEVICE)
    bpm = SlabBPM3D(n_x=config["n"], n_y=config["n"], dx=config["dx"], dy=config["dx"],
                    wavelength_um=config["lam_um"], thickness_um=medium.thickness,
                    n_z=config["n_z"], n0=medium.n0, dtype=torch.complex128).to(DEVICE)
    return rec, bpm


def run_mil(target, rec, bpm, config, seed):
    theta = _seeded_init_theta_2d(config["n"], DEVICE, DTYPE, seed)
    opt = torch.optim.Adam([theta], lr=config["lr"])
    loss_hist = []
    for it in range(config["n_iters"]):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = contrast_project(E, config["dose_budget"], contrast_cap=config["contrast_cap"])
        dn = rec(E)
        recon = bpm(dn)
        loss = si_mse(recon, target)
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == config["n_iters"] - 1:
            loss_hist.append((it, float(loss.detach())))
    with torch.no_grad():
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6,
                            config["dose_budget"], contrast_cap=config["contrast_cap"])
        recon = bpm(rec(E))
    return E.detach(), recon.detach(), loss_hist


def run_bsgd(target, rec, bpm, config, seed):
    """Media-blind SGD: optimize assuming a linear medium (dn = dn_max *
    (E - mean(E))), evaluate on the real 2D twin -- the 2D analogue of
    holomedia.optimize.media_blind_sgd. No NPDD forward pass inside the
    optimization loop, so this is cheap relative to MIL by construction."""
    theta = _seeded_init_theta_2d(config["n"], DEVICE, DTYPE, seed)
    opt = torch.optim.Adam([theta], lr=config["lr"])
    dn_max = rec.p.dn_max
    loss_hist = []
    for it in range(config["n_iters"]):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = contrast_project(E, config["dose_budget"], contrast_cap=config["contrast_cap"])
        dn_ideal = dn_max * (E - E.mean())
        recon = bpm(dn_ideal)
        loss = si_mse(recon, target)
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == config["n_iters"] - 1:
            loss_hist.append((it, float(loss.detach())))
    with torch.no_grad():
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6,
                            config["dose_budget"], contrast_cap=config["contrast_cap"])
        dn_ideal = dn_max * (E - E.mean())
        # Evaluated on the REAL twin, as current practice would be -- not
        # on the linear medium it was optimized against. This matches the
        # 1D media_blind_sgd's arm design exactly.
        recon = bpm(rec(E))
    return E.detach(), recon.detach(), loss_hist


RUNNERS = {"MIL": run_mil, "BSGD": run_bsgd}


def run_job(job: dict) -> dict:
    config = job["config"]
    target = TARGET_FNS[config["target_kind"]]().to(DEVICE, DTYPE)
    rec, bpm = _build_recorder_bpm(config)
    t0 = time.time()
    E, recon, loss_hist = RUNNERS[job["method_id"]](target, rec, bpm, config, job["seed"])
    wall_s = time.time() - t0
    mean = float(E.mean())
    result = dict(
        experiment_id=job["experiment_id"], method_id=job["method_id"], seed=job["seed"],
        config=config, config_hash=job["config_hash"],
        device=str(DEVICE), dtype=str(DTYPE),
        wall_s=wall_s, loss_curve=loss_hist,
        psnr_si=psnr_si(recon, target),
        diffraction_efficiency=diffraction_efficiency(recon, target),
        contrast_realized=(float(E.max()) / mean) if mean > 0 else None,
    )
    return result


def _result_path(job: dict) -> str:
    d = os.path.join(RESULTS_ROOT, job["config_hash"])
    return os.path.join(d, f"{job['method_id']}_seed{job['seed']}.json")


def _write_atomic(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="Run exactly 1 MIL job at a reduced n_iters, report wall_s "
                         "and the linear extrapolation to the real n_iters, then exit "
                         "without writing to results/.")
    ap.add_argument("--calibrate-iters", type=int, default=50)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args()

    if args.calibrate:
        jobs = build_2d_jobs(target_kinds=["disc"], budgets=[2.0], seeds=[0], methods=["MIL"])
        job = jobs[0]
        job["config"] = dict(job["config"], n_iters=args.calibrate_iters)
        print(f"device={DEVICE}, calibrating at n_iters={args.calibrate_iters} "
             f"(real jobs use n_iters={build_2d_jobs()[0]['config']['n_iters']})")
        result = run_job(job)
        real_n_iters = build_2d_jobs()[0]["config"]["n_iters"]
        per_iter = result["wall_s"] / args.calibrate_iters
        est_full = per_iter * real_n_iters
        n_mil_jobs = len(build_2d_jobs(methods=["MIL"]))
        print(f"wall_s at {args.calibrate_iters} iters: {result['wall_s']:.2f}")
        print(f"per-iter: {per_iter:.4f}s -> estimated wall_s at n_iters={real_n_iters}: {est_full:.1f}")
        print(f"n MIL jobs in full grid: {n_mil_jobs} -> estimated total MIL wall (serial): "
             f"{est_full * n_mil_jobs / 3600:.2f} h")
        return

    jobs = build_2d_jobs()
    t_start = time.time()
    n_done = n_run = 0
    for job in jobs:
        path = _result_path(job)
        if os.path.exists(path):
            n_done += 1
            continue
        if args.max_minutes is not None and (time.time() - t_start) / 60.0 > args.max_minutes:
            print(f"time budget exhausted after {n_run} jobs this session")
            break
        result = run_job(job)
        _write_atomic(path, result)
        n_run += 1
        print(f"[{n_run}] {job['method_id']} seed{job['seed']} {job['config']['target_kind']} "
             f"budget{job['config']['contrast_cap']}: psnr_si={result['psnr_si']:.3f} "
             f"wall_s={result['wall_s']:.1f}")
    print(f"done. {n_done} already present, {n_run} run this session, "
         f"{len(jobs) - n_done - n_run} remaining.")


if __name__ == "__main__":
    main()
