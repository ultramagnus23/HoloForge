"""
Regenerates target/BSGD-recon/MIL-recon 2D arrays for the 3 study targets
at one representative budget (4x, the middle of the tested range), seed 0,
for the paper's 2D reconstruction figure. experiments/run_2d.py's main
grid only saves scalar metrics (psnr_si, diffraction_efficiency) to keep
results/M1_2D/ small across 54 jobs -- this script re-runs just 6 of those
jobs (3 targets x {BSGD, MIL}, budget=4x, seed=0) with the same config, so
results are bit-for-bit reproducible, and additionally saves the arrays.

Usage: python -m experiments.make_2d_reconstructions
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from manifest_2d import build_2d_jobs, TARGET_FNS
from run_2d import _build_recorder_bpm, RUNNERS, DEVICE, DTYPE

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "..", "results_2d_reconstructions.json")


def main():
    jobs = build_2d_jobs(budgets=[4.0], seeds=[0])
    out = []
    for job in jobs:
        config = job["config"]
        target = TARGET_FNS[config["target_kind"]]().to(DEVICE, DTYPE)
        rec, bpm = _build_recorder_bpm(config)
        E, recon, _ = RUNNERS[job["method_id"]](target, rec, bpm, config, job["seed"])
        out.append(dict(
            target_kind=config["target_kind"], method_id=job["method_id"],
            budget=config["contrast_cap"], seed=job["seed"],
            target=target.cpu().tolist(), recon=recon.cpu().tolist(),
        ))
        print(f"done: {config['target_kind']} {job['method_id']}")
    with open(OUT_PATH, "w") as f:
        json.dump(dict(results=out), f)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
