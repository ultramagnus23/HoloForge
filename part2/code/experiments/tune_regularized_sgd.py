"""
WP3 item 1: small grid search selecting regularized_media_blind_sgd's
tv_weight per budget, TUNED ON THE TWIN (selected by realized PSNR
against the real recorder+BPM, not the naive linear proxy the optimizer
itself minimizes) -- explicitly disclosed here and in the manuscript,
per the work order's requirement that twin-tuned regularization not be
presented as if discovered unsupervised.

Cost: 4 tv_weights x 3 representative K x 3 budgets x 200 (short) iters
= 36 short runs, ~15-20s each on this machine's RTX 3050 -- a few
minutes total, run before the full 800-iter production grid so the
chosen weight is fixed going into it, not re-tuned per job.

Writes results/summary/rsgd_tv_weight.json: {budget: best_tv_weight}.
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch

from holomedia import NPDDRecorder, MediumParams, SlabBPM
from manifest import DEFAULT_MEDIUM, period_from_K
from methods import run_method

torch.set_default_dtype(torch.float32)

N_X, DX = 1024, 51.2 / 1024
LAM_UM = 0.405
TUNE_N_ITERS = 200  # short: tuning only needs a relative ranking, not convergence
TUNE_K_POINTS = [1.963495, 4.833219, 10.471976]  # low/mid/high, representative
TV_WEIGHTS = [0.0, 0.001, 0.01, 0.1]
BUDGETS = [2.0, 4.0, 8.0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medium = MediumParams(**DEFAULT_MEDIUM)
    rec = NPDDRecorder(N_X, DX, t_total=10.0, n_steps=300, params=medium, dtype=torch.float32).to(device)
    bpm = SlabBPM(N_X, DX, LAM_UM, medium.thickness, n_z=32, n0=medium.n0, dtype=torch.complex64).to(device)
    x = torch.arange(N_X, device=device)

    best_per_budget = {}
    all_results = []
    for budget in BUDGETS:
        scores = {tv: [] for tv in TV_WEIGHTS}
        for K in TUNE_K_POINTS:
            period_px = period_from_K(K, DX)
            target = ((x // (period_px // 2)) % 2).float()
            for tv in TV_WEIGHTS:
                t0 = time.time()
                r = run_method("RSGD", target, rec, bpm, seed=0, n_iters=TUNE_N_ITERS,
                              dose_budget=1.0, contrast_cap=budget, tv_weight=tv)
                dt = time.time() - t0
                scores[tv].append(r["psnr"])
                all_results.append(dict(budget=budget, K=K, tv_weight=tv,
                                        psnr=r["psnr"], wall_s=dt))
                print(f"B={budget:.0f} K={K:.2f} tv={tv:<6} psnr={r['psnr']:.3f} ({dt:.1f}s)", flush=True)
        mean_scores = {tv: sum(v) / len(v) for tv, v in scores.items()}
        best_tv = max(mean_scores, key=mean_scores.get)
        best_per_budget[budget] = best_tv
        print(f"-- budget={budget:.0f}: mean PSNR per tv_weight = {mean_scores}, "
              f"best = {best_tv}", flush=True)

    out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..",
                                              "results", "summary", "rsgd_tv_weight.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict(best_per_budget=best_per_budget, all_results=all_results,
                       tv_weights_tested=TV_WEIGHTS, tune_K_points=TUNE_K_POINTS,
                       tune_n_iters=TUNE_N_ITERS), f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
