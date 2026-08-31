"""
Exposure/index-profile figure (Phase 3 Tier-1 item 3): E(x) and Delta-n(x)
for media-blind SGD vs. media-in-the-loop, at the same three K points and
config as R1 (experiments/make_r1_reconstructions.py) -- same target,
seed, budget, so results are the deterministic twin of R1's reconstructions,
just with the intermediate exposure/index arrays kept instead of discarded.

Checkpointed the same way R1 is: one file per K, atomic write, skips
already-done K's on rerun.

Usage: python -m experiments.make_r1_profiles
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
from methods import media_blind_sgd, media_in_the_loop
from manifest import DEFAULT_MEDIUM, period_from_K, S1_K_POINTS
from make_r1_reconstructions import build_bars_target, N_X, DX, LAM_UM, N_ITERS, BUDGET, SEED

CKPT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results_r1_profiles"))
OUT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results_r1_profiles.json"))


def ckpt_path(K: float) -> str:
    return os.path.join(CKPT_DIR, f"K_{K:.6f}.json")


def run_one_K(K: float, rec, bpm, device) -> dict:
    period_px = period_from_K(K, DX)
    target = build_bars_target(period_px, N_X, device)

    t0 = time.time()
    E_bsgd, recon_bsgd, _ = media_blind_sgd(target, rec, bpm, n_iters=N_ITERS, lr=5e-2,
                                            dose_budget=1.0, seed=SEED, contrast_cap=BUDGET)
    t_bsgd = time.time() - t0

    t0 = time.time()
    E_mil, recon_mil, _ = media_in_the_loop(target, rec, bpm, n_iters=N_ITERS, lr=5e-2,
                                            dose_budget=1.0, seed=SEED, contrast_cap=BUDGET,
                                            verbose=False)
    t_mil = time.time() - t0

    with torch.no_grad():
        dn_bsgd = rec.p.dn_max * (E_bsgd - E_bsgd.mean())  # BSGD's own linear assumption, matching its design objective
        dn_mil = rec(E_mil)  # MIL's exposure through the real twin

    print(f"K={K:.3f} done ({t_bsgd:.0f}s + {t_mil:.0f}s)", flush=True)
    return dict(
        K=K, period_px=period_px, budget=BUDGET, seed=SEED,
        E_bsgd=E_bsgd.detach().cpu().tolist(), E_mil=E_mil.detach().cpu().tolist(),
        dn_bsgd=dn_bsgd.detach().cpu().tolist(), dn_mil=dn_mil.detach().cpu().tolist(),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medium = MediumParams(**DEFAULT_MEDIUM)
    rec = NPDDRecorder(N_X, DX, t_total=10.0, n_steps=300, params=medium, dtype=torch.float32).to(device)
    bpm = SlabBPM(N_X, DX, LAM_UM, medium.thickness, n_z=32, n0=medium.n0, dtype=torch.complex64).to(device)

    os.makedirs(CKPT_DIR, exist_ok=True)
    for K in S1_K_POINTS:
        path = ckpt_path(K)
        if os.path.exists(path):
            print(f"K={K:.3f}: checkpoint exists, skipping", flush=True)
            continue
        result = run_one_K(K, rec, bpm, device)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, path)

    results = []
    for K in S1_K_POINTS:
        path = ckpt_path(K)
        if not os.path.exists(path):
            print(f"K={K:.3f}: still missing, not merging")
            return
        results.append(json.load(open(path)))
    with open(OUT_PATH, "w") as f:
        json.dump(dict(results=results), f)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
