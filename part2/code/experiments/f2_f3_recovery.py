"""
F2/F3 -- Method comparison and recovery curves (paper Figures 2 and 3).

For each medium-parameter setting, run the four methods on a bank of 1D
targets (binary bar patterns of varying spatial frequency + a pseudo-random
"natural" spectrum target), report PSNR and in-support diffraction
efficiency with mean +/- std over seeds.

Sweep axes (each swept with others at defaults):
    sigma      in {0.02, 0.05, 0.08, 0.12, 0.20, 0.30} um
    D0         in {0.01, 0.03, 0.1, 0.3, 1.0}          um^2/s
    dn_max     in {1e-3, 2e-3, 3.5e-3, 5e-3, 6e-3}
    shrinkage  in {0, 0.003, 0.01, 0.02, 0.03}
    thickness  in {10, 20, 30, 50, 80} um
"""
import sys, os, itertools, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
                       media_in_the_loop, media_blind_sgd, media_blind_gs,
                       oracle_ideal, psnr, diffraction_efficiency)

torch.set_default_dtype(torch.float64)
N_X, DX, LAM = 512, 0.1, 0.405
SEEDS = [0, 1, 2]           # bump to 5 for the paper
N_ITERS = 300               # bump to 800-1500 for the paper


def make_targets(device):
    x = torch.arange(N_X, device=device)
    targets = {}
    for period in [16, 32, 64]:
        targets[f"bars{period}"] = ((x // (period // 2)) % 2).double()
    g = torch.zeros(N_X, device=device)
    torch.manual_seed(7)
    for _ in range(6):
        c = torch.randint(N_X // 8, 7 * N_X // 8, (1,)).item()
        w = torch.randint(8, 40, (1,)).item()
        g[max(0, c - w):min(N_X, c + w)] = torch.rand(1).item() + 0.3
    targets["spots"] = g
    return targets


def run_setting(params, device="cpu"):
    rec = NPDDRecorder(N_X, DX, t_total=10, n_steps=250, params=params)
    bpm = SlabBPM(N_X, DX, LAM, params.thickness, n_z=24, n0=params.n0)
    rows = []
    for name, tgt in make_targets(device).items():
        mask = (tgt > 0.05).double()
        for seed in SEEDS:
            _, r_ours, _ = media_in_the_loop(tgt, rec, bpm, n_iters=N_ITERS,
                                             seed=seed, verbose=False)
            _, r_blind, _ = media_blind_sgd(tgt, rec, bpm, n_iters=N_ITERS, seed=seed)
            _, r_gs = media_blind_gs(tgt, rec, bpm)
            _, r_orc = oracle_ideal(tgt, rec, bpm, n_iters=N_ITERS, seed=seed)
            rows.append(dict(
                target=name, seed=seed,
                psnr_ours=psnr(r_ours, tgt), psnr_blind=psnr(r_blind, tgt),
                psnr_gs=psnr(r_gs, tgt), psnr_oracle=psnr(r_orc, tgt),
                de_ours=diffraction_efficiency(r_ours, mask),
                de_blind=diffraction_efficiency(r_blind, mask),
            ))
    return rows


def main():
    sweeps = {
        "sigma": [0.02, 0.05, 0.08, 0.12, 0.20, 0.30],
        "D0": [0.01, 0.03, 0.1, 0.3, 1.0],
        "dn_max": [1e-3, 2e-3, 3.5e-3, 5e-3, 6e-3],
        "shrinkage": [0.0, 0.003, 0.01, 0.02, 0.03],
        "thickness": [10.0, 20.0, 30.0, 50.0, 80.0],
    }
    results = {}
    for axis, values in sweeps.items():
        for v in values:
            p = MediumParams(); setattr(p, axis, v)
            key = f"{axis}={v}"
            print(f"== {key} ==")
            results[key] = run_setting(p)
    with open("results_f2f3.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote results_f2f3.json")


if __name__ == "__main__":
    main()
