"""Remaining sweeps: shrinkage, thickness, D0, and the high-K sigma probe.

The high-K probe tests the prediction from Eq. 5 that the limiting mechanism
switches from transport-limited (D0 K^2 / F0) to blur-limited (Ghat(K)) as K
rises -- i.e. the sigma sweep should be FLAT at low K and STEEP at high K.
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
                       media_in_the_loop, media_blind_sgd, media_blind_gs,
                       oracle_ideal, psnr, diffraction_efficiency)

torch.set_default_dtype(torch.float64)
N_X, DX, LAM = 256, 0.1, 0.405
N_ITERS, N_STEPS, N_Z, SEEDS = 150, 100, 12, [0, 1]


def bars(period_px):
    x = torch.arange(N_X)
    return ((x // (period_px // 2)) % 2).double()


def run_cell(params, tgt):
    rec = NPDDRecorder(N_X, DX, t_total=10, n_steps=N_STEPS, params=params)
    bpm = SlabBPM(N_X, DX, LAM, params.thickness, n_z=N_Z, n0=params.n0)
    mask = (tgt > 0.05).double()
    out = []
    for s in SEEDS:
        _, ro, _ = media_in_the_loop(tgt, rec, bpm, n_iters=N_ITERS, seed=s, verbose=False)
        _, rb = media_blind_sgd(tgt, rec, bpm, n_iters=N_ITERS, seed=s)
        _, rg = media_blind_gs(tgt, rec, bpm)
        _, rc = oracle_ideal(tgt, rec, bpm, n_iters=N_ITERS, seed=s)
        out.append(dict(seed=s, ours=psnr(ro, tgt), blind=psnr(rb, tgt),
                        gs=psnr(rg, tgt), oracle=psnr(rc, tgt),
                        de_ours=diffraction_efficiency(ro, mask),
                        de_blind=diffraction_efficiency(rb, mask)))
    return out


def main(part):
    t0 = time.time()
    f = "results_prelim2.json"
    R = json.load(open(f)) if os.path.exists(f) else {}

    if part == "D":  # shrinkage
        R["shrinkage"] = {}
        for s in [0.0, 0.01, 0.03]:
            p = MediumParams(); p.shrinkage = s
            R["shrinkage"][str(s)] = run_cell(p, bars(16))
            print(f"[D] shrinkage={s} {time.time()-t0:.0f}s", flush=True)

    if part == "E":  # thickness
        R["thickness"] = {}
        for T in [10.0, 30.0, 80.0]:
            p = MediumParams(); p.thickness = T
            R["thickness"][str(T)] = run_cell(p, bars(16))
            print(f"[E] T={T} {time.time()-t0:.0f}s", flush=True)

    if part == "F":  # D0 (transport)
        R["D0"] = {}
        for d in [0.01, 0.1, 1.0]:
            p = MediumParams(); p.D0 = d
            R["D0"][str(d)] = run_cell(p, bars(16))
            print(f"[F] D0={d} {time.time()-t0:.0f}s", flush=True)

    if part == "G":  # HIGH-K sigma crossover probe (period 8px -> K=7.85)
        R["sigma_highK"] = {}
        for sg in [0.02, 0.08, 0.20, 0.30]:
            p = MediumParams(); p.sigma = sg
            R["sigma_highK"][str(sg)] = run_cell(p, bars(8))
            print(f"[G] sigma={sg} @K=7.85 {time.time()-t0:.0f}s", flush=True)

    json.dump(R, open(f, "w"), indent=1)
    print(f"DONE {part} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
