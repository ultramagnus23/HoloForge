"""Preliminary CPU-scale experiment suite (real results, reduced scale).

Exp A (CLIFF): default medium; bar targets spanning spatial frequency K;
               quality of ours vs baselines vs K, against predicted K_c.
Exp B (SIGMA): recovery vs non-locality length sigma.
Exp C (DNMAX): recovery vs index budget dn_max.

Scale: n_x=256, n_steps=100, n_iters=150, 2 seeds. Paper-scale = GPU rerun
with n_x>=1024, n_steps>=300, n_iters>=800, 5 seeds (see README TODO).
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


def spots():
    g = torch.zeros(N_X); torch.manual_seed(7)
    for _ in range(5):
        c = torch.randint(N_X//8, 7*N_X//8, (1,)).item()
        w = torch.randint(6, 24, (1,)).item()
        g[max(0,c-w):min(N_X,c+w)] = torch.rand(1).item() + 0.3
    return g


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
        out.append(dict(seed=s,
            ours=psnr(ro, tgt), blind=psnr(rb, tgt), gs=psnr(rg, tgt),
            oracle=psnr(rc, tgt),
            de_ours=diffraction_efficiency(ro, mask),
            de_blind=diffraction_efficiency(rb, mask)))
    return out


def main(part="all"):
    t0 = time.time()
    R = json.load(open("results_prelim.json")) if os.path.exists("results_prelim.json") else {}

    if part in ("all", "A"):
        p = MediumParams()
        rec_probe = NPDDRecorder(N_X, DX, params=p)
        R["predicted_cliff_K"] = {f"budget{b}": rec_probe.predicted_cliff(budget=b)
                                  for b in [2.0, 4.0, 8.0]}
        R["cliff"] = {}
        for period in [8, 12, 16, 24, 32, 48, 64]:
            K = 2 * math.pi / (period * DX)
            R["cliff"][f"period{period}"] = dict(K=K, rows=run_cell(p, bars(period)))
            print(f"[A] period={period}px K={K:.2f}  {time.time()-t0:.0f}s", flush=True)

    if part in ("all", "B"):
        R["sigma"] = {}
        for sg in [0.02, 0.08, 0.20, 0.30]:
            p = MediumParams(); p.sigma = sg
            R["sigma"][f"{sg}"] = dict(bars=run_cell(p, bars(16)), spots=run_cell(p, spots()))
            print(f"[B] sigma={sg}  {time.time()-t0:.0f}s", flush=True)

    if part in ("all", "C"):
        R["dn_max"] = {}
        for dn in [1e-3, 3.5e-3, 6e-3]:
            p = MediumParams(); p.dn_max = dn
            R["dn_max"][f"{dn}"] = dict(bars=run_cell(p, bars(16)))
            print(f"[C] dn_max={dn}  {time.time()-t0:.0f}s", flush=True)

    with open("results_prelim.json", "w") as f:
        json.dump(R, f, indent=1)
    print(f"DONE part={part} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import sys as _s
    main(_s.argv[1] if len(_s.argv) > 1 else "all")
