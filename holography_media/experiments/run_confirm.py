"""
Confirmation-scale rerun (CPU, this environment has no GPU).

Honest scope note: the README/paper TODO asks for grid >=1024, iters >=800,
5 seeds across the FULL f2_f3_recovery.py combinatorial sweep (26 settings x
4 targets x 5 seeds). Benchmarking in this environment (see PR discussion)
gives ~180s per (target, seed) at n_x=1024/n_iters=800 for the
media-in-the-loop method alone; the full combinatorial sweep at that scale is
~26 hours of CPU time -- not practical in one interactive session, and this
box has no GPU (torch.cuda.is_available() == False).

This script instead reruns the SAME cells reported in the paper's
"PRELIMINARY RESULTS (v0.1)" block -- the cliff sweep (7 K values), the
dynamic-range sweep (3 dn_max values), and the non-locality sweep (4 sigma
values, now also at high-K per the paper's own flagged follow-up) -- at a
"confirmation scale" roughly 2x grid, 3x seeds, and 3-5x iterations above the
v0.1 numbers: n_x=512 (was 256), n_iters=500 (was 150), n_steps=250 (was 100),
seeds=[0,1,2] (was [0,1]). This is a genuine increase in fidelity, not a
relabeled rerun of the same numbers, but it is NOT the full 1024/800/5-seed
paper-scale run: see README.md "GPU rerun" section for the exact command to
finish the job on a GPU box.

Estimated wall-clock at this scale (measured on this CPU): ~2.5 hours total.
"""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
                       media_in_the_loop, media_blind_sgd, media_blind_gs,
                       oracle_ideal, psnr, diffraction_efficiency)

torch.set_default_dtype(torch.float64)
N_X, DX, LAM = 512, 0.1, 0.405
N_ITERS, N_STEPS, N_Z, SEEDS = 500, 250, 20, [0, 1, 2]
OUT = os.path.join(os.path.dirname(__file__), "..", "results_confirm.json")


def bars(period_px):
    x = torch.arange(N_X)
    return ((x // (period_px // 2)) % 2).double()


def spots():
    g = torch.zeros(N_X); torch.manual_seed(7)
    for _ in range(5):
        c = torch.randint(N_X // 8, 7 * N_X // 8, (1,)).item()
        w = torch.randint(12, 48, (1,)).item()
        g[max(0, c - w):min(N_X, c + w)] = torch.rand(1).item() + 0.3
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
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    R["meta"] = dict(n_x=N_X, dx=DX, n_iters=N_ITERS, n_steps=N_STEPS, seeds=SEEDS,
                      scale="confirmation (not full paper-scale; see docstring)")

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
            json.dump(R, open(OUT, "w"), indent=1)

    if part in ("all", "B"):
        R["sigma"] = {}
        for sg in [0.02, 0.08, 0.20, 0.30]:
            p = MediumParams(); p.sigma = sg
            R["sigma"][f"{sg}"] = dict(bars=run_cell(p, bars(16)), spots=run_cell(p, spots()))
            print(f"[B] sigma={sg}  {time.time()-t0:.0f}s", flush=True)
            json.dump(R, open(OUT, "w"), indent=1)

    if part in ("all", "B2"):
        # high-K sigma crossover probe (period 8px), per paper's own flagged follow-up
        R["sigma_highK"] = {}
        for sg in [0.02, 0.08, 0.20, 0.30]:
            p = MediumParams(); p.sigma = sg
            R["sigma_highK"][f"{sg}"] = run_cell(p, bars(8))
            print(f"[B2] sigma={sg} @highK  {time.time()-t0:.0f}s", flush=True)
            json.dump(R, open(OUT, "w"), indent=1)

    if part in ("all", "C"):
        R["dn_max"] = {}
        for dn in [1e-3, 3.5e-3, 6e-3]:
            p = MediumParams(); p.dn_max = dn
            R["dn_max"][f"{dn}"] = dict(bars=run_cell(p, bars(16)))
            print(f"[C] dn_max={dn}  {time.time()-t0:.0f}s", flush=True)
            json.dump(R, open(OUT, "w"), indent=1)

    json.dump(R, open(OUT, "w"), indent=1)
    print(f"DONE part={part} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
