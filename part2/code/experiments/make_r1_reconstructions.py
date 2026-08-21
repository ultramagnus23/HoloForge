"""
R1 -- reconstruction figure (work-spec item D.1): target vs. media-blind
SGD vs. media-in-the-loop reconstructions, plus an error profile, at the
same three representative K points M2 uses (sub/near/post-cliff:
1.31, 3.93, 5.24 rad/um), budget=2x, seed=0.

Target is a 1D transverse bars pattern (this paper's recording kinetics
are 1D in x, see the Discussion) -- so these are 1D profile overlays, not
2D images, and the figure/caption say so plainly rather than implying a
2D reconstruction that doesn't exist in this model.

Cost estimate (checked before launching the full run): a single BSGD job
at this n_x=1024/n_iters=800 configuration took ~75s on this machine's
RTX 3050. MIL costs more per-iteration (full NPDD forward pass vs. one
linear multiply) but this script runs only 6 jobs total (BSGD+MIL x 3 K),
not a sweep -- bounded, small cost, run in the foreground.

Writes results_r1_reconstructions.json (target, BSGD recon, MIL recon,
PSNR for each, per K) for figures/make_all.py's make_R1 to render.

Manifest note: this reuses S1_K_POINTS, DEFAULT_MEDIUM, and budget=2x --
all already-registered manifest constants from experiments/manifest.py,
not a new sweep with its own K grid/budget/condition choices. Seed is
fixed at 0 and stated above. Not added as its own build_R1_jobs() entry
in manifest.py because it is a single deterministic illustrative render
(one seed, no statistics computed from it, like F1's schematic) reusing
an already-manifested condition, not a new experimental design that
needs its own seed/grid/budget recorded.
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
from holomedia.optimize import psnr_si
from methods import media_blind_sgd, media_in_the_loop
from manifest import DEFAULT_MEDIUM, period_from_K, S1_K_POINTS

torch.set_default_dtype(torch.float32)

N_X, DX = 1024, 0.05
LAM_UM = 0.405
N_ITERS = 800
BUDGET = 2.0
SEED = 0


def build_bars_target(period_px: int, n_x: int, device) -> torch.Tensor:
    x = torch.arange(n_x, device=device)
    return ((x // (period_px // 2)) % 2).float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    medium = MediumParams(**DEFAULT_MEDIUM)
    rec = NPDDRecorder(N_X, DX, t_total=10.0, n_steps=300, params=medium, dtype=torch.float32).to(device)
    bpm = SlabBPM(N_X, DX, LAM_UM, medium.thickness, n_z=32, n0=medium.n0, dtype=torch.complex64).to(device)

    results = []
    for K in S1_K_POINTS:  # sub/near/post-cliff, matches M2's shared points
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

        psnr_bsgd = float(psnr_si(recon_bsgd, target))
        psnr_mil = float(psnr_si(recon_mil, target))
        print(f"K={K:.3f}  BSGD PSNR={psnr_bsgd:.2f}dB ({t_bsgd:.0f}s)  "
              f"MIL PSNR={psnr_mil:.2f}dB ({t_mil:.0f}s)  gain={psnr_mil-psnr_bsgd:.2f}dB")

        results.append(dict(
            K=K, period_px=period_px, budget=BUDGET, seed=SEED,
            target=target.detach().cpu().tolist(),
            recon_bsgd=recon_bsgd.detach().cpu().tolist(),
            recon_mil=recon_mil.detach().cpu().tolist(),
            psnr_bsgd=psnr_bsgd, psnr_mil=psnr_mil,
        ))

    # zero-row / silent-success check (ground rule): fail loud if nothing came out
    if not results or any(len(r["target"]) == 0 for r in results):
        raise RuntimeError("make_r1_reconstructions produced empty output -- failing "
                          "loud per the zero-row ground rule, not writing a hollow JSON.")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results_r1_reconstructions.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(dict(results=results, n_x=N_X, dx=DX, n_iters=N_ITERS), f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
