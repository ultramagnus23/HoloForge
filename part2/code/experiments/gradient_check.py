"""
Finite-difference check of the MIL gradient at the paper's configuration.

The gradient that drives media-in-the-loop design is reverse-mode automatic
differentiation through the full chain softplus -> exposure projection ->
NPDD recording (300 IMEX steps) -> split-step BPM (128 slices) -> scale-
invariant loss. This script compares its directional derivative along random
unit directions with a central finite difference of the same loss, in float64
on CPU, at the frozen geometry (n_x = 1024, dx = 0.05 um, unslanted, default
medium, budget 2x, bar target at K = 3.93 rad/um). The exposure is kept away
from the contrast cap so that the projection is smooth at the test point.

Usage (from part2/code): python experiments/gradient_check.py   (CPU, ~1 min)
Output: results_gradient_check.json (read by scripts/make_numbers_tex.py)
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import torch

from holomedia import NPDDRecorder, MediumParams, SlabBPM
from holomedia.optimize import contrast_project, si_mse
from manifest import DEFAULT_MEDIUM
from run_manifest import build_target

N_X, DX, LAM, BUDGET = 1024, 0.05, 0.405, 2.0
K = 3.9269908169872414
N_DIRS = 6
STEP = 1e-5
OUT = os.path.join(os.path.dirname(__file__), "..", "results_gradient_check.json")


def main():
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64
    medium = MediumParams(**DEFAULT_MEDIUM)
    rec = NPDDRecorder(N_X, DX, t_total=10.0, n_steps=300, params=medium, dtype=dtype)
    bpm = SlabBPM(N_X, DX, LAM, medium.thickness, n_z=128, n0=medium.n0,
                  dtype=torch.complex128, slant_deg=0.0)
    period_px = int(round(2 * math.pi / K / DX))
    target = build_target(dict(kind="bars", period_px=period_px), N_X, torch.device("cpu"),
                          dtype=dtype)

    def loss_fn(theta):
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6, 1.0, BUDGET)
        return si_mse(bpm(rec(E), shrinkage=medium.shrinkage), target)

    gen = torch.Generator().manual_seed(0)
    theta = (0.3 * torch.randn(N_X, generator=gen, dtype=dtype)).requires_grad_(True)
    with torch.no_grad():
        E0 = contrast_project(torch.nn.functional.softplus(theta) + 1e-6, 1.0, BUDGET)
        peak_to_mean = float(E0.max() / E0.mean())
    loss = loss_fn(theta)
    (grad,) = torch.autograd.grad(loss, theta)

    rows = []
    with torch.no_grad():
        for i in range(N_DIRS):
            v = torch.randn(N_X, generator=gen, dtype=dtype)
            v /= v.norm()
            fd = float((loss_fn(theta + STEP * v) - loss_fn(theta - STEP * v)) / (2 * STEP))
            ad = float(grad @ v)
            rows.append(dict(direction=i, autodiff=ad, finite_difference=fd,
                             rel_error=abs(ad - fd) / max(abs(fd), 1e-300)))
            print(f"dir {i}: AD {ad:+.6e}  FD {fd:+.6e}  rel err {rows[-1]['rel_error']:.2e}")

    out = dict(n_x=N_X, dx=DX, K=K, budget=BUDGET, n_steps=300, n_z=128, step=STEP,
               dtype="float64", device="cpu", peak_to_mean=peak_to_mean,
               loss=float(loss), directions=rows,
               max_rel_error=max(r["rel_error"] for r in rows))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"max relative error {out['max_rel_error']:.2e} over {N_DIRS} directions "
          f"(peak/mean exposure {peak_to_mean:.2f}, cap {BUDGET}) -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
