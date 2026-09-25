"""
gs_target_fidelity.py
---------------------
Context for the paper's reconstruction-referenced metrics.

Every degradation metric in run_experiments.py compares a degraded
reconstruction with the UNDEGRADED GS reconstruction of the same scene, so it
measures the loss caused by the degradation alone and excludes GS's own error.
This script reports that excluded quantity: how close the undegraded GS
reconstruction (50 iterations, seed 42) is to each target, comparing the
normalised reconstructed intensity with the normalised target intensity
(target amplitude squared).

Output: results/gs_vs_target.csv
Usage:  python gs_target_fidelity.py
"""
import csv
import os

import numpy as np

from run_experiments import (SCENE_SUITE, SIZE, GS_ITER, WAVELENGTH, DX, Z,
                             RESULTS_DIR, reconstruct_phase)
from core.waveoptics import gerchberg_saxton
from core.metrics import psnr, ssim, lpips_real


def main():
    rows = []
    for name, fn in SCENE_SUITE.items():
        target = fn(SIZE).astype(np.float32)
        phase = gerchberg_saxton(target, n_iter=GS_ITER, wavelength=WAVELENGTH, dx=DX, z=Z)
        recon = reconstruct_phase(phase)
        t_int = target.astype(np.float64) ** 2
        t_int = (t_int / (t_int.max() + 1e-12)).astype(np.float32)
        rows.append(dict(scene=name, psnr=psnr(t_int, recon), ssim=ssim(t_int, recon),
                         lpips=lpips_real(t_int, recon)))
        print(f"{name:18s} PSNR={rows[-1]['psnr']:.2f}  SSIM={rows[-1]['ssim']:.3f}  "
              f"LPIPS={rows[-1]['lpips']:.3f}")
    path = os.path.join(RESULTS_DIR, "gs_vs_target.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
