"""
run_experiments.py
------------------
Main experiment runner for Project 2: WaveOptics Degradation Experiments.

Runs all degradation sweeps and saves:
  - Individual reconstruction images
  - Side-by-side comparison figure per sweep
  - metrics_summary.csv  — all PSNR / SSIM / LPIPS-proxy values

Usage
-----
    python run_experiments.py

All outputs go to  ./results/

Sweeps
------
  1. Resolution     : 512 → 256 → 128 → 64 → 32
  2. Phase bits     : 8 → 4 → 2 → 1
  3. Color channels : RGB → RG → mono   (uses a colour scene)
  4. Viewing angle  : 100% → 75% → 50% → 25% → 10%
  5. Depth planes   : 4 → 3 → 2 → 1
  6. Speckle noise  : 0 → 0.05 → 0.1 → 0.2 → 0.4
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Project modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from core.waveoptics  import gerchberg_saxton, reconstruct
from core.degradation import (
    degrade_resolution,
    quantise_phase,
    degrade_color,
    limit_viewing_angle,
    add_speckle,
    depth_planes_to_z_list,
)
from core.metrics  import all_metrics
from core.scenes   import gaussian_spots, resolution_chart, letters, multi_depth_scene

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
SIZE         = 256          # Keep at 256 for fast iteration; crank to 512 later
WAVELENGTH   = 532e-9       # green laser [m]
DX           = 8e-6         # SLM pixel pitch [m]
Z            = 0.15         # reconstruction distance [m]
GS_ITER      = 30           # Gerchberg-Saxton iterations

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison(
    images: list,          # list of (label, float32_array)
    filename: str,
    title: str,
    ref_idx: int = 0,
    metrics_rows: list = None,
):
    """Save a side-by-side comparison figure."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, (label, img) in zip(axes, images):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Metric subtitle below each image
    if metrics_rows:
        for ax, row in zip(axes[1:], metrics_rows):
            txt = f"PSNR={row['psnr']:.1f}  SSIM={row['ssim']:.3f}"
            ax.set_xlabel(txt, fontsize=7.5, color="#444")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")
    return path


def compute_and_log(ref, deg, label, sweep_name, rows):
    m = all_metrics(ref, deg)
    m["label"]  = label
    m["sweep"]  = sweep_name
    rows.append(m)
    print(f"    {label:35s}  PSNR={m['psnr']:5.1f}  SSIM={m['ssim']:.3f}  LPIPS≈{m['lpips_proxy']:.5f}")
    return m


# ─────────────────────────────────────────────────────────────────────────────
#  Build the reference hologram once
# ─────────────────────────────────────────────────────────────────────────────

def build_reference():
    print("Building reference hologram …")
    target = gaussian_spots(SIZE, n_spots=4, sigma=SIZE * 0.025, seed=42)
    t0 = time.time()
    phase = gerchberg_saxton(target, n_iter=GS_ITER, wavelength=WAVELENGTH, dx=DX, z=Z)
    print(f"  GS phase retrieval done in {time.time()-t0:.1f}s  ({SIZE}×{SIZE}, {GS_ITER} iter)")
    ref_recon = reconstruct(phase, wavelength=WAVELENGTH, dx=DX, z=Z)
    return target, phase, ref_recon


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 1 — Resolution
# ─────────────────────────────────────────────────────────────────────────────

def sweep_resolution(phase, ref_recon, all_rows):
    print("\n── Sweep 1: Resolution ──")
    resolutions = [SIZE, SIZE // 2, SIZE // 4, SIZE // 8, SIZE // 16]
    resolutions = [r for r in resolutions if r >= 8]

    images = [("Reference\n(full res)", ref_recon)]
    metric_rows = []

    for res in resolutions[1:]:
        deg_phase = degrade_resolution(phase, res)
        deg_recon = reconstruct(deg_phase, wavelength=WAVELENGTH, dx=DX, z=Z)
        label = f"Resolution\n{res}×{res}"
        images.append((label, deg_recon))
        m = compute_and_log(ref_recon, deg_recon, f"{res}×{res}", "resolution", all_rows)
        metric_rows.append(m)

        # Save individual image
        np.save(os.path.join(RESULTS_DIR, f"recon_res_{res}.npy"), deg_recon)

    save_comparison(images, "sweep_resolution.png",
                    "Sweep 1 — Resolution Degradation", metrics_rows=metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 2 — Phase quantisation
# ─────────────────────────────────────────────────────────────────────────────

def sweep_phase_bits(phase, ref_recon, all_rows):
    print("\n── Sweep 2: Phase Quantisation ──")
    bit_levels = [8, 4, 2, 1]
    images = [("Reference\n(continuous)", ref_recon)]
    metric_rows = []

    for bits in bit_levels:
        q_phase = quantise_phase(phase, bits=bits)
        q_recon = reconstruct(q_phase, wavelength=WAVELENGTH, dx=DX, z=Z)
        label = f"{bits}-bit phase\n({2**bits} levels)"
        images.append((label, q_recon))
        m = compute_and_log(ref_recon, q_recon, f"{bits}-bit", "phase_bits", all_rows)
        metric_rows.append(m)

    save_comparison(images, "sweep_phase_bits.png",
                    "Sweep 2 — Phase Quantisation", metrics_rows=metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 3 — Color channels (grayscale reconstruction comparison)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_color(phase, ref_recon, all_rows):
    """
    Simulate RGB vs RG vs mono by using the same hologram but
    zeroing wavelength contributions — approximated here by
    degrade_color on the *reconstruction visualised as colour*.
    """
    print("\n── Sweep 3: Colour Channels ──")
    # Represent reconstruction in colour (stack to 3-ch)
    ref_rgb = np.stack([ref_recon, ref_recon * 0.9, ref_recon * 0.6], axis=2)
    ref_rgb = np.clip(ref_rgb, 0, 1)

    modes   = ["RGB", "RG", "mono"]
    images  = []
    metric_rows = []

    for mode in modes:
        deg = degrade_color((ref_rgb * 255).astype(np.uint8), mode).astype(np.float32) / 255.0
        images.append((mode, deg))
        if mode != "RGB":
            m = compute_and_log(ref_rgb, deg, mode, "color", all_rows)
            metric_rows.append(m)

    save_comparison(images, "sweep_color.png",
                    "Sweep 3 — Colour Channel Degradation", ref_idx=0,
                    metrics_rows=metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 4 — Viewing angle (bandwidth)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_viewing_angle(phase, ref_recon, all_rows):
    print("\n── Sweep 4: Viewing Angle ──")
    fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
    images = []
    metric_rows = []

    for frac in fractions:
        lim_phase = limit_viewing_angle(phase, bandwidth_fraction=frac)
        lim_recon = reconstruct(lim_phase, wavelength=WAVELENGTH, dx=DX, z=Z)
        pct = int(frac * 100)
        label = f"BW {pct}%\n(~{pct}° rel.)"
        images.append((label, lim_recon))
        if frac < 1.0:
            m = compute_and_log(ref_recon, lim_recon, f"BW {pct}%", "viewing_angle", all_rows)
            metric_rows.append(m)

    save_comparison(images, "sweep_viewing_angle.png",
                    "Sweep 4 — Viewing Angle (Bandwidth) Reduction",
                    metrics_rows=metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 5 — Depth planes
# ─────────────────────────────────────────────────────────────────────────────

def sweep_depth_planes(all_rows):
    print("\n── Sweep 5: Depth Planes ──")
    plane_counts = [4, 3, 2, 1]
    images = []
    ref_recon_multi = None
    metric_rows = []

    for n_planes in plane_counts:
        layers = multi_depth_scene(SIZE, n_planes=n_planes)
        z_list = depth_planes_to_z_list(n_planes, z_near=0.08, z_far=0.25)

        # Superpose holograms from each depth layer
        combined_phase = np.zeros((SIZE, SIZE), dtype=np.float32)
        for layer, z_layer in zip(layers, z_list):
            layer_phase = gerchberg_saxton(layer, n_iter=GS_ITER,
                                           wavelength=WAVELENGTH, dx=DX, z=z_layer)
            combined_phase += layer_phase
        combined_phase = np.angle(np.exp(1j * combined_phase))  # wrap to [-π,π]

        # Reconstruct at middle depth
        z_mid = (z_list[0] + z_list[-1]) / 2
        recon = reconstruct(combined_phase, wavelength=WAVELENGTH, dx=DX, z=z_mid)
        label = f"{n_planes} depth plane{'s' if n_planes>1 else ''}"
        images.append((label, recon))

        if ref_recon_multi is None:
            ref_recon_multi = recon
        else:
            m = compute_and_log(ref_recon_multi, recon, f"{n_planes} planes", "depth_planes", all_rows)
            metric_rows.append(m)

    save_comparison(images, "sweep_depth_planes.png",
                    "Sweep 5 — Number of Depth Planes",
                    metrics_rows=metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 6 — Speckle noise
# ─────────────────────────────────────────────────────────────────────────────

def sweep_speckle(ref_recon, all_rows):
    print("\n── Sweep 6: Speckle Noise ──")
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.4]
    images = []
    metric_rows = []

    for sigma in sigmas:
        if sigma == 0.0:
            noisy = ref_recon.copy()
        else:
            noisy = add_speckle(ref_recon, sigma=sigma)
        label = f"σ = {sigma:.2f}"
        images.append((label, noisy))
        if sigma > 0.0:
            m = compute_and_log(ref_recon, noisy, f"σ={sigma}", "speckle", all_rows)
            metric_rows.append(m)

    save_comparison(images, "sweep_speckle.png",
                    "Sweep 6 — Speckle Noise", metrics_rows=metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Summary plot — metrics vs degradation level
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_summary(all_rows):
    sweeps = {}
    for row in all_rows:
        sweeps.setdefault(row["sweep"], []).append(row)

    fig, axes = plt.subplots(len(sweeps), 2, figsize=(12, 4 * len(sweeps)))
    if len(sweeps) == 1:
        axes = [axes]

    for ax_row, (sweep_name, rows) in zip(axes, sweeps.items()):
        labels = [r["label"] for r in rows]
        psnr_v = [r["psnr"]  for r in rows]
        ssim_v = [r["ssim"]  for r in rows]

        xs = range(len(labels))
        ax_row[0].plot(xs, psnr_v, "o-", color="#2563eb", linewidth=2)
        ax_row[0].set_xticks(xs); ax_row[0].set_xticklabels(labels, fontsize=8)
        ax_row[0].set_ylabel("PSNR (dB)")
        ax_row[0].set_title(f"{sweep_name} — PSNR", fontweight="bold")
        ax_row[0].grid(True, alpha=0.3)

        ax_row[1].plot(xs, ssim_v, "s-", color="#16a34a", linewidth=2)
        ax_row[1].set_xticks(xs); ax_row[1].set_xticklabels(labels, fontsize=8)
        ax_row[1].set_ylabel("SSIM")
        ax_row[1].set_ylim(0, 1.05)
        ax_row[1].set_title(f"{sweep_name} — SSIM", fontweight="bold")
        ax_row[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "metrics_summary_plot.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Metrics plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(all_rows):
    path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    fieldnames = ["sweep", "label", "psnr", "ssim", "mse", "lpips_proxy"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  CSV  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    all_rows = []

    target, phase, ref_recon = build_reference()

    # Save reference
    np.save(os.path.join(RESULTS_DIR, "reference_target.npy"),    target)
    np.save(os.path.join(RESULTS_DIR, "reference_phase.npy"),     phase)
    np.save(os.path.join(RESULTS_DIR, "reference_recon.npy"),     ref_recon)

    sweep_resolution(phase, ref_recon, all_rows)
    sweep_phase_bits(phase, ref_recon, all_rows)
    sweep_color(phase, ref_recon, all_rows)
    sweep_viewing_angle(phase, ref_recon, all_rows)
    sweep_depth_planes(all_rows)
    sweep_speckle(ref_recon, all_rows)

    plot_metrics_summary(all_rows)
    export_csv(all_rows)

    print(f"\n✓ All done in {time.time()-t_start:.1f}s — results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
