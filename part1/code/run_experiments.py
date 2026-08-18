"""
run_experiments.py
------------------
Main experiment runner for HoloForge — systematic degradation analysis in
phase-only computational holography (Part 1).

Pipeline
--------
A target amplitude is encoded into a phase-only hologram with the
Gerchberg-Saxton (GS) algorithm (Angular Spectrum propagation). The hologram
phase is then degraded along four physically independent axes, reconstructed,
and scored against the undegraded reconstruction with PSNR / SSIM / LPIPS.

Degradation axes (Part 1)
-------------------------
  D1. Spatial resolution      — degrade_resolution_phase()  (complex domain)
  D2. Phase quantisation      — quantise_phase()
  D3. Viewing-angle bandwidth — limit_viewing_angle()
  D4. Coherent speckle        — add_speckle_physical()  (pre-reconstruction
                                phase noise)

Multi-wavelength (RGB) colour holography is explicitly OUT OF SCOPE for Part 1
(see sweep_color / results/sweep_color_EXCLUDED.txt). The legacy depth-plane
sweep is retained as a coherent-superposition demonstration.

Outputs (all under ./results/)
------------------------------
  Per-sweep comparison figures, GS convergence curve, the main metrics CSV,
  multi-scene generalisation CSVs + figure, and seed-sensitivity CSVs.

Usage
-----
    python run_experiments.py
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from core.waveoptics  import gerchberg_saxton, reconstruct
from core.degradation import (
    quantise_phase,
    degrade_resolution_phase,
    limit_viewing_angle,
    add_speckle_physical,
    depth_planes_to_z_list,
)
from core.metrics  import all_metrics, ssim
from core.scenes   import (
    gaussian_spots, resolution_chart, letters, circle_ring, natural_photo,
    multi_depth_scene,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
SIZE         = 256          # working resolution (256×256)
WAVELENGTH   = 532e-9       # green laser [m]
DX           = 8e-6         # SLM pixel pitch [m]
Z            = 0.15         # reconstruction distance [m] = 150 mm
GS_ITER      = 50           # Gerchberg-Saxton iterations (research-grade)

# Degradation parameter grids (shared by individual + aggregate analyses)
RES_LEVELS   = [SIZE // 2, SIZE // 4, SIZE // 8, SIZE // 16]   # 128,64,32,16
BIT_LEVELS   = [8, 4, 2, 1]
BW_FRACTIONS = [0.75, 0.5, 0.25, 0.1]
SPECKLE_SIGMAS = [0.0, 0.1, 0.3, 0.6, 1.0, np.pi]
SEEDS        = [42, 7, 123, 999, 17]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison(images, filename, title, metrics_rows=None):
    """Save a side-by-side comparison figure. images = list of (label, array)."""
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


def log_row(metrics, label, sweep_name, rows):
    """Attach label/sweep to a precomputed metrics dict, store, and print it."""
    m = dict(metrics)
    m["label"] = label
    m["sweep"] = sweep_name
    rows.append(m)
    print(f"    {label:18s}  PSNR={m['psnr']:5.1f}  SSIM={m['ssim']:.3f}  "
          f"LPIPS={m['lpips']:.4f}  (proxy={m['lpips_proxy']:.5f})")
    return m


def reconstruct_phase(phase):
    return reconstruct(phase, wavelength=WAVELENGTH, dx=DX, z=Z)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-axis degradation (single source of truth used by every analysis)
# ─────────────────────────────────────────────────────────────────────────────

def degrade_axis(axis, phase, ref_recon):
    """
    Apply one degradation axis to `phase`, reconstruct each variant, and score
    it against `ref_recon`.

    Returns a list of (param_label, metrics_dict, recon_image).
    """
    out = []
    if axis == "resolution":
        for res in RES_LEVELS:
            if res < 8:
                continue
            deg = degrade_resolution_phase(phase, res)
            rec = reconstruct_phase(deg)
            out.append((f"{res}x{res}", all_metrics(ref_recon, rec), rec))
    elif axis == "phase_bits":
        for bits in BIT_LEVELS:
            deg = quantise_phase(phase, bits=bits)
            rec = reconstruct_phase(deg)
            out.append((f"{bits}-bit", all_metrics(ref_recon, rec), rec))
    elif axis == "viewing_angle":
        for frac in BW_FRACTIONS:
            deg = limit_viewing_angle(phase, bandwidth_fraction=frac)
            rec = reconstruct_phase(deg)
            out.append((f"{int(frac*100)}%", all_metrics(ref_recon, rec), rec))
    else:
        raise ValueError(f"Unknown axis '{axis}'")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Reference hologram
# ─────────────────────────────────────────────────────────────────────────────

def build_reference():
    print("Building reference hologram …")
    target = gaussian_spots(SIZE, n_spots=4, sigma=SIZE * 0.025, seed=42)
    t0 = time.time()
    phase = gerchberg_saxton(target, n_iter=GS_ITER, wavelength=WAVELENGTH, dx=DX, z=Z)
    print(f"  GS phase retrieval done in {time.time()-t0:.1f}s  "
          f"({SIZE}×{SIZE}, {GS_ITER} iter)")
    ref_recon = reconstruct_phase(phase)
    return target, phase, ref_recon


# ─────────────────────────────────────────────────────────────────────────────
#  GS convergence (Task 6)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gs_convergence():
    print("\n── GS Convergence ──")
    target = gaussian_spots(SIZE, n_spots=4, sigma=SIZE * 0.025, seed=42)
    _, hist = gerchberg_saxton(
        target, n_iter=100, wavelength=WAVELENGTH, dx=DX, z=Z, return_history=True
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(hist) + 1), hist, "-", color="#2563eb", linewidth=2)
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("Reconstruction MSE (normalised amplitude)")
    ax.set_title("Gerchberg–Saxton Convergence (256×256)", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    for n_iter_mark in (30, 50):
        ax.axvline(n_iter_mark, color="#999", linestyle="--", linewidth=1)
        ax.text(n_iter_mark, max(hist), f"{n_iter_mark}", fontsize=7,
                color="#666", ha="center", va="bottom")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "gs_convergence.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  MSE: iter1={hist[0]:.5f} → iter30={hist[29]:.5f} → "
          f"iter50={hist[49]:.5f} → iter100={hist[-1]:.5f}")
    print(f"  saved → {path}")
    np.save(os.path.join(RESULTS_DIR, "gs_convergence.npy"), np.array(hist))


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 1 — Resolution
# ─────────────────────────────────────────────────────────────────────────────

def sweep_resolution(phase, ref_recon, all_rows):
    print("\n── Sweep 1: Resolution (complex-domain phase downsampling) ──")
    images = [("Reference\n(full res)", ref_recon)]
    metric_rows = []
    for label, m, rec in degrade_axis("resolution", phase, ref_recon):
        images.append((f"Resolution\n{label}", rec))
        metric_rows.append(log_row(m, label, "resolution", all_rows))
        res = int(label.split("x")[0])
        np.save(os.path.join(RESULTS_DIR, f"recon_res_{res}.npy"), rec)
    save_comparison(images, "sweep_resolution.png",
                    "Sweep 1 — Spatial Resolution Degradation", metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 2 — Phase quantisation
# ─────────────────────────────────────────────────────────────────────────────

def sweep_phase_bits(phase, ref_recon, all_rows):
    print("\n── Sweep 2: Phase Quantisation ──")
    images = [("Reference\n(continuous)", ref_recon)]
    metric_rows = []
    for label, m, rec in degrade_axis("phase_bits", phase, ref_recon):
        bits = int(label.split("-")[0])
        images.append((f"{bits}-bit phase\n({2**bits} levels)", rec))
        metric_rows.append(log_row(m, label, "phase_bits", all_rows))
    save_comparison(images, "sweep_phase_bits.png",
                    "Sweep 2 — Phase Quantisation", metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 3 — Colour channels (EXCLUDED from Part 1)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_color(phase, ref_recon, all_rows):
    """
    COLOUR SWEEP — EXCLUDED FROM PART 1.

    Real multi-wavelength (RGB) holography requires three separate phase
    holograms computed at lambda_R=638nm, lambda_G=532nm, lambda_B=450nm,
    combined on an RGB SLM or via time-multiplexing. The previous
    implementation stacked a monochrome reconstruction as a fake RGB image
    — this does not model any holographic colour physics.

    Multi-wavelength holographic colour is deferred to Part 2. The present
    paper explicitly limits scope to single-wavelength (532 nm).
    """
    print("\n── Sweep 3: Colour Channels — EXCLUDED "
          "(see results/sweep_color_EXCLUDED.txt) ──")
    note = (
        "COLOR CHANNELS SWEEP EXCLUDED FROM PART 1\n"
        "==========================================\n"
        "Multi-wavelength RGB holography requires separate phase retrieval\n"
        "at each primary wavelength (638nm / 532nm / 450nm). This is deferred\n"
        "to Part 2. The present simulation is single-wavelength (532nm green).\n"
        "\nThis exclusion is explicitly documented in the paper's scope section.\n"
    )
    with open(os.path.join(RESULTS_DIR, "sweep_color_EXCLUDED.txt"), "w") as f:
        f.write(note)
    print(f"  {note.splitlines()[0]}")


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 4 — Viewing angle (bandwidth)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_viewing_angle(phase, ref_recon, all_rows):
    print("\n── Sweep 4: Viewing Angle (Bandwidth) ──")
    images = [("BW 100%\n(reference)", ref_recon)]
    metric_rows = []
    for label, m, rec in degrade_axis("viewing_angle", phase, ref_recon):
        images.append((f"BW {label}", rec))
        metric_rows.append(log_row(m, label, "viewing_angle", all_rows))
    save_comparison(images, "sweep_viewing_angle.png",
                    "Sweep 4 — Viewing Angle (Bandwidth) Reduction", metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 5 — Depth planes (coherent complex-field superposition)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_depth_planes(all_rows):
    print("\n── Sweep 5: Depth Planes (coherent field superposition) ──")
    plane_counts = [4, 3, 2, 1]
    images = []
    ref_recon_multi = None
    metric_rows = []

    for n_planes in plane_counts:
        layers = multi_depth_scene(SIZE, n_planes=n_planes)
        z_list = depth_planes_to_z_list(n_planes, z_near=0.08, z_far=0.25)

        # Coherent superposition: sum complex fields, then extract phase.
        combined_field = np.zeros((SIZE, SIZE), dtype=np.complex64)
        for layer, z_layer in zip(layers, z_list):
            layer_phase = gerchberg_saxton(
                layer, n_iter=GS_ITER, wavelength=WAVELENGTH, dx=DX, z=z_layer
            )
            combined_field += np.exp(1j * layer_phase).astype(np.complex64)
        combined_phase = np.angle(combined_field).astype(np.float32)

        z_mid = (z_list[0] + z_list[-1]) / 2
        recon = reconstruct(combined_phase, wavelength=WAVELENGTH, dx=DX, z=z_mid)
        label = f"{n_planes} depth plane{'s' if n_planes > 1 else ''}"
        images.append((label, recon))

        if ref_recon_multi is None:
            ref_recon_multi = recon
        else:
            m = all_metrics(ref_recon_multi, recon)
            metric_rows.append(log_row(m, f"{n_planes} planes", "depth_planes", all_rows))

    save_comparison(images, "sweep_depth_planes.png",
                    "Sweep 5 — Number of Depth Planes (coherent superposition)",
                    metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep 6 — Coherent speckle (physical, pre-reconstruction phase noise)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_speckle(phase, ref_recon, all_rows):
    print("\n── Sweep 6: Coherent Speckle (pre-reconstruction phase noise) ──")
    images = [("σ = 0.0\n(reference)", ref_recon)]
    metric_rows = []
    for sigma in SPECKLE_SIGMAS:
        if sigma == 0.0:
            continue
        noisy_phase = add_speckle_physical(phase, sigma_rad=sigma, seed=0)
        noisy = reconstruct_phase(noisy_phase)
        label = f"σ={sigma:.2f}" if sigma != np.pi else "σ=π"
        images.append((f"{label} rad", noisy))
        metric_rows.append(log_row(all_metrics(ref_recon, noisy), label,
                                    "speckle", all_rows))
    save_comparison(images, "sweep_speckle.png",
                    "Sweep 6 — Coherent Speckle (SLM phase noise)", metric_rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-scene validation (Task 7)
# ─────────────────────────────────────────────────────────────────────────────

SCENE_SUITE = {
    "gaussian_spots":   lambda s: gaussian_spots(s, n_spots=4, sigma=s * 0.025, seed=42),
    "resolution_chart": resolution_chart,
    "letters":          letters,
    "circle_ring":      circle_ring,
    "natural_photo":    natural_photo,
}

MULTI_SCENE_AXES = ["resolution", "phase_bits", "viewing_angle"]


def _summarise(records, key_fields, value_fields):
    """
    Group `records` by the tuple of key_fields and compute mean/std of each
    value_field across the group. Returns list of dicts.
    """
    groups = {}
    for r in records:
        key = tuple(r[k] for k in key_fields)
        groups.setdefault(key, []).append(r)
    summary = []
    for key, rs in groups.items():
        row = dict(zip(key_fields, key))
        for v in value_fields:
            vals = np.array([r[v] for r in rs], dtype=np.float64)
            row[f"{v}_mean"] = float(np.mean(vals))
            row[f"{v}_std"] = float(np.std(vals))
        summary.append(row)
    return summary


def run_multi_scene_validation():
    print("\n══ Multi-Scene Validation (4 scenes) ══")
    records = []  # per (sweep, parameter, scene)

    for scene_name, scene_fn in SCENE_SUITE.items():
        print(f"\n  Scene: {scene_name}")
        target = scene_fn(SIZE).astype(np.float32)
        phase = gerchberg_saxton(target, n_iter=GS_ITER,
                                 wavelength=WAVELENGTH, dx=DX, z=Z)
        ref_recon = reconstruct_phase(phase)
        for axis in MULTI_SCENE_AXES:
            for label, m, _rec in degrade_axis(axis, phase, ref_recon):
                records.append({
                    "sweep": axis, "parameter": label, "scene": scene_name,
                    "psnr": m["psnr"], "ssim": m["ssim"],
                    "lpips": m["lpips"], "lpips_proxy": m["lpips_proxy"],
                })
                print(f"    [{axis:13s} {label:8s}] "
                      f"PSNR={m['psnr']:5.1f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.4f}")

    # Per-scene CSV
    path = os.path.join(RESULTS_DIR, "multi_scene_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "parameter", "scene",
                                          "psnr", "ssim", "lpips", "lpips_proxy"])
        w.writeheader()
        w.writerows(records)
    print(f"\n  CSV saved → {path}")

    # Summary CSV (mean ± std across scenes)
    summary = _summarise(records, ["sweep", "parameter"],
                         ["psnr", "ssim", "lpips"])
    spath = os.path.join(RESULTS_DIR, "multi_scene_summary.csv")
    with open(spath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sweep", "parameter", "psnr_mean", "psnr_std",
            "ssim_mean", "ssim_std", "lpips_mean", "lpips_std"])
        w.writeheader()
        for row in summary:
            w.writerow({k: row[k] for k in w.fieldnames})
    print(f"  Summary CSV saved → {spath}")

    _plot_multi_scene_summary(summary)
    return records, summary


def _plot_multi_scene_summary(summary):
    fig, axes = plt.subplots(1, len(MULTI_SCENE_AXES),
                             figsize=(5 * len(MULTI_SCENE_AXES), 4))
    if len(MULTI_SCENE_AXES) == 1:
        axes = [axes]
    for ax, axis in zip(axes, MULTI_SCENE_AXES):
        rows = [r for r in summary if r["sweep"] == axis]
        labels = [r["parameter"] for r in rows]
        means = [r["ssim_mean"] for r in rows]
        stds = [r["ssim_std"] for r in rows]
        xs = range(len(labels))
        ax.errorbar(xs, means, yerr=stds, fmt="s-", color="#16a34a",
                    capsize=4, linewidth=2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("SSIM (mean ± std across scenes)")
        ax.set_title(axis, fontweight="bold")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Multi-Scene Generalisation — SSIM (mean ± std, 5 scenes)",
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "multi_scene_summary.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Viewing-angle diagnostic: retained spectral energy vs reconstruction SSIM
# ─────────────────────────────────────────────────────────────────────────────

def _retained_energy(phase, frac):
    """Fraction of hologram (|FFT|^2) energy kept by a circular mask of
    radius frac * Nyquist."""
    H, W = phase.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase)))) ** 2
    cx, cy = W // 2, H // 2
    radius = frac * min(H, W) / 2
    Y, X = np.ogrid[:H, :W]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2
    return float(power[mask].sum() / power.sum())


def plot_viewing_angle_energy():
    """
    Show that the bandwidth-limit operator is functioning: retained spectral
    energy decreases monotonically with bandwidth, even where reconstruction
    SSIM plateaus. Saves results/viewing_energy.{png,csv}.
    """
    print("\n── Viewing-Angle Diagnostic: retained energy vs SSIM ──")
    fracs = [1.0, 0.75, 0.5, 0.25, 0.1]
    ssim_fracs = [0.75, 0.5, 0.25, 0.1]
    energy = {}   # scene -> list over fracs
    ssim_v = {}   # scene -> list over ssim_fracs

    for name, scene_fn in SCENE_SUITE.items():
        target = scene_fn(SIZE).astype(np.float32)
        phase = gerchberg_saxton(target, n_iter=GS_ITER,
                                 wavelength=WAVELENGTH, dx=DX, z=Z)
        ref = reconstruct_phase(phase)
        energy[name] = [_retained_energy(phase, f) for f in fracs]
        ssim_v[name] = [ssim(ref, reconstruct_phase(limit_viewing_angle(phase, f)))
                        for f in ssim_fracs]

    mean_energy = np.mean([energy[n] for n in SCENE_SUITE], axis=0)
    mean_ssim = np.mean([ssim_v[n] for n in SCENE_SUITE], axis=0)

    # CSV
    cpath = os.path.join(RESULTS_DIR, "viewing_energy.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene"] + [f"energy_{int(f*100)}" for f in fracs]
                   + [f"ssim_{int(f*100)}" for f in ssim_fracs])
        for name in SCENE_SUITE:
            w.writerow([name] + [f"{v:.4f}" for v in energy[name]]
                       + [f"{v:.4f}" for v in ssim_v[name]])
        w.writerow(["MEAN"] + [f"{v:.4f}" for v in mean_energy]
                   + [f"{v:.4f}" for v in mean_ssim])
    print(f"  CSV saved → {cpath}")

    # Figure: retained energy (log, left) vs mean SSIM (linear, right)
    xe = [f * 100 for f in fracs]
    xs = [f * 100 for f in ssim_fracs]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    for name in SCENE_SUITE:
        ax1.plot(xe, energy[name], "-", color="#94a3b8", linewidth=1, alpha=0.7)
    ax1.plot(xe, mean_energy, "o-", color="#2563eb", linewidth=2.5,
             label="retained energy (mean)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Bandwidth (% of Nyquist)")
    ax1.set_ylabel("Retained spectral energy (fraction)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, mean_ssim, "s--", color="#16a34a", linewidth=2.5,
             label="reconstruction SSIM (mean)")
    ax2.set_ylabel("Reconstruction SSIM (mean)", color="#16a34a")
    ax2.tick_params(axis="y", labelcolor="#16a34a")
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Viewing angle: energy removed monotonically, SSIM plateaus then drops",
                 fontsize=10, fontweight="bold")
    lines = ax1.get_lines()[-1:] + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "viewing_energy.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  retained energy (mean): "
          + ", ".join(f"{int(f*100)}%={v:.3f}" for f, v in zip(fracs, mean_energy)))
    print(f"  Figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Seed sensitivity (Task 8)
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_sensitivity():
    print("\n══ Seed Sensitivity (5 GS seeds) ══")
    target = gaussian_spots(SIZE, n_spots=4, sigma=SIZE * 0.025, seed=42)
    records = []  # per (sweep, parameter, seed)

    for seed in SEEDS:
        phase = gerchberg_saxton(target, n_iter=GS_ITER,
                                 wavelength=WAVELENGTH, dx=DX, z=Z, seed=seed)
        ref_recon = reconstruct_phase(phase)
        for axis in ["phase_bits", "resolution"]:
            for label, m, _rec in degrade_axis(axis, phase, ref_recon):
                records.append({
                    "sweep": axis, "parameter": label, "seed": seed,
                    "psnr": m["psnr"], "ssim": m["ssim"], "lpips": m["lpips"],
                })

    path = os.path.join(RESULTS_DIR, "seed_sensitivity.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "parameter", "seed",
                                          "psnr", "ssim", "lpips"])
        w.writeheader()
        w.writerows(records)
    print(f"  CSV saved → {path}")

    summary = _summarise(records, ["sweep", "parameter"],
                         ["psnr", "ssim", "lpips"])
    spath = os.path.join(RESULTS_DIR, "seed_sensitivity_summary.csv")
    with open(spath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sweep", "parameter", "psnr_mean", "psnr_std",
            "ssim_mean", "ssim_std", "lpips_mean", "lpips_std"])
        w.writeheader()
        for row in summary:
            w.writerow({k: row[k] for k in w.fieldnames})
    print(f"  Summary CSV saved → {spath}")

    # Print a table to stdout
    print("\n  Seed-sensitivity summary (mean ± std across 5 seeds):")
    print(f"  {'sweep':13s} {'param':9s} {'PSNR (dB)':16s} {'SSIM':14s} {'LPIPS':14s}")
    for row in sorted(summary, key=lambda r: (r["sweep"], r["parameter"])):
        print(f"  {row['sweep']:13s} {row['parameter']:9s} "
              f"{row['psnr_mean']:6.2f} ± {row['psnr_std']:4.2f}    "
              f"{row['ssim_mean']:.3f} ± {row['ssim_std']:.3f}   "
              f"{row['lpips_mean']:.3f} ± {row['lpips_std']:.3f}")
    return records, summary


# ─────────────────────────────────────────────────────────────────────────────
#  Metric-divergence summary plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_summary(all_rows):
    # Only the four reported degradation axes (D1-D4). The depth-plane sweep is a
    # coherent-superposition demonstration, not a reported Part 1 axis, so it is
    # excluded from this summary figure (see paper scope statement).
    reported = ["resolution", "phase_bits", "viewing_angle", "speckle"]
    sweeps = {}
    for row in all_rows:
        if row["sweep"] not in reported:
            continue
        sweeps.setdefault(row["sweep"], []).append(row)

    fig, axes = plt.subplots(len(sweeps), 2, figsize=(12, 4 * len(sweeps)))
    if len(sweeps) == 1:
        axes = [axes]

    for ax_row, (sweep_name, rows) in zip(axes, sweeps.items()):
        labels = [r["label"] for r in rows]
        psnr_v = [r["psnr"] for r in rows]
        ssim_v = [r["ssim"] for r in rows]
        xs = range(len(labels))

        ax_row[0].plot(xs, psnr_v, "o-", color="#2563eb", linewidth=2)
        ax_row[0].set_xticks(list(xs)); ax_row[0].set_xticklabels(labels, fontsize=8)
        ax_row[0].set_ylabel("PSNR (dB)")
        ax_row[0].set_title(f"{sweep_name} — PSNR", fontweight="bold")
        ax_row[0].grid(True, alpha=0.3)

        ax_row[1].plot(xs, ssim_v, "s-", color="#16a34a", linewidth=2)
        ax_row[1].set_xticks(list(xs)); ax_row[1].set_xticklabels(labels, fontsize=8)
        ax_row[1].set_ylabel("SSIM"); ax_row[1].set_ylim(0, 1.05)
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
    fieldnames = ["sweep", "label", "psnr", "ssim", "mse", "lpips_proxy", "lpips"]
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

    np.save(os.path.join(RESULTS_DIR, "reference_target.npy"), target)
    np.save(os.path.join(RESULTS_DIR, "reference_phase.npy"), phase)
    np.save(os.path.join(RESULTS_DIR, "reference_recon.npy"), ref_recon)

    # GS convergence diagnostic first
    plot_gs_convergence()

    # Single-scene degradation sweeps
    sweep_resolution(phase, ref_recon, all_rows)
    sweep_phase_bits(phase, ref_recon, all_rows)
    sweep_color(phase, ref_recon, all_rows)          # EXCLUDED stub
    sweep_viewing_angle(phase, ref_recon, all_rows)
    sweep_depth_planes(all_rows)
    sweep_speckle(phase, ref_recon, all_rows)

    plot_metrics_summary(all_rows)
    export_csv(all_rows)

    # Generalisation + statistical rigour
    run_multi_scene_validation()
    plot_viewing_angle_energy()
    run_seed_sensitivity()

    print(f"\n✓ All done in {time.time()-t_start:.1f}s — results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
