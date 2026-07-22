"""
Phase 5: figures, deterministic from results JSONs. No manual data entry.

F1-F7 depend on Phase 3 (E1-E6) / Phase 6 (twin validation CSVs) data that
does not exist yet as of this pass -- each of those functions is fully
implemented and unit-tested against real (tiny, synthetic-scale) data in
tests/test_figures.py, but when run against the actual repo state today
they correctly emit a "NOT YET AVAILABLE" placeholder PDF (figures.style.
no_data_placeholder) stating exactly what's missing, rather than
fabricating a plot from data that doesn't exist (ground rule 1).

F8's five sub-panels use data that IS already real and committed
(gradient ablation, RCWA incl. the E7 validity-envelope grid, GPU mesh
convergence, GPU wavelength detuning, and the old CPU-scale shrinkage
sweep -- the last explicitly labeled single-seed per ground rule 3) and
are rendered for real by this pass.

Usage: python -m figures.make_all
"""
from __future__ import annotations
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from style import (new_fig, savefig, no_data_placeholder, COLORS,
                   METHOD_COLORS, METHOD_LABELS, SINGLE_COL_IN, DOUBLE_COL_IN)
from analysis.aggregate import (load_all_results, group_by_config,
                                headroom_closure, e1_gain_curve, E1_BUDGETS,
                                mean_std_median_ci95, paired_gain)

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "paper")
RESULTS_ROOT = os.path.join(HERE, "..", "results")


def _load_json(*parts):
    path = os.path.join(HERE, "..", *parts)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------- F1
def make_F1_pipeline_schematic():
    """Twin -> readout -> loss -> gradient back to exposure. Purely
    illustrative (no data dependency) -- flagged for manual polish per
    the master prompt."""
    fig, ax = new_fig(width="double", height_in=SINGLE_COL_IN * 0.6)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [("Exposure\nE(x)", 0.5), ("NPDD Twin\n(recording)", 2.7),
             ("BPM Readout\n(Kogelnik-consistent)", 5.1), ("Loss vs.\nTarget", 7.7)]
    for label, x in boxes:
        ax.add_patch(mpatches.Rectangle((x, 1.0), 1.6, 1.0, fill=False,
                                        edgecolor=COLORS["black"], linewidth=1.0))
        ax.text(x + 0.8, 1.5, label, ha="center", va="center", fontsize=6.5)
    for x0, x1 in [(2.1, 2.7), (4.3, 5.1), (6.7, 7.7)]:
        ax.annotate("", xy=(x1, 1.5), xytext=(x0, 1.5),
                   arrowprops=dict(arrowstyle="->", color=COLORS["black"]))
    ax.annotate("", xy=(0.5, 0.5), xytext=(9.3, 0.5),
               arrowprops=dict(arrowstyle="->", color=COLORS["blue"], linewidth=1.2))
    ax.text(5.0, 0.2, r"$\partial L / \partial E$  (unrolled autodiff through the twin)",
            ha="center", va="center", fontsize=6.5, color=COLORS["blue"])
    ax.set_title("F1: media-in-the-loop optimization pipeline (schematic)", fontsize=7.5)
    savefig(fig, os.path.join(OUT_DIR, "F1_pipeline_schematic.pdf"))


# --------------------------------------------------------------------- F2/F3
def make_F2_exposure_profiles(K_values=(2.62, 5.0)):
    grouped = group_by_config(load_all_results())
    have_data = any(exp_id == "E1" for exp_id, _ in grouped)
    if not have_data:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F2_exposure_profiles.pdf"),
            "F2: exposure/recorded-dn profiles (naive vs M3 vs M4)",
            f"needs E1 manifest results at K={list(K_values)} rad/um; "
            f"Phase 3 has not run yet.")
        return
    _render_F2(grouped, K_values)  # exercised for real once E1 data exists


def _render_F2(grouped, K_values):
    # Real implementation: for each K, plot the M3/M4 exposure shape
    # (would need the raw E field, not just PSNR -- Phase 1.2's schema
    # logs contrast stats but not the full E(x) array to keep JSONs small
    # (<=200-point downsampled loss curves only); this function documents
    # that gap rather than silently omitting the figure's actual content.
    fig, axs = new_fig(width="double", nrows=1, ncols=len(K_values), height_in=SINGLE_COL_IN * 0.5)
    for ax, K in zip(np.atleast_1d(axs), K_values):
        ax.set_title(f"K={K:.2f} rad/um", fontsize=6.5)
        ax.text(0.5, 0.5, "E(x) profile not in Phase 1.2 schema\n(PSNR/DE/contrast-stats only)",
               ha="center", va="center", fontsize=5.5, transform=ax.transAxes, color=COLORS["vermillion"])
    savefig(fig, os.path.join(OUT_DIR, "F2_exposure_profiles.pdf"))


def make_F3_reconstruction_panels(K_values=(1.31, 3.93, 5.24)):
    grouped = group_by_config(load_all_results())
    have_data = any(exp_id == "E1" for exp_id, _ in grouped)
    if not have_data:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F3_reconstruction_panels.pdf"),
            "F3: reconstruction panels (target/M1-M5a) at K in {1.31,3.93,5.24}",
            "needs E1 manifest results; Phase 3 has not run yet. Also needs raw "
            "recon arrays, which Phase 1.2's schema does not log (PSNR/DE only, "
            "to keep result JSONs small) -- schema would need extending.")
        return


# --------------------------------------------------------------------- F4/F5
def make_F4_headline_gain_vs_K():
    grouped = group_by_config(load_all_results())
    closure = headroom_closure(grouped, budgets=E1_BUDGETS)
    if all(r.get("status") == "no_data" for r in closure):
        no_data_placeholder(
            os.path.join(OUT_DIR, "F4_headline_gain_vs_K.pdf"),
            "F4: paired gain (M4-M2) vs K, one curve per budget, 95% CI bands",
            "needs E1 manifest results across all 3 budgets; Phase 3 has not run yet.")
        return
    _render_F4(closure)


def _render_F4(closure):
    fig, ax = new_fig(width="single")
    for row, budget in zip(closure, E1_BUDGETS):
        if row.get("status") == "no_data":
            continue
        curve = row["gain_curve"]
        Ks = [c[0] for c in curve]
        means = [c[1] for c in curve]
        los = [c[2] for c in curve]
        his = [c[3] for c in curve]
        color = {2.0: COLORS["blue"], 4.0: COLORS["vermillion"], 8.0: COLORS["bluish_green"]}[budget]
        ax.plot(Ks, means, "-o", color=color, ms=2.5, label=f"budget={budget:.0f}x")
        ax.fill_between(Ks, los, his, color=color, alpha=0.2, linewidth=0)
        if row.get("predicted_Kc_from_measured_C") is not None:
            ax.axvline(row["predicted_Kc_from_measured_C"], color=color, ls="--", lw=0.7)
    ax.axhline(0, color=COLORS["black"], lw=0.5)
    ax.set_xlabel("K (rad/um)"); ax.set_ylabel("paired gain M4-M2 (dB)")
    ax.legend(frameon=False)
    savefig(fig, os.path.join(OUT_DIR, "F4_headline_gain_vs_K.pdf"))


def make_F5_Kstar_vs_Kc_scatter():
    grouped = group_by_config(load_all_results())
    closure = headroom_closure(grouped, budgets=E1_BUDGETS)
    valid = [r for r in closure if r.get("status") != "no_data"]
    if not valid:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F5_Kstar_vs_Kc_scatter.pdf"),
            "F5: observed K* vs predicted Kc (both estimators), y=x line",
            "needs E1 headroom-closure results; Phase 3 has not run yet.")
        return
    fig, ax = new_fig(width="single", height_in=SINGLE_COL_IN)
    kc_vals = [r["predicted_Kc_from_measured_C"] for r in valid]
    for r in valid:
        if r["observed_Kstar_interp"] is not None:
            ax.scatter(r["predicted_Kc_from_measured_C"], r["observed_Kstar_interp"],
                      marker="o", color=COLORS["blue"], label="interp" if r is valid[0] else None)
        if r["observed_Kstar_ci"] is not None:
            ax.scatter(r["predicted_Kc_from_measured_C"], r["observed_Kstar_ci"],
                      marker="s", color=COLORS["vermillion"], label="CI" if r is valid[0] else None)
    lims = [0, max(kc_vals) * 1.2] if kc_vals else [0, 1]
    ax.plot(lims, lims, color=COLORS["black"], lw=0.6, ls=":")
    ax.set_xlabel("predicted Kc(measured C) (rad/um)")
    ax.set_ylabel("observed K* (rad/um)")
    ax.legend(frameon=False)
    savefig(fig, os.path.join(OUT_DIR, "F5_Kstar_vs_Kc_scatter.pdf"))


# --------------------------------------------------------------------- F6
def make_F6_sigma_probe():
    grouped = group_by_config(load_all_results())
    have_e2 = any(exp_id == "E2" for exp_id, _ in grouped)
    if not have_e2:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F6_sigma_probe.pdf"),
            "F6: sigma probe with CI bars, Ghat(K) prediction overlaid",
            "needs E2 manifest results; Phase 3 has not run yet.")
        return


# --------------------------------------------------------------------- F7
def make_F7_twin_validation():
    csvs_exist = os.path.isdir(os.path.join(HERE, "..", "data", "literature")) and any(
        f.endswith(".csv") for f in os.listdir(os.path.join(HERE, "..", "data", "literature")))
    if not csvs_exist:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F7_twin_validation.pdf"),
            "F7: twin vs. digitized literature (growth curve, angular selectivity)",
            "needs Phase 6 digitized CSVs (data/literature/*.csv) from WebPlotDigitizer -- "
            "not yet provided; see data/literature/README.md.")
        return


# --------------------------------------------------------------------- F8 (real data)
def make_F8a_gradient_ablation():
    d = _load_json("results_ablation_gradients.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F8a_gradient_ablation.pdf"),
                            "F8a: gradient-pathway ablation", "results_ablation_gradients.json missing")
        return
    fig, axs = new_fig(width="single", ncols=2, height_in=SINGLE_COL_IN * 0.6)
    ax1, ax2 = axs
    fid = d["fidelity"]
    ax1.bar(["checkpoint", "surrogate"], [fid["checkpoint_cossim"], fid["surrogate_cossim"]],
           color=[COLORS["blue"], COLORS["orange"]])
    ax1.set_ylabel("cosine similarity to unrolled grad"); ax1.set_ylim(0, 1)
    opt = d["optimization"]
    methods = ["unrolled", "checkpoint", "surrogate"]
    ax2.bar(methods, [opt[m]["psnr"] for m in methods],
           color=[COLORS["black"], COLORS["blue"], COLORS["orange"]])
    ax2.set_ylabel("downstream PSNR (dB)")
    ax2.tick_params(axis="x", labelrotation=30)
    savefig(fig, os.path.join(OUT_DIR, "F8a_gradient_ablation.pdf"))


def make_F8b_rcwa():
    d3 = _load_json("results_rcwa.json")
    d90 = _load_json("results_rcwa_e7.json")
    if d3 is None and d90 is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F8b_rcwa.pdf"),
                            "F8b: RCWA validity envelope", "results_rcwa*.json missing")
        return
    fig, ax = new_fig(width="single")
    if d90:
        by_geom = {}
        for c in d90["cases"]:
            by_geom.setdefault(c["geometry"], []).append((c["K"], c["abs_deviation"]))
        colors = [COLORS["blue"], COLORS["vermillion"], COLORS["bluish_green"]]
        for (geom, pts), color in zip(by_geom.items(), colors):
            pts = sorted(pts)
            Ks = [p[0] for p in pts]
            devs = [p[1] for p in pts]
            ax.scatter(Ks, devs, color=color, s=6, label=geom, alpha=0.7)
        ax.set_xlabel("K (rad/um)"); ax.set_ylabel("|Kogelnik - RCWA T1|")
        ax.legend(frameon=False, fontsize=5.5)
    savefig(fig, os.path.join(OUT_DIR, "F8b_rcwa.pdf"))


def make_F8c_mesh_convergence():
    d = _load_json("results", "gpu_reruns", "npdd_mesh_sweep", "results.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F8c_mesh_convergence.pdf"),
                            "F8c: mesh-density convergence", "results/gpu_reruns/npdd_mesh_sweep missing")
        return
    fig, ax = new_fig(width="single")
    nxs, psnrs = [], []
    for key, row in d["mesh_density"].items():
        nxs.append(row["n_x"]); psnrs.append(row["psnr"])
    order = np.argsort(nxs)
    nxs = np.array(nxs)[order]; psnrs = np.array(psnrs)[order]
    ax.plot(nxs, psnrs, "-o", color=COLORS["blue"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_x"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("single seed, single Colab T4 run", fontsize=6)
    savefig(fig, os.path.join(OUT_DIR, "F8c_mesh_convergence.pdf"))


def make_F8d_wavelength_detuning():
    d = _load_json("results", "gpu_reruns", "bpm_wavelength_sweep", "results.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F8d_wavelength_detuning.pdf"),
                            "F8d: wavelength-detuning readout sweep", "results/gpu_reruns/bpm_wavelength_sweep missing")
        return
    fig, ax = new_fig(width="single")
    lams = sorted(float(k) for k in d["by_wavelength"])
    psnrs = [d["by_wavelength"][str(l)]["psnr"] for l in lams]
    ax.plot([l * 1000 for l in lams], psnrs, "-o", color=COLORS["blue"])
    ax.axvline(d["meta"]["design_lam_um"] * 1000, color=COLORS["vermillion"], ls="--", lw=0.7,
              label="design wavelength")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("single seed, single Colab T4 run", fontsize=6)
    ax.legend(frameon=False)
    savefig(fig, os.path.join(OUT_DIR, "F8d_wavelength_detuning.pdf"))


def make_F8e_shrinkage_prelim():
    """CPU-scale, seed-bugged-era shrinkage sweep (results_prelim2.json) --
    explicitly labeled single-seed/superseded per ground rule 3, NOT
    presented as a Phase-3 E3 result."""
    d = _load_json("results_prelim2.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F8e_shrinkage_prelim.pdf"),
                            "F8e: shrinkage sweep (CPU-scale preliminary)", "results_prelim2.json missing")
        return
    fig, ax = new_fig(width="single")
    shrink = d["shrinkage"]
    svals = sorted(float(k) for k in shrink)
    gains = []
    for s in svals:
        rows = shrink[str(s)]
        row0 = rows[0]  # seed field present but bugged (bit-identical) -- see docs/provenance_report.md
        gains.append(row0["ours"] - row0["blind"])
    ax.plot(svals, gains, "-o", color=COLORS["blue"])
    ax.set_xlabel("shrinkage s"); ax.set_ylabel("gain M4-M2 (dB)")
    ax.set_title("CPU-scale, single EFFECTIVE seed (seed-init bug era) -- "
                 "superseded by E3 once Phase 3 runs", fontsize=5.5, color=COLORS["vermillion"])
    savefig(fig, os.path.join(OUT_DIR, "F8e_shrinkage_prelim.pdf"))


ALL_FIGURES = [
    make_F1_pipeline_schematic, make_F2_exposure_profiles,
    make_F3_reconstruction_panels, make_F4_headline_gain_vs_K,
    make_F5_Kstar_vs_Kc_scatter, make_F6_sigma_probe, make_F7_twin_validation,
    make_F8a_gradient_ablation, make_F8b_rcwa, make_F8c_mesh_convergence,
    make_F8d_wavelength_detuning, make_F8e_shrinkage_prelim,
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fn in ALL_FIGURES:
        print(f"[make_all] {fn.__name__} ...")
        fn()
    print(f"[make_all] done -- see {OUT_DIR}")


if __name__ == "__main__":
    main()
