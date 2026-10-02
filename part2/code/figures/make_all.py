"""
Every figure in the Part 2 manuscript and its supplement, generated
deterministically from committed result files. No manual data entry.

Main text (paper/manuscript.tex)
  Fig1_paired_gain.pdf        (a) MIL - BSGD vs K, three budgets (tier M1)
                              (b) every non-oracle method vs BSGD at 2x (M1)
  Fig2_reconstructions.pdf    target / BSGD / MIL profiles and errors, three K
  Fig3_slant_mismatch.pdf     (a) gain vs grating slant (S8)
                              (b) gain vs one-parameter twin error (S3)
  Fig4_twin_validation.pdf    twin fits to digitized literature curves

Supplement (paper/supplement.tex)
  FigS1_exposure_profiles.pdf exposure and index profiles behind Fig. 2
  FigS2_ablation.pdf          one-mechanism-at-a-time ablation (S1)
  FigS3_factorial.pdf         saturation-mechanism factorial (S1X)
  FigS4_rcwa_envelope.pdf     Kogelnik vs RCWA efficiency deviation
  FigS5_twin_fits_held_kbleach.pdf  twin fits with k_bleach held (incl. PQ/PMMA)

A figure whose input data are missing is written as a labelled
"not available" placeholder PDF instead of being silently skipped, and its
function returns False (True = real content).

Usage (from part2/code): python figures/make_all.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np

from style import (new_fig, savefig, no_data_placeholder, panel_label, COLORS,
                   METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS, METHOD_LINESTYLES,
                   BUDGET_COLORS, BUDGET_LINESTYLES, BUDGET_MARKERS,
                   SINGLE_COL_IN, DOUBLE_COL_IN, K_LABEL, GAIN_LABEL)
from analysis.aggregate import (load_all_results as _load_all_results_raw,
                                group_by_config, split_complete_m1,
                                headroom_closure, gain_curve, BUDGETS,
                                mean_std_median_ci95, paired_gain,
                                gain_vs_bsgd_seed_mean)
import run_manifest as rm

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "paper")

# Grid spacing of every 1D job (Table 1: n_x = 1024, dx = 0.05 um).
DX_UM = 0.05
# Window shown in the profile figures: 9.6 um, i.e. 2, 6 and 8 periods at
# K = 1.31, 3.93 and 5.24 rad/um -- the full 51.2 um window is unreadable.
CROP_PX = 192
K_TICKS = [2, 3, 4, 5, 7, 10, 15]


def load_all_results():
    """analysis.aggregate.load_all_results, reading run_manifest.RESULTS_ROOT
    so that tests can redirect it with rm.set_results_root()."""
    return _load_all_results_raw(rm.RESULTS_ROOT)


def load_grouped_complete():
    """Grouped results with incomplete M1 cells dropped (the same rule as
    analysis.aggregate.load_grouped_complete)."""
    return split_complete_m1(group_by_config(load_all_results()))[0]


def _load_json(name):
    path = os.path.join(HERE, "..", name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _out(name):
    return os.path.join(OUT_DIR, name)


def _log_K_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(K_TICKS)
    ax.set_xticklabels([str(k) for k in K_TICKS])
    ax.minorticks_off()
    ax.set_xlabel(K_LABEL)


def _ci_err(means, los, his):
    return [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]


# --------------------------------------------------------------------- main text
def make_fig1_paired_gain():
    """(a) Paired gain vs K for the three contrast budgets, 95% t-intervals
    over seeds, heuristic K_c(B_c) as dashed lines. (b) Every non-oracle
    method against BSGD at budget 2x. GS/LPC/GPC are deterministic (seed 0
    only) and are compared with BSGD's seed mean; MIL/SAT/RSGD are paired
    per seed."""
    path = _out("Fig1_paired_gain.pdf")
    grouped = load_grouped_complete()
    closure = headroom_closure(grouped, "M1", budgets=BUDGETS)
    if all(r.get("status") == "no_data" for r in closure):
        no_data_placeholder(path, "Fig. 1: paired gain vs K", "needs tier M1 results.",
                            width="double")
        return False

    fig, (ax_a, ax_b) = new_fig(width="text", height_in=2.0, ncols=2)
    for row, budget in zip(closure, BUDGETS):
        if row.get("status") == "no_data":
            continue
        Ks, means, los, his = zip(*[c[:4] for c in row["gain_curve"]])
        col = BUDGET_COLORS[budget]
        ax_a.plot(Ks, means, color=col, ls=BUDGET_LINESTYLES[budget],
                  marker=BUDGET_MARKERS[budget], ms=3, label=rf"$B_c={budget:.0f}$")
        ax_a.fill_between(Ks, los, his, color=col, alpha=0.2, linewidth=0)
        kc = row.get("predicted_Kc_from_measured_C")
        if kc is not None:
            ax_a.axvline(kc, color=col, ls=(0, (2, 2)), lw=0.7)
    ax_a.axhline(0, color=COLORS["black"], lw=0.5)
    _log_K_axis(ax_a)
    ax_a.set_ylabel(GAIN_LABEL)
    ax_a.legend(frameon=False, loc="upper right")
    panel_label(ax_a, "a")

    budget = 2.0
    curves = {m: gain_curve(grouped, "M1", budget, method=m) for m in ("MIL", "SAT", "RSGD")}
    for m in ("GS", "LPC", "GPC"):
        curves[m] = gain_vs_bsgd_seed_mean(grouped, "M1", budget, method=m)
    for m in ("GS", "LPC", "GPC", "RSGD", "SAT", "MIL"):
        if not curves[m]:
            continue
        Ks = [c[0] for c in curves[m]]
        means = [c[1] for c in curves[m]]
        ax_b.plot(Ks, means, color=METHOD_COLORS[m], ls=METHOD_LINESTYLES[m],
                  marker=METHOD_MARKERS[m], ms=3, label=METHOD_LABELS[m])
    ax_b.axhline(0, color=COLORS["black"], lw=0.5)
    _log_K_axis(ax_b)
    ax_b.set_ylabel(r"gain over BSGD at $B_c=2$ (dB)")
    if ax_b.get_legend_handles_labels()[0]:
        ax_b.legend(frameon=False, ncol=2, loc="upper right", columnspacing=1.0)
    panel_label(ax_b, "b")
    savefig(fig, path)
    return True


def make_fig2_reconstructions():
    """Target, BSGD and MIL reconstructions (top) and their errors (bottom),
    single seed, budget 2x, at the three K points of the ablation studies.
    The stored reconstructions are already scaled by the optimal factor
    alpha* of the scale-invariant loss, so they are plotted as stored."""
    path = _out("Fig2_reconstructions.pdf")
    d = _load_json("results_r1_reconstructions.json")
    if not d or not d.get("results"):
        no_data_placeholder(path, "Fig. 2: reconstructions",
                            "needs results_r1_reconstructions.json "
                            "(experiments/make_r1_reconstructions.py).", width="double")
        return False

    results = d["results"]
    fig, axes = new_fig(width="text", height_in=2.3, ncols=len(results), nrows=2,
                        squeeze=False, sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    x = np.arange(CROP_PX) * DX_UM
    for j, r in enumerate(results):
        sl = slice(0, CROP_PX)
        t = np.array(r["target"])[sl]
        b = np.array(r["recon_bsgd"])[sl]
        m = np.array(r["recon_mil"])[sl]
        top, bot = axes[0][j], axes[1][j]
        top.fill_between(x, 0, t, step="mid", color="0.85", linewidth=0, label="target")
        top.plot(x, b, color=METHOD_COLORS["BSGD"], ls="--", lw=0.9,
                 label=f"BSGD, {r['psnr_bsgd']:.1f} dB")
        top.plot(x, m, color=METHOD_COLORS["MIL"], lw=0.9, label=f"MIL, {r['psnr_mil']:.1f} dB")
        top.set_title(rf"$K={r['K']:.2f}$ rad/$\mu$m")
        top.set_ylim(0, 1.6)
        top.legend(frameon=False, loc="upper right", fontsize=6, ncol=1,
                   handlelength=1.6, borderaxespad=0.1)
        bot.plot(x, b - t, color=METHOD_COLORS["BSGD"], ls="--", lw=0.8)
        bot.plot(x, m - t, color=METHOD_COLORS["MIL"], lw=0.8)
        bot.axhline(0, color=COLORS["black"], lw=0.4)
        bot.set_ylim(-1.0, 1.0)
        bot.set_xlabel(r"$x$ ($\mu$m)")
        if j == 0:
            top.set_ylabel("intensity (a.u.)")
            bot.set_ylabel("error")
        panel_label(top, "abc"[j])
    savefig(fig, path)
    return True


def _s3_param_curves(results):
    """K-averaged paired gain per seed, then mean and 95% t-interval over
    seeds, for each (parameter, perturbation). Pairs within each K first:
    paired_gain is seed-keyed and must never see a pooled set of K."""
    buckets = {}
    for r in results:
        c = r["config"]
        buckets.setdefault((c.get("mismatch_param"), c.get("mismatch_pct"),
                            c.get("K_nominal"), r["method_id"]), []).append(r)
    all_K = sorted({r["config"]["K_nominal"] for r in results})
    params = sorted({k[0] for k in buckets})
    nominal_param = next(k[0] for k in buckets if k[1] == 0)  # pct 0 is stored once
    curves = {}
    for p in params:
        pts = []
        for pct in sorted({k[1] for k in buckets if k[0] == p} | {0}):
            prefix = (nominal_param if pct == 0 else p, pct)
            per_seed = {}
            for K in all_K:
                for seed, g in paired_gain(buckets.get(prefix + (K, "MIL"), []),
                                           buckets.get(prefix + (K, "BSGD"), []), key="psnr"):
                    per_seed.setdefault(seed, []).append(g)
            gains = [sum(v) / len(v) for v in per_seed.values() if v]
            if gains:
                st = mean_std_median_ci95(gains)
                pts.append((pct, st["mean"], st["ci95_lo"], st["ci95_hi"]))
        curves[p] = pts
    return curves


S3_PARAM_LABELS = {"D0": r"$D_0$", "sigma": r"$\sigma$", "kappa": r"$\kappa$",
                   "dn_max": r"$\Delta n_{\max}$"}


def make_fig3_slant_mismatch():
    """(a) Paired gain vs grating slant at two K (S8). (b) Paired gain when
    the exposure is designed on the nominal twin and recorded on a twin with
    one parameter scaled (S3), K-averaged over the three ablation K points,
    plotted against the recorded-to-design parameter ratio on a log axis so
    that the wide dn_max sweep and the +/-50% sweeps share one panel."""
    path = _out("Fig3_slant_mismatch.pdf")
    s8 = [r for r in load_all_results() if r["experiment_id"] == "S8"]
    s3 = [r for r in load_all_results() if r["experiment_id"] == "S3"]
    if not s8 or not s3:
        no_data_placeholder(path, "Fig. 3: slant and miscalibration",
                            "needs tier S8 and S3 results.", width="double")
        return False

    fig, (ax_a, ax_b) = new_fig(width="text", height_in=1.95, ncols=2)
    by = {}
    for r in s8:
        c = r["config"]
        by.setdefault((round(c["K_nominal"], 4), float(c["slant_deg"])), {}) \
          .setdefault(r["method_id"], []).append(r)
    for i, K in enumerate(sorted({k for k, _ in by})):
        xs, ms, los, his = [], [], [], []
        for slant in sorted({s for k, s in by if k == K}):
            arms = by[(K, slant)]
            gains = [g for _, g in paired_gain(arms.get("MIL", []), arms.get("BSGD", []),
                                               key="psnr")]
            if gains:
                st = mean_std_median_ci95(gains)
                xs.append(slant); ms.append(st["mean"])
                los.append(st["ci95_lo"]); his.append(st["ci95_hi"])
        ax_a.errorbar(xs, ms, yerr=_ci_err(ms, los, his), color=[COLORS["blue"],
                      COLORS["vermillion"]][i % 2], marker="os"[i % 2], ls=["-", "--"][i % 2],
                      ms=3.5, capsize=2, label=rf"$K={K:.2f}$ rad/$\mu$m")
    ax_a.axhline(0, color=COLORS["black"], lw=0.5)
    ax_a.set_xlabel(r"grating slant $\phi$ (deg)")
    ax_a.set_ylabel(GAIN_LABEL)
    ax_a.legend(frameon=False)
    panel_label(ax_a, "a")

    styles = {"D0": ("o", "-"), "sigma": ("s", "--"), "kappa": ("^", ":"),
              "dn_max": ("D", "-.")}
    palette = {"D0": COLORS["blue"], "sigma": COLORS["vermillion"],
               "kappa": COLORS["bluish_green"], "dn_max": COLORS["reddish_purple"]}
    for p, pts in _s3_param_curves(s3).items():
        if not pts:
            continue
        ratio = [1 + q[0] / 100 for q in pts]
        means = [q[1] for q in pts]
        ax_b.plot(ratio, means, color=palette.get(p, "0.4"), marker=styles.get(p, ("o", "-"))[0],
                  ls=styles.get(p, ("o", "-"))[1], ms=3, label=S3_PARAM_LABELS.get(p, p))
        ax_b.fill_between(ratio, [q[2] for q in pts], [q[3] for q in pts],
                          color=palette.get(p, "0.4"), alpha=0.15, linewidth=0)
    ax_b.axvspan(0.5, 1.5, color=COLORS["black"], alpha=0.06, linewidth=0)
    ax_b.axhline(0, color=COLORS["black"], lw=0.5)
    ax_b.set_xscale("log")
    ticks = [0.2, 0.5, 1, 2, 5]
    ax_b.set_xticks(ticks)
    ax_b.set_xticklabels([f"{t:g}" for t in ticks])
    ax_b.minorticks_off()
    ax_b.set_xlabel("recorded-medium parameter / design value")
    ax_b.set_ylabel(r"$K$-averaged gain (dB)")
    ax_b.legend(frameon=False, loc="lower left", ncol=2)
    panel_label(ax_b, "b")
    savefig(fig, path)
    return True


def make_fig4_twin_validation():
    """Bayfol HX Delta-n1(dose), source model and measurement (Bruder et al.
    2017, Fig. 3), with the twin fitted to each series with (kappa, dn_max,
    k_bleach) free (solid) and the parameter-free prediction from the fit to
    the OTHER series (dashed). Data: experiments/fit_twin_holdout.py."""
    path = _out("Fig4_twin_validation.pdf")
    d = _load_json("results_twin_holdout.json")
    if not d or "fits" not in d:
        no_data_placeholder(path, "Fig. 4: twin validation",
                            "needs results_twin_holdout.json "
                            "(experiments/fit_twin_holdout.py).", width="double")
        return False
    fits, data = d["fits"], d["data"]
    other = {"sim": "exp", "exp": "sim"}
    titles = {"sim": "source kinetic model", "exp": "measurement"}
    fig, axes = new_fig(width="text", height_in=1.85, ncols=2)
    for j, s in enumerate(("sim", "exp")):
        ax = axes[j]
        fit = fits[f"B_kbleach_free/{s}"]
        pred = fits[f"B_kbleach_free/{other[s]}"]
        x = np.array(data[s]["x"])
        ax.plot(x, 1e3 * np.array(data[s]["y"]), ls="none", marker="s", ms=3.5,
                color=COLORS["vermillion"], label="digitized", zorder=3)
        ax.plot(x, 1e3 * np.array(fit["train_pred"]), color=COLORS["blue"], marker="o", ms=2.5,
                label=f"fit, NRMSE {fit['nrmse']:.2f}")
        ax.plot(x, 1e3 * np.array(pred["heldout_pred"]), color=COLORS["black"], ls="--",
                marker="^", ms=2.5,
                label=f"held-out prediction, {pred['heldout_nrmse']:.2f}")
        ax.set_xscale("log")
        ax.set_title(f"Bayfol HX, {titles[s]}")
        ax.set_xlabel(r"exposure dose (mJ/cm$^2$)")
        ax.set_ylabel(r"$\Delta n_1$ ($\times10^{-3}$)")
        ax.set_ylim(0, 17)
        ax.legend(frameon=False, loc="lower right",
                  handlelength=1.8)
        panel_label(ax, "ab"[j])
    savefig(fig, path)
    return True


# --------------------------------------------------------------------- supplement
def make_figS1_exposure_profiles():
    """Exposure E(x) and recorded index Delta n(x), BSGD vs MIL, same K,
    budget, seed and crop window as Fig. 2."""
    path = _out("FigS1_exposure_profiles.pdf")
    d = _load_json("results_r1_profiles.json")
    if not d or not d.get("results"):
        no_data_placeholder(path, "Fig. S1: exposure and index profiles",
                            "needs results_r1_profiles.json "
                            "(experiments/make_r1_profiles.py).", width="double")
        return False
    results = d["results"]
    fig, axes = new_fig(width="text", height_in=2.5, ncols=len(results), nrows=2,
                        squeeze=False, sharex=True)
    x = np.arange(CROP_PX) * DX_UM
    for j, r in enumerate(results):
        top, bot = axes[0][j], axes[1][j]
        for key, ax, scale in (("E", top, 1.0), ("dn", bot, 1e3)):
            ax.plot(x, scale * np.array(r[f"{key}_bsgd"])[:CROP_PX], color=METHOD_COLORS["BSGD"],
                    ls="--", lw=0.9, label="BSGD")
            ax.plot(x, scale * np.array(r[f"{key}_mil"])[:CROP_PX], color=METHOD_COLORS["MIL"],
                    lw=0.9, label="MIL")
        top.set_title(rf"$K={r['K']:.2f}$ rad/$\mu$m")
        top.set_ylim(-0.05, 2.75)
        bot.set_xlabel(r"$x$ ($\mu$m)")
        if j == 0:
            top.set_ylabel(r"exposure $E/E_0$")
            bot.set_ylabel(r"$\Delta n$ ($\times10^{-3}$)")
            top.legend(frameon=False, loc="upper right", ncol=2)
        panel_label(top, "abc"[j])
    savefig(fig, path)
    return True


ABLATION_LABELS = {
    "baseline": "full model", "no_nonlocality": r"no non-locality ($\sigma=0$)",
    "no_diffusion": r"no diffusion ($D_0=0$)", "no_dye_depletion": "no dye depletion",
    "no_saturation": r"no saturation ($\tanh$ linearized)",
}
# Factorial conditions in the order and wording of the supplement table.
FACTORIAL_ORDER = [
    "baseline", "no_monomer_depletion", "no_monomer_depletion_matched", "no_dye_depletion",
    "no_saturation", "only_tanh", "only_tanh_matched", "only_dye", "only_dye_matched",
    "only_monomer", "linear_recording", "linear_slope_matched",
]
FACTORIAL_LABELS = {
    "baseline": "full model", "no_monomer_depletion": "no monomer depletion",
    "no_monomer_depletion_matched": "no monomer depletion, matched",
    "no_dye_depletion": "no dye bleaching", "no_saturation": r"no $\tanh$ saturation",
    "only_tanh": r"only $\tanh$", "only_tanh_matched": r"only $\tanh$, matched",
    "only_dye": "only dye bleaching", "only_dye_matched": "only dye bleaching, matched",
    "only_monomer": "only monomer depletion", "linear_recording": "linear",
    "linear_slope_matched": "linear, slope-matched",
}
RCWA_LABELS = {"unslanted_bragg": "unslanted, Bragg incidence",
               "unslanted_normal": "unslanted, normal incidence",
               "slanted20_bragg": r"$20^\circ$ slant, Bragg incidence"}


def _ablation_bars(results, conditions, path, labels=None):
    from manifest import S1_K_POINTS
    Ks = sorted(round(k, 3) for k in S1_K_POINTS)
    by_key = {}
    for r in results:
        key = (r["config"]["ablation_condition"], round(r["config"]["K_nominal"], 3),
               r["method_id"])
        by_key.setdefault(key, []).append(r)
    fig, ax = new_fig(width="text", height_in=2.4)
    n = len(conditions)
    w = 0.84 / n
    shades = [COLORS["black"], COLORS["blue"], COLORS["bluish_green"], COLORS["vermillion"],
              COLORS["orange"], COLORS["sky_blue"], COLORS["reddish_purple"], "0.35",
              "0.55", "0.75", "#8c6d31", "#bd9e39"]
    hatches = ["", "//", "\\\\", "..", "xx", "--", "++", "oo", "", "//", "\\\\", ".."]
    for i, cond in enumerate(conditions):
        means, los, his = [], [], []
        for K in Ks:
            st = mean_std_median_ci95([g for _, g in paired_gain(
                by_key.get((cond, K, "MIL"), []), by_key.get((cond, K, "BSGD"), []), key="psnr")])
            means.append(st["mean"] or 0.0)
            los.append(st["ci95_lo"] if st["ci95_lo"] is not None else st["mean"] or 0.0)
            his.append(st["ci95_hi"] if st["ci95_hi"] is not None else st["mean"] or 0.0)
        xs = [j + (i - (n - 1) / 2) * w for j in range(len(Ks))]
        ax.bar(xs, means, width=w, color=shades[i % len(shades)], hatch=hatches[i % len(hatches)],
               edgecolor="white", linewidth=0.3, yerr=_ci_err(means, los, his), capsize=1.2,
               error_kw=dict(lw=0.5), label=(labels or {}).get(cond, cond.replace("_", " ")))
    ax.axhline(0, color=COLORS["black"], lw=0.5)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([rf"$K={k:.2f}$ rad/$\mu$m" for k in Ks])
    ax.set_ylabel(GAIN_LABEL)
    ax.legend(frameon=False, fontsize=5.5, ncol=3 if n > 6 else 2, loc="upper right")
    savefig(fig, path)


def make_figS2_ablation():
    """One mechanism removed at a time (tier S1), budget 2x, three seeds."""
    from manifest import S1_CONDITIONS
    path = _out("FigS2_ablation.pdf")
    results = [r for r in load_all_results() if r["experiment_id"] == "S1"]
    if not results:
        no_data_placeholder(path, "Fig. S2: single-mechanism ablation", "needs tier S1 results.",
                            width="double")
        return False
    _ablation_bars(results, list(S1_CONDITIONS.keys()), path, ABLATION_LABELS)
    return True


def make_figS3_factorial():
    """2^3 factorial over the saturating mechanisms plus operating-point-
    matched and slope-matched linear controls (tier S1X), budget 2x."""
    from manifest import S1X_CONDITIONS
    path = _out("FigS3_factorial.pdf")
    results = [r for r in load_all_results() if r["experiment_id"] == "S1X"]
    if not results:
        no_data_placeholder(path, "Fig. S3: mechanism factorial", "needs tier S1X results.",
                            width="double")
        return False
    order = [c for c in FACTORIAL_ORDER if c in S1X_CONDITIONS]
    _ablation_bars(results, order, path, FACTORIAL_LABELS)
    return True


def make_figS4_rcwa_envelope():
    """|eta_Kogelnik - eta_RCWA| over the 90-case grid, one series per
    geometry."""
    path = _out("FigS4_rcwa_envelope.pdf")
    d = _load_json("results_rcwa_e7.json")
    if not d or not d.get("cases"):
        no_data_placeholder(path, "Fig. S4: Kogelnik vs RCWA", "needs results_rcwa_e7.json "
                            "(experiments/rcwa_crosscheck.py).")
        return False
    by_geom = {}
    for c in d["cases"]:
        by_geom.setdefault(c["geometry"], []).append((c["K"], c["abs_deviation"]))
    fig, ax = new_fig(width="single")
    for (geom, pts), col, mk in zip(by_geom.items(),
                                    [COLORS["blue"], COLORS["vermillion"], COLORS["bluish_green"]],
                                    ["o", "s", "^"]):
        pts = sorted(pts)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, s=9,
                   alpha=0.8, label=RCWA_LABELS.get(geom, geom))
    ax.set_xlabel(K_LABEL)
    ax.set_ylabel(r"$|\eta_{\mathrm{Kogelnik}}-\eta_{\mathrm{RCWA}}|$")
    ax.legend(frameon=False)
    savefig(fig, path)
    return True


HELD_KBLEACH_PANELS = [  # file, panel title, x-axis label, log-x
    ("bruder2017_growth_dn_K8.98_sim.csv", "Bayfol HX, source kinetic model",
     r"exposure dose (mJ/cm$^2$)", True),
    ("bruder2017_growth_dn_K8.98_exp.csv", "Bayfol HX, measured",
     r"exposure dose (mJ/cm$^2$)", True),
    ("hsieh2022_growth_dn_K24.94.csv", "PQ/PMMA, source kinetic model",
     "exposure time (s)", False),
]


def make_figS5_twin_fits_held_kbleach():
    """Twin fits with only (kappa, dn_max) free and k_bleach held at 0.2,
    best of ten starts, for every digitized curve (Bayfol HX and the
    out-of-range PQ/PMMA curve). Data: experiments/fit_literature_curves.py."""
    path = _out("FigS5_twin_fits_held_kbleach.pdf")
    d = _load_json("results_literature_fit.json")
    fits = {f["file"]: f for f in (d or {}).get("fits", [])}
    if not fits:
        no_data_placeholder(path, "Fig. S5: twin fits, k_bleach held",
                            "needs results_literature_fit.json "
                            "(experiments/fit_literature_curves.py).", width="double")
        return False
    panels = [p for p in HELD_KBLEACH_PANELS if p[0] in fits]
    fig, axes = new_fig(width="text", height_in=2.1, ncols=len(panels), squeeze=False)
    for j, (fname, title, xlabel, logx) in enumerate(panels):
        f, ax = fits[fname], axes[0][j]
        order = np.argsort(f["x"])
        x = np.array(f["x"])[order]
        ax.plot(x, 1e3 * np.array(f["y_data"])[order], ls="none", marker="s", ms=3.5,
                color=COLORS["vermillion"], label="digitized", zorder=3)
        ax.plot(x, 1e3 * np.array(f["y_model"])[order], color=COLORS["blue"], marker="o",
                ms=2.5, label="twin fit")
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"{title}\n" + rf"$K={f['K']:.2f}$ rad/$\mu$m, NRMSE {f['nrmse']:.2f}")
        ax.set_xlabel(xlabel)
        if j == 0:
            ax.set_ylabel(r"$\Delta n_1$ ($\times10^{-3}$)")
            ax.legend(frameon=False, loc="lower right")
        panel_label(ax, "abc"[j])
    savefig(fig, path)
    return True


ALL_FIGURES = [
    make_fig1_paired_gain, make_fig2_reconstructions, make_fig3_slant_mismatch,
    make_fig4_twin_validation,
    make_figS1_exposure_profiles, make_figS2_ablation, make_figS3_factorial,
    make_figS4_rcwa_envelope, make_figS5_twin_fits_held_kbleach,
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_real = 0
    for fn in ALL_FIGURES:
        is_real = fn()
        print(f"[make_all] {fn.__name__}: {'ok' if is_real else 'PLACEHOLDER (data missing)'}")
        n_real += bool(is_real)
    print(f"[make_all] {n_real}/{len(ALL_FIGURES)} figures rendered from data -> {OUT_DIR}")
    return 0 if n_real == len(ALL_FIGURES) else 1


if __name__ == "__main__":
    sys.exit(main())
