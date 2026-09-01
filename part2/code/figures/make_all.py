"""
Figures, deterministic from results JSONs. No manual data entry.

Numbering follows the V1-V3 (validation) / M1-M2 (main) / S1-S2
(supporting) tier structure -- see the figure-remapping table agreed with
the user (not stored in-repo; see PR/commit history) for the old E1-E7
scheme's F1-F8(a-e) -> this F1-F9(a-c) mapping and the rationale for each
change. Summary:
  F1        pipeline schematic (illustrative, no data dependency)
  F2        V1: twin vs. digitized literature curves
  F3a       V3: Kogelnik vs. RCWA validity envelope (real data)
  F3b       V2: Kogelnik/BPM/RCWA 3-way regime map (blocked -- no BPM-leg
            runner yet; kept as its OWN figure, not merged into F3a, so a
            missing V2 doesn't silently hide inside F3a's real V3 data)
  F4        M1/M2: paired gain vs K, panel (a) iteration-matched (M1),
            panel (b) compute-matched (M2) -- panel layout deferred until
            M1 data exists (see the run-order agreement); still a
            single-panel placeholder-or-M1-only render until then
  F4b       M1: GS, LPC and SAT (saturation-only surrogate) paired gain
            over BSGD, alongside MIL, at budget=2x -- the baselines Sec.
            4.1 defines but Sec. 5 originally never plotted (work-spec
            item D.2), plus the cheap-surrogate control
  F5        M1: observed K* vs predicted Kc scatter
  F6        M3: cliff-location shift (M2-M1) per budget, per-seed band
  F7        S1: physics-component ablation
  F8        S2: cliff-location sensitivity band under NPDD parameter error
  F10       S3: twin-MISCALIBRATION robustness -- gain when the exposure is
            designed at theta_nominal and recorded at theta'. Distinct
            from F8/S2, which lets both arms re-optimize on the perturbed
            medium (so the perturbation cancels in the paired difference
            and no mismatch is ever tested). Numbered F10, not F9d: the
            F9a-c block is the supplementary-data group and this is a
            main-text result.
  F9a-c     supplementary, real data already committed (gradient-pathway
            ablation, GPU mesh convergence, GPU wavelength detuning)
  R1        reconstruction figure (work-spec item D.1): target vs.
            media-blind SGD vs. media-in-the-loop 1D profile overlays
            plus error, at M2's three shared K points, budget=2x

Old F2/F3 (exposure/reconstruction panels) and F8e (shrinkage-prelim,
seed-bug-era data) are retired, not renumbered -- see the figure-remapping
discussion for why (schema gap and superseded-data respectively). Old F6
(sigma probe) is also retired -- it depended on E2, which does not exist
under the V/M/S tier structure.

Every figure whose data doesn't exist yet correctly emits a "NOT YET
AVAILABLE" placeholder PDF (figures.style.no_data_placeholder) stating
exactly what's missing, rather than fabricating a plot (ground rule 1).

Usage: python -m figures.make_all
"""
from __future__ import annotations
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from style import (new_fig, savefig, no_data_placeholder, COLORS,
                   METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS,
                   BUDGET_LINESTYLES, BUDGET_MARKERS,
                   SINGLE_COL_IN, DOUBLE_COL_IN)
from analysis.aggregate import (load_all_results as _load_all_results_raw,
                                group_by_config,
                                headroom_closure, gain_curve, BUDGETS,
                                mean_std_median_ci95, paired_gain,
                                gain_vs_bsgd_seed_mean)
import run_manifest as rm

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "paper")
RESULTS_ROOT = os.path.join(HERE, "..", "results")


def load_all_results():
    """Wraps analysis.aggregate.load_all_results, but always reads from
    run_manifest.RESULTS_ROOT (the single mutable global tests already
    redirect via run_manifest.set_results_root()) rather than
    aggregate.py's own separate, independently-hardcoded RESULTS_ROOT
    constant. Without this, rm.set_results_root(tmp_dir) in a test had NO
    effect on what make_all.py's figure functions actually read -- they
    always read the real repo results/ tree regardless, which silently
    made test isolation for figures non-functional (caught via a
    regression test that fixed a broken byte-size heuristic and then
    found the figure it was "testing" wasn't even seeing the test data)."""
    return _load_all_results_raw(rm.RESULTS_ROOT)


def _load_json(*parts):
    path = os.path.join(HERE, "..", *parts)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------- F1
def make_F1_pipeline_schematic():
    """Twin -> readout -> loss -> gradient back to exposure, annotated
    with where each tier sits relative to this loop: V1-V3 validate the
    twin/readout themselves (upstream, gate everything below), M1-M2 run
    this exact loop (the main cliff/budget sweep, two arms), S1-S2 perturb
    the twin's physics/parameters and re-run the same loop. Purely
    illustrative (no data dependency)."""
    fig, ax = new_fig(width="double", height_in=SINGLE_COL_IN * 0.75)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [("Exposure\nE(x)", 0.5), ("NPDD Twin\n(recording)", 2.7),
             ("BPM Readout\n(Kogelnik-consistent)", 5.1), ("Loss vs.\nTarget", 7.7)]
    for label, x in boxes:
        ax.add_patch(mpatches.Rectangle((x, 1.7), 1.6, 1.0, fill=False,
                                        edgecolor=COLORS["black"], linewidth=1.0))
        ax.text(x + 0.8, 2.2, label, ha="center", va="center", fontsize=6.5)
    for x0, x1 in [(2.1, 2.7), (4.3, 5.1), (6.7, 7.7)]:
        ax.annotate("", xy=(x1, 2.2), xytext=(x0, 2.2),
                   arrowprops=dict(arrowstyle="->", color=COLORS["black"]))
    ax.annotate("", xy=(0.5, 1.2), xytext=(9.3, 1.2),
               arrowprops=dict(arrowstyle="->", color=COLORS["blue"], linewidth=1.2))
    ax.text(5.0, 0.9, r"$\partial L / \partial E$  (unrolled autodiff through the twin)",
            ha="center", va="center", fontsize=6.5, color=COLORS["blue"])
    ax.text(1.3, 3.4, "V1-V3: validate the twin + readout above\n"
                      "(gates everything below)",
            ha="center", va="center", fontsize=6, color=COLORS["vermillion"])
    ax.text(5.0, 3.4, "M1-M2: run this loop twice --\nmedia-aware vs. media-unaware arm",
            ha="center", va="center", fontsize=6, color=COLORS["bluish_green"])
    ax.text(8.5, 3.4, "S1-S2: perturb the twin's physics/\nparameters, re-run this loop",
            ha="center", va="center", fontsize=6, color=COLORS["orange"])
    ax.set_title("F1: media-in-the-loop optimization pipeline (schematic)", fontsize=7.5)
    savefig(fig, os.path.join(OUT_DIR, "F1_pipeline_schematic.pdf"))
    return True


# --------------------------------------------------------------------- F2 (V1)
def make_F2_twin_validation():
    """One panel per digitized-and-fit literature curve: data points vs.
    the twin's fitted model curve, NRMSE and fit_quality annotated. Shows
    every fitted curve, not just the good ones (ground rule: never show
    only the good fits). Fit against {kappa, dn_max} (Gate-A finding: the
    original {kappa, D0} parameterization was fitting the wrong second
    knob -- see docs/parameter_provenance.md). The two in-regime Bayfol HX
    sources (K=8.98 rad/um, inside this paper's tested K range) land under
    NRMSE 0.3; the PQ/PMMA source (K=24.94, outside the tested range, and
    a still-rising curve that can't actually pin down dn_max) is shown too
    but is not treated as in-regime evidence -- see Section 6 and
    docs/parameter_provenance.md."""
    fit_data = _load_json("results_literature_fit.json")
    if not fit_data or not fit_data.get("fits"):
        csvs_exist = os.path.isdir(os.path.join(HERE, "..", "data", "literature")) and any(
            f.endswith(".csv") for f in os.listdir(os.path.join(HERE, "..", "data", "literature")))
        no_data_placeholder(
            os.path.join(OUT_DIR, "F2_twin_validation.pdf"),
            "F2 (V1): twin vs. digitized literature (growth curve, angular selectivity)",
            "results_literature_fit.json missing or empty -- run "
            "experiments/fit_literature_curves.py" if csvs_exist else
            "needs digitized CSVs (data/literature/*.csv) from WebPlotDigitizer -- "
            "not yet provided; see data/literature/README.md.")
        return False

    fits = fit_data["fits"]
    n = len(fits)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = new_fig(width="double" if n > 1 else "single",
                        height_in=2.6 * nrows, ncols=ncols, nrows=nrows, squeeze=False)
    axes_flat = axes.flatten()

    for ax, fit in zip(axes_flat, fits):
        x = fit["x"]
        y_data = fit["y_data"]
        y_model = fit["y_model"]
        order = np.argsort(x)
        x_sorted = np.array(x)[order]
        ax.scatter(x, y_data, color=COLORS["vermillion"], marker="s",
                  s=14, label="literature", zorder=3)
        ax.plot(x_sorted, np.array(y_model)[order], color=COLORS["blue"],
               marker="o", markersize=3, linewidth=1.0, label="twin (fit)", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("dose or exposure (source units)")
        ax.set_ylabel(r"$|\Delta n_1|$ or DE")
        quality = fit["fit_quality"]
        q_color = {"GOOD": COLORS["bluish_green"], "MODERATE": COLORS["orange"],
                  "POOR": COLORS["vermillion"]}[quality]
        title = fit["file"].replace(".csv", "")
        ax.set_title(f"{title}\nK={fit['K']:.2f} rad/um  NRMSE={fit['nrmse']:.2f} ({quality})",
                    fontsize=6, color=q_color)
        ax.legend(frameon=False, fontsize=5.5, loc="best")

    for ax in axes_flat[n:]:
        ax.axis("off")

    savefig(fig, os.path.join(OUT_DIR, "F2_twin_validation.pdf"))
    return True


# --------------------------------------------------------------------- F3a/F3b (V3/V2)
def make_F3a_rcwa_validity_envelope():
    """V3: Kogelnik vs. RCWA -- real data, already committed. Kept as its
    own figure (not merged with F3b/V2) so a missing V2 doesn't hide
    behind this figure's real content."""
    d3 = _load_json("results_rcwa.json")
    d90 = _load_json("results_rcwa_e7.json")
    if d3 is None and d90 is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F3a_rcwa_validity_envelope.pdf"),
                            "F3a (V3): Kogelnik vs. RCWA validity envelope",
                            "results_rcwa*.json missing")
        return False
    fig, ax = new_fig(width="single")
    if d90:
        by_geom = {}
        for c in d90["cases"]:
            by_geom.setdefault(c["geometry"], []).append((c["K"], c["abs_deviation"]))
        colors = [COLORS["blue"], COLORS["vermillion"], COLORS["bluish_green"]]
        markers = ["o", "s", "^"]
        for (geom, pts), color, marker in zip(by_geom.items(), colors, markers):
            pts = sorted(pts)
            Ks = [p[0] for p in pts]
            devs = [p[1] for p in pts]
            ax.scatter(Ks, devs, color=color, marker=marker, s=8, label=geom, alpha=0.8)
        ax.set_xlabel("K (rad/um)"); ax.set_ylabel("|Kogelnik - RCWA T1|")
        ax.legend(frameon=False, fontsize=5.5)
    savefig(fig, os.path.join(OUT_DIR, "F3a_rcwa_validity_envelope.pdf"))
    return True


def make_F3b_regime_map():
    """V2: genuine 3-way Kogelnik/BPM/RCWA regime map. Blocked -- needs a
    BPM-leg runner that doesn't exist yet (rcwa_crosscheck.py's grid only
    compares Kogelnik vs RCWA). This is its own figure, deliberately not
    folded into F3a, so this gap stays visible rather than silently
    absent from a figure that otherwise looks complete."""
    no_data_placeholder(
        os.path.join(OUT_DIR, "F3b_regime_map.pdf"),
        "F3b (V2): Kogelnik/BPM/RCWA 3-way regime map",
        "needs a genuine 3-way comparison runner (Kogelnik closed-form vs. "
        "SlabBPM split-step vs. RCWA) -- does not exist yet; "
        "experiments/manifest.py's build_V2_jobs has the job configs but "
        "no execution path through run_job() yet. See manifest.py's "
        "VALIDATION_BUILDERS docstring.")
    return False


# --------------------------------------------------------------------- F4/F5 (M1/M2)
def make_F4_headline_gain_vs_K():
    """M1 (iteration-matched) is rendered now; the M2 (compute-matched)
    second panel and final two-panel layout are deferred until M1 data
    exists to design around (agreed run-order: look at M1 before
    committing to a panel layout for the headline figure)."""
    grouped = group_by_config(load_all_results())
    closure = headroom_closure(grouped, "M1", budgets=BUDGETS)
    if all(r.get("status") == "no_data" for r in closure):
        no_data_placeholder(
            os.path.join(OUT_DIR, "F4_headline_gain_vs_K.pdf"),
            "F4 (M1/M2): paired gain (MIL-BSGD) vs K, panel (a) M1 "
            "iteration-matched / panel (b) M2 compute-matched, one curve "
            "per budget, 95% CI bands",
            "needs M1 manifest results across all 3 budgets. Panel (b) "
            "(M2) and the final 2-panel layout are deferred until M1 "
            "data exists to design around -- see the run-order agreement.")
        return False
    _render_F4(closure)
    return True


def _render_F4(closure):
    fig, ax = new_fig(width="single")
    for row, budget in zip(closure, BUDGETS):
        if row.get("status") == "no_data":
            continue
        curve = row["gain_curve"]
        Ks = [c[0] for c in curve]
        means = [c[1] for c in curve]
        los = [c[2] for c in curve]
        his = [c[3] for c in curve]
        color = {2.0: COLORS["blue"], 4.0: COLORS["vermillion"], 8.0: COLORS["bluish_green"]}[budget]
        ax.plot(Ks, means, marker=BUDGET_MARKERS[budget], ls=BUDGET_LINESTYLES[budget],
                color=color, ms=2.5, label=f"budget={budget:.0f}x")
        ax.fill_between(Ks, los, his, color=color, alpha=0.2, linewidth=0)
        if row.get("predicted_Kc_from_measured_C") is not None:
            ax.axvline(row["predicted_Kc_from_measured_C"], color=color, ls="--", lw=0.7)
    ax.axhline(0, color=COLORS["black"], lw=0.5)
    ax.set_xlabel("K (rad/um)"); ax.set_ylabel("paired gain MIL-BSGD (dB)")
    ax.legend(frameon=False)
    savefig(fig, os.path.join(OUT_DIR, "F4_headline_gain_vs_K.pdf"))


# --------------------------------------------------------------------- F4b
def make_F4b_baseline_comparison():
    """GS and LPC vs. BSGD, alongside MIL vs. BSGD, at budget=2x -- Sec.
    4.1 defines all three non-oracle methods but the original Results
    section only ever plotted MIL. MIL uses gain_curve (proper per-seed
    paired comparison, real CI). GS/LPC use gain_vs_bsgd_seed_mean instead
    of gain_curve -- checked: GS/LPC are logged at seed=0 only
    (deterministic, "closed-form, no optimizer" per Sec. 4.1), and
    gain_curve's seed-matched pairing would compare them against BSGD's
    single seed=0 draw rather than BSGD's seed-mean, which produced a
    visibly noisy, non-physical curve on the real data before this fix."""
    grouped = group_by_config(load_all_results())
    budget = 2.0
    curves = {"MIL": gain_curve(grouped, "M1", budget, method="MIL"),
              # SAT is a seeded optimizer like MIL, so it gets the proper
              # per-seed paired comparison with a real CI -- unlike GS/LPC
              # below, which are closed-form single-draw methods.
              "SAT": gain_curve(grouped, "M1", budget, method="SAT")}
    for m in ("GS", "LPC"):
        curves[m] = gain_vs_bsgd_seed_mean(grouped, "M1", budget, method=m)
    if all(len(c) == 0 for c in curves.values()):
        no_data_placeholder(
            os.path.join(OUT_DIR, "F4b_baseline_comparison.pdf"),
            "F4b (M1): GS/LPC/SAT/MIL paired gain over BSGD, budget=2x",
            "needs M1 manifest results for GS/LPC/SAT/MIL at budget=2x.")
        return False

    fig, ax = new_fig(width="single")
    for method in ("GS", "LPC", "SAT", "MIL"):
        curve = curves[method]
        if not curve:
            continue
        Ks = [c[0] for c in curve]
        means = [c[1] for c in curve]
        los = [c[2] for c in curve]
        his = [c[3] for c in curve]
        ax.plot(Ks, means, marker=METHOD_MARKERS[method], ls="-",
                color=METHOD_COLORS[method], ms=2.5, label=METHOD_LABELS[method])
        if all(lo is not None for lo in los):
            ax.fill_between(Ks, los, his, color=METHOD_COLORS[method], alpha=0.15, linewidth=0)
    ax.axhline(0, color=COLORS["black"], lw=0.5)
    ax.set_xlabel("K (rad/um)")
    ax.set_ylabel("paired gain over media-blind SGD (dB)\nbudget = 2x")
    ax.legend(frameon=False, fontsize=6)
    savefig(fig, os.path.join(OUT_DIR, "F4b_baseline_comparison.pdf"))
    return True


def make_F5_Kstar_vs_Kc_scatter():
    grouped = group_by_config(load_all_results())
    closure = headroom_closure(grouped, "M1", budgets=BUDGETS)
    valid = [r for r in closure if r.get("status") != "no_data"]
    if not valid:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F5_Kstar_vs_Kc_scatter.pdf"),
            "F5 (M1): observed K* vs predicted Kc (both estimators), y=x line",
            "needs M1 headroom-closure results.")
        return False
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
    return True


# --------------------------------------------------------------------- F6 (M3)
def make_F6_cliff_shift():
    """M3: comparison of the M1 (iteration-matched) and M2 (compute-
    matched) arms at M2's 3 shared K-points, per budget.

    NOT a shift-in-K* plot: analysis.aggregate.m3_cliff_shift's
    point_estimate_shift is structurally None here, because find_*_crossing_K
    require an actual sign change and NEITHER arm's gain curve has one (see
    F4 -- gain is positive at every K, every budget, in both arms). That is
    not missing data to wait on, it IS the confound-control result: paired
    gain at M2's matched K's, side by side with M1's gain at the same K's,
    shows the compute-matched arm doesn't change the story -- MIL's
    advantage isn't an artifact of more optimizer iterations."""
    grouped = group_by_config(load_all_results())
    m1_curve = gain_curve(grouped, "M1", BUDGETS[0])
    m2_curve = gain_curve(grouped, "M2", BUDGETS[0])
    if not m1_curve or not m2_curve:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F6_cliff_shift.pdf"),
            "F6 (M3): M1 vs M2 paired gain at M2's shared K-points, per budget",
            "needs M1+M2 manifest results across all 3 budgets.")
        return False
    _render_F6(grouped)
    return True


def _render_F6(grouped):
    from manifest import S1_K_POINTS
    # M2's 3 K-points (S1_K_POINTS: sub/near/post-cliff) are chosen for
    # physical meaning, not to sit exactly on M1's 15-point grid -- M1's
    # minimum grid K is 1.963, above S1_K_POINTS' sub-cliff point 1.309.
    # So M1 has data at only 2 of the 3 shared-intent K's; bars are placed
    # by K's actual index in the full (sorted) K list, not by array
    # position, so a K missing from one arm just leaves that slot empty
    # rather than misaligning the rest.
    all_Ks = sorted(round(k, 3) for k in S1_K_POINTS)
    fig, axs = new_fig(width="double", ncols=3, height_in=SINGLE_COL_IN * 0.7)
    for ax, budget in zip(axs, BUDGETS):
        m1_curve = {round(k, 3): (g, lo, hi) for k, g, lo, hi in gain_curve(grouped, "M1", budget)}
        m2_curve = {round(k, 3): (g, lo, hi) for k, g, lo, hi in gain_curve(grouped, "M2", budget)}
        w = 0.35
        for label, curve, dx, color, hatch in [
            ("M1 (iter-matched)", m1_curve, -w/2, COLORS["blue"], None),
            ("M2 (compute-matched)", m2_curve, w/2, COLORS["vermillion"], "//"),
        ]:
            idx = [i for i, k in enumerate(all_Ks) if k in curve]
            if not idx:
                continue
            g = [curve[all_Ks[i]][0] for i in idx]
            lo = [curve[all_Ks[i]][1] for i in idx]
            hi = [curve[all_Ks[i]][2] for i in idx]
            ax.bar([i + dx for i in idx], g,
                  yerr=[[a - b for a, b in zip(g, lo)], [b - a for a, b in zip(g, hi)]],
                  width=w, color=color, hatch=hatch, label=label, capsize=2)
        ax.axhline(0, color=COLORS["black"], lw=0.5)
        ax.set_xticks(range(len(all_Ks)))
        ax.set_xticklabels([f"{k:.2f}" for k in all_Ks], fontsize=5.5)
        ax.set_title(f"budget={budget:.0f}x", fontsize=6.5)
        ax.set_xlabel("K (rad/um)", fontsize=6)
        if budget == BUDGETS[0]:
            ax.set_ylabel("paired gain MIL-BSGD (dB)")
            ax.legend(frameon=False, fontsize=5)
    savefig(fig, os.path.join(OUT_DIR, "F6_cliff_shift.pdf"))


# --------------------------------------------------------------------- F7 (S1)
def make_F7_physics_ablation():
    """S1: physics-component ablation (sigma/D0/k_bleach/dn_max toggles)
    at sub/near/post-cliff K. Not to be confused with F9a's gradient-
    pathway ablation (an engineering question, not a physics one) --
    see docs/legacy_results_audit.md for the naming-collision this
    guards against."""
    results = [r for r in load_all_results() if r["experiment_id"] == "S1"]
    if not results:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F7_physics_ablation.pdf"),
            "F7 (S1): physics-component ablation (no_nonlocality/"
            "no_diffusion/no_dye_depletion/no_saturation_approx vs. baseline)",
            "needs S1 manifest results.")
        return False
    _render_F7(results)
    return True


def _render_F7(results):
    from manifest import S1_CONDITIONS, S1_K_POINTS
    conditions = list(S1_CONDITIONS.keys())  # baseline first, by construction
    Ks = sorted(round(k, 3) for k in S1_K_POINTS)
    by_key = {}
    for r in results:
        key = (r["config"]["ablation_condition"], round(r["config"]["K_nominal"], 3), r["method_id"])
        by_key.setdefault(key, []).append(r)

    fig, ax = new_fig(width="double", height_in=SINGLE_COL_IN * 0.7)
    n_cond = len(conditions)
    w = 0.8 / n_cond
    colors = [COLORS["black"], COLORS["blue"], COLORS["bluish_green"],
             COLORS["vermillion"], COLORS["orange"]]
    for i, cond in enumerate(conditions):
        means, los, his = [], [], []
        for K in Ks:
            mil = by_key.get((cond, K, "MIL"), [])
            bsgd = by_key.get((cond, K, "BSGD"), [])
            pairs = paired_gain(mil, bsgd, key="psnr")
            stat = mean_std_median_ci95([g for _, g in pairs])
            means.append(stat["mean"] if stat["mean"] is not None else 0.0)
            los.append(stat["ci95_lo"] if stat["ci95_lo"] is not None else 0.0)
            his.append(stat["ci95_hi"] if stat["ci95_hi"] is not None else 0.0)
        x = [j + (i - (n_cond - 1) / 2) * w for j in range(len(Ks))]
        yerr = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
        ax.bar(x, means, width=w, color=colors[i % len(colors)],
              label=cond.replace("_", " "), yerr=yerr, capsize=1.5)
    ax.axhline(0, color=COLORS["black"], lw=0.5)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"{k:.2f}" for k in Ks])
    ax.set_xlabel("K (rad/um) [sub-/near-/post-cliff]")
    ax.set_ylabel("paired gain MIL-BSGD (dB)")
    ax.legend(frameon=False, fontsize=5, ncol=2)
    savefig(fig, os.path.join(OUT_DIR, "F7_physics_ablation.pdf"))


def _k_averaged_seed_gains(by_key, key_prefix, all_K):
    """Per-seed, K-averaged paired gain (MIL - BSGD).

    Pairs within each K (paired_gain is seed-keyed and silently collapses
    repeated seeds if handed several K at once), then averages each
    seed's gains over K. Returns one value per seed, which is the unit
    the 95% CI should be taken over -- between-K spread is a real effect
    being averaged, not uncertainty about it.
    """
    per_seed = {}
    for K in all_K:
        for seed, g in paired_gain(by_key.get(key_prefix + (K, "MIL"), []),
                                   by_key.get(key_prefix + (K, "BSGD"), []),
                                   key="psnr"):
            per_seed.setdefault(seed, []).append(g)
    return [sum(v) / len(v) for v in per_seed.values() if v]


# --------------------------------------------------------------------- F8 (S2)
def make_F8_sensitivity_band():
    """S2: paired-gain sensitivity to +/-NPDD-parameter error, K-averaged
    over the 4 collapse-region K-points, one curve per parameter (D0,
    sigma, kappa). NOTE title/framing changed from the original "cliff-
    location sensitivity band" -- there is no cliff-location K* to band
    (M1/M2 found no zero-crossing anywhere, see F4/F6), so a K*-vs-
    perturbation plot has nothing to show. Gain-vs-perturbation is the
    honest version of this question given that finding, and it answers
    the same practical concern (is the result robust to getting D0/sigma/
    kappa wrong). See docs/s2_sensitivity_notes.md for the two write-up
    caveats (one-at-a-time perturbation, no interaction terms;
    perturbation magnitudes need justifying against the literature spread
    once V1 exists) -- carry those into the caption/discussion."""
    results = [r for r in load_all_results() if r["experiment_id"] == "S2"]
    if not results:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F8_sensitivity_band.pdf"),
            "F8 (S2): paired-gain sensitivity to NPDD parameter error",
            "needs S2 manifest results.")
        return False
    _render_F8(results)
    return True


def _render_F8(results):
    from manifest import S2_PARAMS
    # K is part of the key: paired_gain matches arms by seed alone, so
    # pooling K points into one bucket before pairing would compare MIL
    # at one spatial frequency against BSGD at another (its seed->value
    # dict keeps only the last K per seed). Pair within K, pool after.
    by_key = {}
    for r in results:
        key = (r["config"]["sensitivity_param"], r["config"]["sensitivity_pct"],
               r["config"]["K_nominal"], r["method_id"])
        by_key.setdefault(key, []).append(r)
    all_K = sorted({r["config"]["K_nominal"] for r in results})
    # baseline (pct=0) is only tagged under S2_PARAMS[0] by construction
    # (build_S2_jobs emits it once, not once per param) -- shared across
    # all three curves below.
    baseline_param = S2_PARAMS[0]
    pcts = sorted({r["config"]["sensitivity_pct"] for r in results} | {0})

    fig, ax = new_fig(width="single")
    colors = {p: c for p, c in zip(S2_PARAMS, [COLORS["blue"], COLORS["vermillion"], COLORS["bluish_green"]])}
    markers = {p: m for p, m in zip(S2_PARAMS, ["o", "s", "^"])}
    for param in S2_PARAMS:
        means, los, his, xs = [], [], [], []
        for pct in pcts:
            lookup_param = baseline_param if pct == 0 else param
            gains = _k_averaged_seed_gains(by_key, (lookup_param, pct), all_K)
            if not gains:
                continue
            stat = mean_std_median_ci95(gains)
            xs.append(pct); means.append(stat["mean"])
            los.append(stat["ci95_lo"]); his.append(stat["ci95_hi"])
        ax.plot(xs, means, marker=markers[param], color=colors[param], ms=3, label=param)
        ax.fill_between(xs, los, his, color=colors[param], alpha=0.15, linewidth=0)
    ax.axvline(0, color=COLORS["black"], lw=0.5, ls=":")
    ax.set_xlabel("parameter perturbation (%)")
    ax.set_ylabel("paired gain MIL$-$BSGD (dB)\n(K-averaged, 95% CI over seeds)")
    ax.legend(frameon=False, title="perturbed param")
    savefig(fig, os.path.join(OUT_DIR, "F8_sensitivity_band.pdf"))


# --------------------------------------------------------------------- F10 (S3)
def make_F10_twin_mismatch():
    """S3: paired gain when the exposure is DESIGNED against the nominal
    twin and then RECORDED on a miscalibrated one.

    This is the figure that supports a robustness claim; F8/S2 cannot,
    because it re-optimizes both arms on the perturbed medium, keeping
    the perturbation common to both arms -- the exact condition under
    which the paper's own paired-comparison argument says a systematic
    twin error cancels. Here the exposure is frozen at theta_nominal, so
    the error is one-sided and real.

    dn_max is plotted over a wider perturbation range than the other
    three parameters, and that asymmetry is deliberate, not an accident
    of gridding: our own literature fits (Sec. 6) disagree on Bayfol's
    dn_max by 2.7x (about +170%/-63%), far outside +/-50%, so testing
    that parameter only to +/-50% would be testing a range we already
    know from our own data is too narrow.
    """
    results = [r for r in load_all_results() if r["experiment_id"] == "S3"]
    if not results:
        no_data_placeholder(
            os.path.join(OUT_DIR, "F10_twin_mismatch.pdf"),
            "F10 (S3): robustness to twin miscalibration",
            "needs S3 results -- run experiments/run_s3_mismatch.py.")
        return False
    _render_F10(results)
    return True


def _render_F10(results):
    # K in the key for the same reason as F8 above -- see this module's
    # pairing note; paired_gain is seed-keyed and must be called within a
    # single K, never across a pooled set.
    buckets = {}
    for r in results:
        cfg = r["config"]
        key = (cfg.get("mismatch_param"), cfg.get("mismatch_pct"),
               cfg.get("K_nominal"), r["method_id"])
        buckets.setdefault(key, []).append(r)
    all_K = sorted({r["config"]["K_nominal"] for r in results})

    params = sorted({k[0] for k in buckets})
    # pct=0 is emitted under one parameter only (build_S3_conditions), and
    # is the shared origin of every curve.
    nominal_param = next(k[0] for k in buckets if k[1] == 0)

    palette = [COLORS["blue"], COLORS["vermillion"], COLORS["bluish_green"],
               COLORS["reddish_purple"]]
    markers = ["o", "s", "^", "D"]
    colors = {p: c for p, c in zip(params, palette)}
    marks = {p: m for p, m in zip(params, markers)}

    fig, ax = new_fig(width="single")
    for param in params:
        pcts = sorted({k[1] for k in buckets if k[0] == param} | {0})
        xs, means, los, his = [], [], [], []
        for pct in pcts:
            lookup = nominal_param if pct == 0 else param
            gains = _k_averaged_seed_gains(buckets, (lookup, pct), all_K)
            if not gains:
                continue
            stat = mean_std_median_ci95(gains)
            xs.append(pct); means.append(stat["mean"])
            los.append(stat["ci95_lo"]); his.append(stat["ci95_hi"])
        if not xs:
            continue
        ax.plot(xs, means, marker=marks[param], color=colors[param], ms=3,
                label=param)
        ax.fill_between(xs, los, his, color=colors[param], alpha=0.15,
                        linewidth=0)

    ax.axhline(0, color=COLORS["black"], lw=0.7)
    ax.axvline(0, color=COLORS["black"], lw=0.5, ls=":")
    # Mark the +/-50% band the paper's robustness sentence is stated over,
    # so a reader can see at a glance that every curve is flat inside it
    # and that the only excursion is one parameter outside it.
    ax.axvspan(-50, 50, color=COLORS["black"], alpha=0.05, linewidth=0)
    ax.annotate("+/-50% (claim range)", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=5.5, color=COLORS["black"])
    ax.set_xlabel("design-to-record parameter mismatch (%)")
    ax.set_ylabel("paired gain MIL$-$BSGD (dB)\n"
                  "(designed at nominal; K-avg, 95% CI over seeds)")
    ax.legend(frameon=False, title="miscalibrated param", fontsize=6,
              title_fontsize=6)
    savefig(fig, os.path.join(OUT_DIR, "F10_twin_mismatch.pdf"))


# --------------------------------------------------------------------- F9a-c (supplementary, real data)
def make_F9a_gradient_ablation():
    d = _load_json("results_ablation_gradients.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F9a_gradient_ablation.pdf"),
                            "F9a: gradient-pathway ablation", "results_ablation_gradients.json missing")
        return False
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
    savefig(fig, os.path.join(OUT_DIR, "F9a_gradient_ablation.pdf"))
    return True


def make_F9b_mesh_convergence():
    d = _load_json("results", "gpu_reruns", "npdd_mesh_sweep", "results.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F9b_mesh_convergence.pdf"),
                            "F9b: mesh-density convergence", "results/gpu_reruns/npdd_mesh_sweep missing")
        return False
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
    savefig(fig, os.path.join(OUT_DIR, "F9b_mesh_convergence.pdf"))
    return True


def make_F9c_wavelength_detuning():
    d = _load_json("results", "gpu_reruns", "bpm_wavelength_sweep", "results.json")
    if d is None:
        no_data_placeholder(os.path.join(OUT_DIR, "F9c_wavelength_detuning.pdf"),
                            "F9c: wavelength-detuning readout sweep", "results/gpu_reruns/bpm_wavelength_sweep missing")
        return False
    fig, ax = new_fig(width="single")
    lams = sorted(float(k) for k in d["by_wavelength"])
    psnrs = [d["by_wavelength"][str(l)]["psnr"] for l in lams]
    ax.plot([l * 1000 for l in lams], psnrs, "-o", color=COLORS["blue"])
    ax.axvline(d["meta"]["design_lam_um"] * 1000, color=COLORS["vermillion"], ls="--", lw=0.7,
              label="design wavelength")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("single seed, single Colab T4 run", fontsize=6)
    ax.legend(frameon=False)
    savefig(fig, os.path.join(OUT_DIR, "F9c_wavelength_detuning.pdf"))
    return True


# --------------------------------------------------------------------- R1
def make_R1_reconstructions():
    """Target vs. media-blind SGD vs. media-in-the-loop, 1D profile
    overlays (this paper's recording kinetics are 1D in transverse x --
    said plainly in the caption, not presented as a 2D image), plus an
    error profile, at M2's three shared K points, budget=2x. Data from
    experiments/make_r1_reconstructions.py (results_r1_reconstructions.json)."""
    d = _load_json("results_r1_reconstructions.json")
    if d is None or not d.get("results"):
        no_data_placeholder(
            os.path.join(OUT_DIR, "R1_reconstructions.pdf"),
            "R1: target / media-blind SGD / media-in-the-loop 1D profiles + error",
            "needs results_r1_reconstructions.json -- run "
            "experiments/make_r1_reconstructions.py")
        return False

    results = d["results"]
    n = len(results)
    fig, axes = new_fig(width="double", height_in=2.4 * n, ncols=2, nrows=n, squeeze=False)

    for row, r in enumerate(results):
        target = np.array(r["target"])
        recon_bsgd = np.array(r["recon_bsgd"])
        recon_mil = np.array(r["recon_mil"])
        x = np.arange(len(target))

        ax_prof = axes[row][0]
        ax_prof.plot(x, target, color=COLORS["black"], lw=0.8, label="target")
        ax_prof.plot(x, recon_bsgd, color=METHOD_COLORS["BSGD"], lw=0.8,
                    label=f"BSGD ({r['psnr_bsgd']:.1f} dB)")
        ax_prof.plot(x, recon_mil, color=METHOD_COLORS["MIL"], lw=0.8,
                    label=f"MIL ({r['psnr_mil']:.1f} dB)")
        ax_prof.set_ylabel("intensity (a.u.)")
        ax_prof.set_title(f"K={r['K']:.2f} rad/um, budget={r['budget']:.0f}x", fontsize=6.5)
        ax_prof.legend(frameon=False, fontsize=5.5, loc="upper right", ncol=3,
                      bbox_to_anchor=(1.0, 1.18))

        ax_err = axes[row][1]
        ax_err.plot(x, recon_bsgd - target, color=METHOD_COLORS["BSGD"], lw=0.7, label="BSGD error")
        ax_err.plot(x, recon_mil - target, color=METHOD_COLORS["MIL"], lw=0.7, label="MIL error")
        ax_err.axhline(0, color=COLORS["black"], lw=0.4)
        ax_err.set_ylabel("recon - target")
        ax_err.legend(frameon=False, fontsize=5.5, loc="upper right", ncol=2,
                     bbox_to_anchor=(1.0, 1.18))

        if row == n - 1:
            ax_prof.set_xlabel("x (pixels)")
            ax_err.set_xlabel("x (pixels)")

    savefig(fig, os.path.join(OUT_DIR, "R1_reconstructions.pdf"))
    return True


# --------------------------------------------------------------------- R2 (2D)
def make_R2_2d_reconstructions():
    """Bounded 2D study (experiments/run_2d.py, experiments/manifest_2d.py):
    target vs. media-blind SGD vs. media-in-the-loop, actual 2D images, one
    row per target type, budget=4x, seed=0. Data from
    experiments/make_2d_reconstructions.py (results_2d_reconstructions.json).
    The direct answer to "this looks like a grating paper, not a CGH paper"
    -- unlike R1's 1D profiles, these are real (synthetic) 2D images."""
    d = _load_json("results_2d_reconstructions.json")
    if d is None or not d.get("results"):
        no_data_placeholder(
            os.path.join(OUT_DIR, "R2_2d_reconstructions.pdf"),
            "R2: target / media-blind SGD / media-in-the-loop, 2D images",
            "needs results_2d_reconstructions.json -- run "
            "experiments/make_2d_reconstructions.py")
        return False

    by_target = {}
    for r in d["results"]:
        by_target.setdefault(r["target_kind"], {})[r["method_id"]] = r
    order = [k for k in ("disc", "checkerboard", "reschart") if k in by_target]
    n = len(order)

    fig, axes = new_fig(width="double", height_in=2.0 * n, ncols=3, nrows=n, squeeze=False)
    for row, kind in enumerate(order):
        target = np.array(by_target[kind]["BSGD"]["target"])
        bsgd = np.array(by_target[kind]["BSGD"]["recon"])
        mil = np.array(by_target[kind]["MIL"]["recon"])
        vmax = max(target.max(), bsgd.max(), mil.max())
        for col, (label, img) in enumerate([("target", target), ("BSGD", bsgd), ("MIL", mil)]):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=7)
            if col == 0:
                ax.set_ylabel(kind, fontsize=6.5)
    savefig(fig, os.path.join(OUT_DIR, "R2_2d_reconstructions.pdf"))
    return True


# --------------------------------------------------------------------- R3
def make_R3_exposure_profiles():
    """Phase 3 Tier-1 item 3: the mechanism figure this paper was missing --
    not just what the reconstructions look like, but what the method
    actually does. E(x) (optimized exposure) and Delta-n(x) (recorded
    index profile) for media-blind SGD vs. media-in-the-loop, at the same
    three K points as R1, budget=2x, seed=0. Data from
    experiments/make_r1_profiles.py (results_r1_profiles.json)."""
    d = _load_json("results_r1_profiles.json")
    if d is None or not d.get("results"):
        no_data_placeholder(
            os.path.join(OUT_DIR, "R3_exposure_profiles.pdf"),
            "R3: exposure E(x) and index profile dn(x), BSGD vs. MIL",
            "needs results_r1_profiles.json -- run experiments/make_r1_profiles.py")
        return False

    results = d["results"]
    n = len(results)
    fig, axes = new_fig(width="double", height_in=1.9 * n, ncols=2, nrows=n, squeeze=False)

    for row, r in enumerate(results):
        E_bsgd = np.array(r["E_bsgd"]); E_mil = np.array(r["E_mil"])
        dn_bsgd = np.array(r["dn_bsgd"]); dn_mil = np.array(r["dn_mil"])
        x = np.arange(len(E_bsgd))

        ax_E = axes[row][0]
        ax_E.plot(x, E_bsgd, color=METHOD_COLORS["BSGD"], lw=0.8, label="BSGD")
        ax_E.plot(x, E_mil, color=METHOD_COLORS["MIL"], lw=0.8, label="MIL")
        ax_E.set_ylabel("E(x)")
        ax_E.set_title(f"K={r['K']:.2f} rad/um, budget={r['budget']:.0f}x", fontsize=6.5)
        ax_E.legend(frameon=False, fontsize=5.5, loc="upper right")

        ax_dn = axes[row][1]
        ax_dn.plot(x, dn_bsgd, color=METHOD_COLORS["BSGD"], lw=0.8, label="BSGD")
        ax_dn.plot(x, dn_mil, color=METHOD_COLORS["MIL"], lw=0.8, label="MIL")
        ax_dn.set_ylabel(r"$\Delta n(x)$")
        ax_dn.legend(frameon=False, fontsize=5.5, loc="upper right")

        if row == n - 1:
            ax_E.set_xlabel("x (pixels)")
            ax_dn.set_xlabel("x (pixels)")

    savefig(fig, os.path.join(OUT_DIR, "R3_exposure_profiles.pdf"))
    return True


ALL_FIGURES = [
    make_F1_pipeline_schematic,
    make_F2_twin_validation,
    make_F3a_rcwa_validity_envelope, make_F3b_regime_map,
    make_F4_headline_gain_vs_K, make_F4b_baseline_comparison, make_F5_Kstar_vs_Kc_scatter,
    make_F6_cliff_shift, make_F7_physics_ablation, make_F8_sensitivity_band,
    make_F10_twin_mismatch,
    make_F9a_gradient_ablation, make_F9b_mesh_convergence, make_F9c_wavelength_detuning,
    make_R1_reconstructions, make_R2_2d_reconstructions, make_R3_exposure_profiles,
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_real = 0
    for fn in ALL_FIGURES:
        is_real = fn()
        status = "real content" if is_real else "placeholder"
        print(f"[make_all] {fn.__name__} ... {status}")
        n_real += bool(is_real)
    print(f"[make_all] done -- {n_real}/{len(ALL_FIGURES)} figures have real "
         f"content, see {OUT_DIR}")


if __name__ == "__main__":
    main()
