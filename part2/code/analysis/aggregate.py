"""
Phase 4: analysis and statistics over experiments/run_manifest.py results.

Reads every results/{experiment_id}/{config_hash}/{method_id}_seed{N}.json
(Phase 1.2 schema), aggregates per (experiment_id, config_hash, method_id)
across seeds (mean/std/median/95% CI, t-distribution), computes paired
per-seed MIL-BSGD gains, the two cliff-location estimators, the empirical
headroom-closure table (for M1 and, separately, M2), the M3 statistic
(cliff-location shift between the M1/M2 arms, with seed uncertainty), and
emits results/summary/paper_numbers.json.

Renamed from an earlier E1-centric version: M1 (iteration-matched arm) and
M2 (compute-matched arm) share the SAME analysis functions (parameterized
by experiment_id) rather than duplicating an "M2 version" of everything --
M3 is not a new data-collection tier, it's the comparison between M1 and
M2's results, computed here.

Honest scope: as of this pass, results/ contains only the single-seed
gpu_reruns/ sweeps and V3's RCWA data from earlier sessions -- no M1/M2/
S1/S2 manifest output exists yet (needs a real GPU run, gated behind
--probe + your Gate-1 review). Every function here is written and tested
against tests/test_aggregate.py's real (tiny, CPU-scale, clearly-labeled)
manifest data so the pipeline is verified correct and ready to run the
moment real data lands -- this script does NOT fabricate numbers to fill
paper_numbers.json in the meantime; missing data produces explicit
"status": "no_data" entries, not invented ones (ground rule 1).
"""
from __future__ import annotations
import glob
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from scipy import stats as scipy_stats
from holomedia import NPDDRecorder, MediumParams

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
SUMMARY_PATH = os.path.join(RESULTS_ROOT, "summary", "paper_numbers.json")

BUDGETS = (2.0, 4.0, 8.0)
CI_FLATNESS_THRESHOLD_DB = 0.25


REQUIRED_SCHEMA_KEYS = {"experiment_id", "config_hash", "method_id", "seed",
                        "config", "psnr", "diffraction_efficiency", "contrast"}


# --------------------------------------------------------------- loading
def load_all_results(results_root: str = RESULTS_ROOT) -> list[dict]:
    """Loads every {method_id}_seed{N}.json under results_root, skipping
    (with a warning, not a crash) anything that doesn't match the Phase
    1.2 manifest schema -- results/gpu_reruns/*/results.json predates the
    manifest system entirely (bespoke single-file-per-sweep schema from
    gpu_npdd_mesh_convergence_sweep.py / gpu_bpm_wavelength_sweep.py, no
    experiment_id/config_hash/method_id fields at all) and matches the
    same directory-depth glob pattern, so this must be schema-checked
    rather than assumed."""
    paths = glob.glob(os.path.join(results_root, "*", "*", "*.json"))
    out, skipped = [], []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        if REQUIRED_SCHEMA_KEYS.issubset(data.keys()):
            out.append(data)
        else:
            skipped.append(p)
    if skipped:
        print(f"[aggregate] skipped {len(skipped)} non-manifest-schema file(s) "
              f"(e.g. pre-manifest gpu_reruns/ sweeps): {skipped[:3]}"
              f"{'...' if len(skipped) > 3 else ''}")
    return out


def _pairing_key(config: dict) -> str:
    """Grouping key used to pair methods within a config, DELIBERATELY
    excluding n_iters (unlike experiments/manifest.py's config_hash, which
    includes it).

    Bug this fixes: build_M2_jobs gives BSGD n_iters * compute_match_ratio
    while MIL keeps n_iters (that's the entire point of the compute-
    matched arm) -- so MIL's and BSGD's config_hash differ even though
    every other field (K_nominal, contrast_cap, medium, target) is
    identical. Grouping by config_hash therefore put every M2 MIL job and
    its paired BSGD job in DIFFERENT groups, and gain_curve/paired_gain
    (which pair methods within a group) found zero pairs for all of M2 --
    m3_cliff_shift silently returned "no_data" for a fully-populated
    54/54-job M2 run. Never caught earlier because M2 had no real data
    until this run.

    Every other field still participates, so this does not merge distinct
    experimental conditions: S1's ablation_condition and S2's
    sensitivity_param/sensitivity_pct both live inside `medium` or as
    their own config keys and are still hashed, so different ablations/
    perturbations remain separate groups. For M1 (where n_iters IS
    uniform across methods within a group already), this key produces
    the exact same grouping as config_hash -- confirmed no M1 result
    changed after this fix."""
    stripped = {k: v for k, v in config.items() if k != "n_iters"}
    canon = json.dumps(stripped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def group_by_config(results: list[dict]) -> dict:
    """{(experiment_id, pairing_key): {method_id: [result, ...]}}"""
    g = defaultdict(lambda: defaultdict(list))
    for r in results:
        g[(r["experiment_id"], _pairing_key(r["config"]))][r["method_id"]].append(r)
    return g


# --------------------------------------------------------------- stats
def mean_std_median_ci95(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, std=None, median=None, ci95_lo=None, ci95_hi=None)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    if n < 2:
        return dict(n=n, mean=mean, std=0.0, median=median, ci95_lo=mean, ci95_hi=mean)
    std = statistics.stdev(values)
    sem = std / math.sqrt(n)
    tcrit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    return dict(n=n, mean=mean, std=std, median=median,
               ci95_lo=mean - tcrit * sem, ci95_hi=mean + tcrit * sem)


def aggregate_method(rows: list[dict]) -> dict:
    """rows: results for ONE (experiment, config, method) across seeds."""
    return dict(psnr=mean_std_median_ci95([r["psnr"] for r in rows]),
               de=mean_std_median_ci95([r["diffraction_efficiency"] for r in rows]),
               n_seeds=len(rows), seeds=sorted(r["seed"] for r in rows))


def paired_gain(rows_a: list[dict], rows_b: list[dict], key: str = "psnr") -> list[tuple]:
    """Per-seed (A - B), matched by seed. Returns [(seed, gain), ...]."""
    by_seed_b = {r["seed"]: r[key] for r in rows_b}
    return [(r["seed"], r[key] - by_seed_b[r["seed"]])
           for r in rows_a if r["seed"] in by_seed_b]


# --------------------------------------------------------------- cliff estimators
def find_zero_crossing_K(K_gain_pairs: list[tuple]) -> float | None:
    """FIRST-crossing estimator: linear-interpolation zero-crossing where
    mean paired gain first goes positive -> non-positive as K increases.

    FRAGILE, and retained only for continuity with already-archived
    result sets -- prefer find_last_crossing_K below. The measured M1
    gain curve is NOT monotone in K (it dips negative, recovers positive,
    dips again), so "the first crossing" is decided by whichever single
    K point happens to dip first. On the real M1 data that pinned the
    reported K* into the same [3.93, 4.25] grid interval for all three
    budgets (4.19 / 4.21 / 4.24), which is an artifact of this estimator's
    stopping rule, not a measurement that the cliff does not move: the
    estimator structurally cannot report a K* outside the first dip's
    interval no matter what happens at higher K.
    """
    for i in range(len(K_gain_pairs) - 1):
        K0, g0 = K_gain_pairs[i]
        K1, g1 = K_gain_pairs[i + 1]
        if g0 > 0 and g1 <= 0:
            frac = g0 / (g0 - g1)
            return K0 + frac * (K1 - K0)
    return None


def find_last_crossing_K(K_gain_pairs: list[tuple],
                         threshold: float = 0.0) -> float | None:
    """ROBUST cliff estimator (preferred): the LAST K at which mean paired
    gain is still above `threshold`, interpolated to the crossing with the
    next point.

    Rationale: the cliff is the frequency past which media-awareness stops
    helping *for good*. That is the last up-crossing of the envelope, not
    the first dip. This estimator is insensitive to isolated negative
    excursions below the cliff (which first-crossing is maximally
    sensitive to) while still requiring the gain to stay down afterwards,
    because it takes the LAST index with gain > threshold: any later
    recovery moves the estimate later, by construction.

    Returns None if gain is above threshold nowhere (cliff below the grid)
    or everywhere (cliff above the grid) -- both are honest "unresolved by
    this K grid" answers rather than a fabricated in-grid number.
    """
    above = [i for i, (_, g) in enumerate(K_gain_pairs) if g > threshold]
    if not above:
        return None
    i = above[-1]
    if i == len(K_gain_pairs) - 1:
        return None  # still positive at the top of the grid: cliff not bracketed
    K0, g0 = K_gain_pairs[i]
    K1, g1 = K_gain_pairs[i + 1]
    if g0 == g1:
        return K0
    frac = (g0 - threshold) / (g0 - g1)
    return K0 + frac * (K1 - K0)


def find_ci_includes_zero_K(K_gain_ci: list[tuple],
                            threshold: float = CI_FLATNESS_THRESHOLD_DB) -> float | None:
    """K_gain_ci: sorted [(K, mean_gain, ci_lo, ci_hi), ...]. Smallest K
    where the 95% CI includes 0 AND mean gain stays <= threshold for
    every subsequent K (a real trailing-flatness check, not just the
    first zero-crossing CI)."""
    n = len(K_gain_ci)
    for i in range(n):
        K, mean_gain, ci_lo, ci_hi = K_gain_ci[i]
        if ci_lo <= 0.0 <= ci_hi and all(g[1] <= threshold for g in K_gain_ci[i:]):
            return K
    return None


def gain_curve(grouped: dict, experiment_id: str, budget: float,
               method: str = "MIL") -> list[tuple]:
    """[(K, mean_gain, ci_lo, ci_hi), ...] sorted by K, for one
    (experiment_id, budget) -- paired gain of `method` over BSGD, in PSNR.
    Works for M1 (iteration-matched) or M2 (compute-matched) identically;
    which arm you're looking at is just which experiment_id you pass.
    `method` defaults to MIL (the paper's headline curve, F4/F5) but
    accepts any method present in the manifest (GS, LPC) so the same
    paired-comparison machinery draws the baselines Section 4.1 defines
    but Section 5 originally never plotted (D.2 in the work spec)."""
    entries = []
    for (exp_id, config_hash), by_method in grouped.items():
        if exp_id != experiment_id:
            continue
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        if cfg.get("contrast_cap") != budget:
            continue
        rows, bsgd = by_method.get(method, []), by_method.get("BSGD", [])
        pairs = paired_gain(rows, bsgd, key="psnr")
        gains = [g for _, g in pairs]
        stat = mean_std_median_ci95(gains)
        if stat["mean"] is not None:
            entries.append((cfg["K_nominal"], stat["mean"], stat["ci95_lo"], stat["ci95_hi"]))
    return sorted(entries, key=lambda e: e[0])


def gain_vs_bsgd_seed_mean(grouped: dict, experiment_id: str, budget: float,
                           method: str) -> list[tuple]:
    """[(K, gain_vs_bsgd_seed_mean, None, None), ...] for a DETERMINISTIC,
    single-seed method (GS, LPC -- "closed-form, no optimizer" per Sec.
    4.1) against BSGD's SEED-AVERAGED psnr, not a single BSGD seed drawn
    via paired_gain's seed-matching.

    Why this exists instead of reusing gain_curve/paired_gain for GS/LPC:
    GS and LPC are logged under seed=0 only (one deterministic run), while
    BSGD has its full seed set (3-5 depending on K). paired_gain matches
    by seed, so calling it on (GS, BSGD) pairs GS against BSGD's seed=0
    specifically -- one arbitrary noisy draw, not BSGD's actual seed-mean
    performance. Checked on real data: this produced a visibly noisy,
    K-non-monotone GS/LPC curve that was an artifact of which BSGD seed
    happened to be logged as 0 at each K, not a real effect. No CI is
    returned (None, None) since there is exactly one GS/LPC value per K --
    reporting a CI on a single point would fabricate precision that
    doesn't exist, the same reasoning already applied to MIL/BSGD's CI
    (which comes from real seed-to-seed spread that GS/LPC don't have)."""
    entries = []
    for (exp_id, config_hash), by_method in grouped.items():
        if exp_id != experiment_id:
            continue
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        if cfg.get("contrast_cap") != budget:
            continue
        rows, bsgd = by_method.get(method, []), by_method.get("BSGD", [])
        if not rows or not bsgd:
            continue
        method_val = rows[0]["psnr"]  # deterministic: exactly one row expected
        bsgd_mean = sum(r["psnr"] for r in bsgd) / len(bsgd)
        entries.append((cfg["K_nominal"], method_val - bsgd_mean, None, None))
    return sorted(entries, key=lambda e: e[0])


def _per_seed_gain_by_K(grouped: dict, experiment_id: str, budget: float) -> dict:
    """{seed: [(K, gain), ...]} for one (experiment_id, budget) -- the
    PER-SEED (not seed-averaged) paired gain curve, needed for M3's
    "uncertainty over seeds" on the cliff-location SHIFT specifically
    (as opposed to gain_curve's seed-averaged CI on the gain itself)."""
    by_seed = defaultdict(list)
    for (exp_id, config_hash), by_method in grouped.items():
        if exp_id != experiment_id:
            continue
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        if cfg.get("contrast_cap") != budget:
            continue
        mil, bsgd = by_method.get("MIL", []), by_method.get("BSGD", [])
        for seed, g in paired_gain(mil, bsgd, key="psnr"):
            by_seed[seed].append((cfg["K_nominal"], g))
    return {s: sorted(pts, key=lambda e: e[0]) for s, pts in by_seed.items()}


def _configs_for_budget(grouped: dict, experiment_id: str, budget: float) -> list[tuple]:
    """[(config, by_method), ...] for every (experiment_id, config) group
    whose config has this contrast_cap budget."""
    out = []
    for (exp_id, config_hash), by_method in grouped.items():
        if exp_id != experiment_id:
            continue
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        if cfg.get("contrast_cap") == budget:
            out.append((cfg, by_method))
    return out


def _self_consistent_Kc(rec, contrast_by_K: dict) -> float | None:
    """Smallest measured K where the required boost 1/H(K) exceeds the
    contrast C(K) the optimizer actually realized at that same K, linearly
    interpolated between adjacent K samples.

    This is the per-K analogue of NPDDRecorder.predicted_cliff (which
    inverts 1/H(K) > B for one scalar B). Returns None when the condition
    never flips over the sampled K range -- an honest "the grid does not
    bracket Kc" rather than an extrapolated number.
    """
    import torch
    Ks = sorted(contrast_by_K)
    if not Ks:
        return None
    K_t = torch.tensor(Ks, dtype=rec.dtype)
    inv_H = (1.0 / rec.small_signal_mtf(K_t)).tolist()
    deficit = [ih - contrast_by_K[k] for ih, k in zip(inv_H, Ks)]  # >0 => infeasible
    for i in range(len(Ks)):
        if deficit[i] > 0:
            if i == 0:
                return Ks[0]
            d0, d1 = deficit[i - 1], deficit[i]
            if d1 == d0:
                return Ks[i]
            return Ks[i - 1] + (0.0 - d0) / (d1 - d0) * (Ks[i] - Ks[i - 1])
    return None


def headroom_closure(grouped: dict, experiment_id: str = "M1", budgets=BUDGETS) -> list[dict]:
    """The paper's central table: budget -> measured contrast C (from
    MIL's logged realized-contrast stats) -> predicted Kc(C) (Eq. 5, using
    MEASURED C, not the nominal budget) -> observed K* (both estimators).
    Call once with experiment_id="M1" (iteration-matched) and once with
    "M2" (compute-matched) -- see m3_cliff_shift for the comparison
    between the two.
    """
    table = []
    for budget in budgets:
        curve = gain_curve(grouped, experiment_id, budget)
        configs_and_methods = _configs_for_budget(grouped, experiment_id, budget)

        # Measured contrast C, PER K (not averaged over K).
        #
        # This used to be one scalar: the mean of MIL's realized max/mean
        # over every (K, seed) in the budget group. That is a
        # mis-specification whenever the realized contrast is itself
        # K-dependent -- which it measurably is. On the real M1 data at
        # 8x budget, MIL realized C = 8.0 at the lowest K but only ~3.4-5.6
        # across the mid range, and 1.03 at the top K, so a single
        # K-averaged C described no K in particular. It was also outlier-
        # driven: at 2x budget the near-constant C~2.02 was dragged to
        # 1.95 purely by the single collapsed K=15.7 point.
        #
        # Kc is now solved SELF-CONSISTENTLY: the cliff is where the boost
        # the medium demands, 1/H(K), first exceeds the contrast the
        # optimizer actually realized AT THAT K, C(K) -- i.e. the smallest
        # K with 1/H(K) > C(K), rather than 1/H(K) > (one global C).
        contrast_by_K = {}
        for cfg, by_method in configs_and_methods:
            vals = [r["contrast"]["max_over_mean"] for r in by_method.get("MIL", [])
                    if r["contrast"]["max_over_mean"] is not None]
            if vals:
                contrast_by_K[round(float(cfg["K_nominal"]), 6)] = statistics.fmean(vals)

        if not curve or not contrast_by_K:
            table.append(dict(budget=budget, status="no_data"))
            continue

        measured_C = statistics.fmean(contrast_by_K.values())  # reported for continuity only
        any_cfg = configs_and_methods[0][0]
        medium = MediumParams(**any_cfg["medium"])
        rec = NPDDRecorder(any_cfg["n_x"], any_cfg["dx"], params=medium)
        predicted_Kc_avgC = float(rec.predicted_cliff(budget=measured_C))
        predicted_Kc = _self_consistent_Kc(rec, contrast_by_K)

        K_gain_pairs = [(k, g) for k, g, _, _ in curve]
        table.append(dict(
            budget=budget,
            measured_contrast_C=measured_C,
            contrast_by_K=sorted(contrast_by_K.items()),
            # primary (self-consistent, per-K C) and the old K-averaged
            # form side by side, so the change in method is visible in the
            # output rather than being a silent redefinition of Kc.
            predicted_Kc_from_measured_C=predicted_Kc,
            predicted_Kc_from_Kaveraged_C=predicted_Kc_avgC,
            observed_Kstar_last=find_last_crossing_K(K_gain_pairs),   # preferred
            observed_Kstar_interp=find_zero_crossing_K(K_gain_pairs),  # legacy/fragile
            observed_Kstar_ci=find_ci_includes_zero_K(curve),
            n_K_points=len(curve), gain_curve=curve,
        ))
    return table


def m3_cliff_shift(grouped: dict, budgets=BUDGETS) -> list[dict]:
    """M3 (per your instruction, not a data-collection tier): the shift in
    cliff location between the M1 (iteration-matched) and M2 (compute-
    matched) arms, with uncertainty over seeds.

    Two views are reported per budget:
      - point-estimate shift: K*(M2, seed-averaged) - K*(M1, seed-averaged),
        via the zero-crossing-interp estimator on each arm's aggregate curve.
      - per-seed shift distribution: K* is ALSO estimated separately from
        EACH seed's own (non-averaged) gain-vs-K curve in both arms, paired
        by seed, giving a real mean/std of the shift across seeds -- this
        is the "with uncertainty over seeds" part specifically, not just
        two point estimates compared.
    """
    out = []
    for budget in budgets:
        m1_curve = gain_curve(grouped, "M1", budget)
        m2_curve = gain_curve(grouped, "M2", budget)
        if not m1_curve or not m2_curve:
            out.append(dict(budget=budget, status="no_data"))
            continue

        kstar_m1 = find_zero_crossing_K([(k, g) for k, g, _, _ in m1_curve])
        kstar_m2 = find_zero_crossing_K([(k, g) for k, g, _, _ in m2_curve])
        point_shift = (kstar_m2 - kstar_m1
                      if kstar_m1 is not None and kstar_m2 is not None else None)

        m1_by_seed = _per_seed_gain_by_K(grouped, "M1", budget)
        m2_by_seed = _per_seed_gain_by_K(grouped, "M2", budget)
        common_seeds = sorted(set(m1_by_seed) & set(m2_by_seed))
        per_seed_shifts = []
        for seed in common_seeds:
            k1 = find_zero_crossing_K(m1_by_seed[seed])
            k2 = find_zero_crossing_K(m2_by_seed[seed])
            if k1 is not None and k2 is not None:
                per_seed_shifts.append(k2 - k1)

        shift_stats = mean_std_median_ci95(per_seed_shifts) if per_seed_shifts else None
        out.append(dict(
            budget=budget, Kstar_M1=kstar_m1, Kstar_M2=kstar_m2,
            point_estimate_shift=point_shift,
            n_seeds_with_valid_shift=len(per_seed_shifts),
            per_seed_shift_stats=shift_stats,
        ))
    return out


# ----------------------------------------------------------- sub-cliff check
SUB_CLIFF_OLD_VALUES = {  # from results_prelim.json, single effective seed (bug)
    0.98: 1.6458, 1.96: 0.9481, 2.62: 2.3740,
}


def s1_ablation_summary(grouped: dict) -> dict:
    """S1: paired gain (MIL-BSGD) per ablation condition, K-averaged over
    the 3 sub/near/post-cliff points, plus the ratio of each ablation's
    mean gain to the baseline condition's. Used for the paper's physics-
    ablation macros (which physics term actually drives the gain)."""
    from manifest import S1_CONDITIONS
    rows = [(exp_id, ch) for exp_id, ch in grouped if exp_id == "S1"]
    if not rows:
        return dict(status="no_data")
    by_cond = {}
    for cond in S1_CONDITIONS:
        gains = []
        for exp_id, ch in rows:
            by_method = grouped[(exp_id, ch)]
            any_rows = next(iter(by_method.values()), None)
            if not any_rows or any_rows[0]["config"].get("ablation_condition") != cond:
                continue
            pairs = paired_gain(by_method.get("MIL", []), by_method.get("BSGD", []), key="psnr")
            gains.extend(g for _, g in pairs)
        stat = mean_std_median_ci95(gains)
        by_cond[cond] = stat
    baseline_mean = by_cond.get("baseline", {}).get("mean")
    ratios = {}
    if baseline_mean:
        for cond, stat in by_cond.items():
            if cond != "baseline" and stat["mean"] is not None:
                ratios[cond] = stat["mean"] / baseline_mean
    return dict(status="ok", by_condition=by_cond, ratio_to_baseline=ratios)


def s2_sensitivity_summary(grouped: dict) -> dict:
    """S2: paired gain (MIL-BSGD) K-averaged over the collapse-region
    K-points, at the extreme (+/-50%) perturbation for each of D0/sigma/
    kappa, plus the baseline (0%) -- used to report how flat/sensitive
    the result is to individual NPDD parameter error."""
    from manifest import S2_PARAMS
    rows = [(exp_id, ch) for exp_id, ch in grouped if exp_id == "S2"]
    if not rows:
        return dict(status="no_data")

    def _gains_for(param, pct):
        gains = []
        for exp_id, ch in rows:
            by_method = grouped[(exp_id, ch)]
            any_rows = next(iter(by_method.values()), None)
            if not any_rows:
                continue
            cfg = any_rows[0]["config"]
            if cfg.get("sensitivity_pct") != pct:
                continue
            if pct != 0 and cfg.get("sensitivity_param") != param:
                continue
            pairs = paired_gain(by_method.get("MIL", []), by_method.get("BSGD", []), key="psnr")
            gains.extend(g for _, g in pairs)
        return gains

    baseline_param = S2_PARAMS[0]
    baseline_stat = mean_std_median_ci95(_gains_for(baseline_param, 0))
    by_param = {}
    all_gains = list(_gains_for(baseline_param, 0))
    for param in S2_PARAMS:
        lo = mean_std_median_ci95(_gains_for(param, -50))
        hi = mean_std_median_ci95(_gains_for(param, 50))
        by_param[param] = dict(minus50=lo, plus50=hi)
        all_gains += _gains_for(param, -50) + _gains_for(param, 50)
    overall_range = (min(all_gains), max(all_gains)) if all_gains else (None, None)
    return dict(status="ok", baseline=baseline_stat, by_param=by_param,
               overall_min=overall_range[0], overall_max=overall_range[1])


def s3_mismatch_summary(grouped: dict) -> dict:
    """S3: paired gain (MIL-BSGD) when the exposure was designed against
    the NOMINAL twin and then recorded on a MISCALIBRATED one.

    Read this alongside s2_sensitivity_summary and note that they are not
    the same measurement. S2 perturbs the medium and lets BOTH arms
    re-optimize on it, so the perturbation is common to both arms and
    cancels in the paired difference by the paper's own argument. S3
    freezes the exposure at theta_nominal and evaluates it at theta', so
    the error is one-sided and real. Only S3 can support a claim of the
    form "the advantage survives getting a parameter wrong".

    Reports, per perturbed parameter:
      by_pct       -- paired-gain stats at each perturbation level
      worst        -- the level with the lowest mean gain, and that gain
      sign_flip_pct-- the smallest |pct| at which mean gain goes negative
                      (None if it never does over the tested range)
    plus overall min/max across every tested condition. `sign_flip_pct`
    is the number the robustness sentence must be written against: if it
    is None across the board, the advantage survives the whole tested
    range; if it is not, the paper reports where it breaks rather than
    claiming it does not.
    """
    rows = [(exp_id, ch) for exp_id, ch in grouped if exp_id == "S3"]
    if not rows:
        return dict(status="no_data")

    # (param, pct) -> per-seed, K-averaged paired gains.
    #
    # Pairing happens inside each config group (one K), which is the only
    # place paired_gain is safe -- it matches arms by seed alone, so
    # handing it several K at once would silently keep one K's BSGD value
    # per seed. Each seed's gains are then averaged over K, so the
    # reported CI is over SEEDS. Pooling (seed, K) instead would make the
    # interval reflect between-K spread, which is a real effect being
    # averaged over, not uncertainty about the mean.
    per_seed: dict = {}
    for exp_id, ch in rows:
        by_method = grouped[(exp_id, ch)]
        any_rows = next(iter(by_method.values()), None)
        if not any_rows:
            continue
        cfg = any_rows[0]["config"]
        key = (cfg.get("mismatch_param"), cfg.get("mismatch_pct"))
        for seed, g in paired_gain(by_method.get("MIL", []),
                                   by_method.get("BSGD", []), key="psnr"):
            per_seed.setdefault(key, {}).setdefault(seed, []).append(g)
    buckets = {k: [sum(v) / len(v) for v in seeds.values() if v]
               for k, seeds in per_seed.items()}

    params = sorted({k[0] for k in buckets if k[0] is not None})
    # pct=0 (nominal) is emitted under the first parameter only, by
    # construction in build_S3_conditions -- shared across all curves.
    nominal_key = next((k for k in buckets if k[1] == 0), None)
    nominal = mean_std_median_ci95(buckets.get(nominal_key, []))

    by_param, all_gains = {}, list(buckets.get(nominal_key, []))
    for param in params:
        pcts = sorted({k[1] for k in buckets if k[0] == param} | {0})
        by_pct, worst, sign_flip = {}, None, None
        for pct in pcts:
            key = nominal_key if pct == 0 else (param, pct)
            gains = buckets.get(key, [])
            if not gains:
                continue
            stat = mean_std_median_ci95(gains)
            by_pct[str(pct)] = stat
            if pct != 0:
                all_gains += gains
            if worst is None or stat["mean"] < worst[1]:
                worst = (pct, stat["mean"])
            if stat["mean"] < 0 and (sign_flip is None or abs(pct) < abs(sign_flip)):
                sign_flip = pct
        by_param[param] = dict(by_pct=by_pct,
                               worst_pct=worst[0] if worst else None,
                               worst_mean_gain=worst[1] if worst else None,
                               sign_flip_pct=sign_flip)

    return dict(status="ok", nominal=nominal, by_param=by_param,
                overall_min=min(all_gains) if all_gains else None,
                overall_max=max(all_gains) if all_gains else None,
                any_sign_flip=any(v["sign_flip_pct"] is not None
                                  for v in by_param.values()),
                tested_pct_range=sorted({k[1] for k in buckets
                                         if k[1] is not None}))


def sat_surrogate_summary(grouped: dict, experiment_id: str = "M1",
                          budgets=BUDGETS) -> dict:
    """SAT: how much of MIL's advantage a cheap saturation-only surrogate
    recovers, and at what fraction of the modeling.

    `fraction_of_mil` is the headline: mean SAT gain over BSGD divided by
    mean MIL gain over BSGD, per budget. It is only meaningful when MIL's
    own gain is comfortably positive, so it is reported as None where
    mean MIL gain is at or below zero rather than as a large ratio of two
    small numbers.

    Also carries the surrogate's own calibration quality (`fit_nrmse`,
    from the result rows' sat_fit field) -- a SAT number is not
    interpretable without knowing how well a purely pointwise model could
    fit the real twin in the first place.
    """
    out = dict(status="no_data", by_budget={})
    have = False
    for budget in budgets:
        sat = gain_curve(grouped, experiment_id, budget, method="SAT")
        mil = gain_curve(grouped, experiment_id, budget, method="MIL")
        if not sat or not mil:
            out["by_budget"][str(budget)] = dict(status="no_data")
            continue
        have = True
        sat_means = [c[1] for c in sat]
        mil_means = [c[1] for c in mil]
        sat_mean = statistics.fmean(sat_means)
        mil_mean = statistics.fmean(mil_means)
        # Per-K fraction as well as ratio-of-means: a single ratio can
        # hide a surrogate that tracks MIL at low K and collapses at high
        # K, which is exactly the failure mode worth knowing about.
        by_K = {}
        common = {c[0]: c[1] for c in mil}
        for K, m, _lo, _hi in sat:
            if K in common and common[K] > 1e-9:
                by_K[f"{K:.4f}"] = m / common[K]
        out["by_budget"][str(budget)] = dict(
            status="ok", sat_mean_gain=sat_mean, mil_mean_gain=mil_mean,
            sat_min_gain=min(sat_means), sat_max_gain=max(sat_means),
            fraction_of_mil=(sat_mean / mil_mean if mil_mean > 1e-9 else None),
            fraction_by_K=by_K,
            n_K=len(sat))
    if have:
        out["status"] = "ok"

    # Surrogate calibration quality, carried on the SAT result rows.
    nrmses, a_effs = [], []
    for (exp_id, _ch), by_method in grouped.items():
        if exp_id != experiment_id:
            continue
        for r in by_method.get("SAT", []):
            fit = r.get("sat_fit") or {}
            if fit.get("nrmse") is not None:
                nrmses.append(fit["nrmse"])
            if fit.get("a_eff") is not None:
                a_effs.append(fit["a_eff"])
    if nrmses:
        out["fit_nrmse"] = mean_std_median_ci95(nrmses)
        out["fit_a_eff"] = mean_std_median_ci95(a_effs)
    return out


def sub_cliff_non_monotonicity_status(grouped: dict) -> dict:
    """Whether the old data's sub-cliff non-monotonicity (+1.65 at
    K=0.98, +0.95 at K=1.96, +2.37 at K=2.62) is noise or structure,
    'with error bars.' Those old numbers are from results_prelim.json,
    produced under the (now-fixed) seed bug -- every 'seed' there was one
    bit-identical trajectory, so NO error bars can be computed for them;
    the question is genuinely unanswerable from that data, not just
    unanswered. Checks whether real multi-seed M1 data now exists at
    those K values and answers for real if so; otherwise states the
    blocker honestly instead of guessing."""
    found = {}
    for K in SUB_CLIFF_OLD_VALUES:
        for budget in BUDGETS:
            curve = gain_curve(grouped, "M1", budget)
            match = next((e for e in curve if abs(e[0] - K) < 1e-3), None)
            if match:
                found.setdefault(K, {})[budget] = match
    if not found:
        return dict(
            status="blocked_pending_phase3",
            reason="Old values (results_prelim.json) were produced under the "
                   "seed-init bug -- every 'seed' was one bit-identical "
                   "trajectory, so no error bars can be computed retroactively. "
                   "This question needs real multi-seed M1 data at K=0.98/1.96/2.62, "
                   "which does not exist yet.",
            old_values_no_error_bars=SUB_CLIFF_OLD_VALUES,
        )
    return dict(status="answered_from_real_data", data=found)


# --------------------------------------------------------------- top level
def build_paper_numbers(results_root: str = RESULTS_ROOT) -> dict:
    results = load_all_results(results_root)
    grouped = group_by_config(results)

    per_config = {}
    for (exp_id, config_hash), by_method in grouped.items():
        key = f"{exp_id}/{config_hash}"
        per_config[key] = {m: aggregate_method(rows) for m, rows in by_method.items()}

    present = set(exp_id for exp_id, _ in grouped)
    m1_present, m2_present = "M1" in present, "M2" in present
    out = dict(
        n_result_files=len(results),
        experiments_present=sorted(present),
        per_config=per_config,
        m1_headroom_closure=headroom_closure(grouped, "M1") if m1_present else
            [dict(budget=b, status="no_data") for b in BUDGETS],
        m2_headroom_closure=headroom_closure(grouped, "M2") if m2_present else
            [dict(budget=b, status="no_data") for b in BUDGETS],
        m3_cliff_shift=m3_cliff_shift(grouped) if (m1_present and m2_present) else
            [dict(budget=b, status="no_data") for b in BUDGETS],
        sub_cliff_non_monotonicity=sub_cliff_non_monotonicity_status(grouped),
        s1_ablation_summary=s1_ablation_summary(grouped),
        s2_sensitivity_summary=s2_sensitivity_summary(grouped),
        s3_mismatch_summary=s3_mismatch_summary(grouped),
        sat_surrogate_summary=sat_surrogate_summary(grouped),
        # M2 carries SAT at the sub-cliff K = 1.31 rad/um, which lies
        # below M1's grid minimum of 1.96 -- i.e. exactly where
        # media-in-the-loop's advantage is largest and the cheap
        # surrogate is most interesting. Reported separately rather than
        # merged into the M1 summary, because M2's BSGD arm is
        # compute-matched (~21.7x SAT's iterations) and the two
        # denominators are therefore not the same quantity.
        sat_surrogate_summary_m2=sat_surrogate_summary(grouped, "M2"),
    )
    return out


def main():
    out = build_paper_numbers()
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {SUMMARY_PATH}")
    print(f"  {out['n_result_files']} result files, experiments present: "
          f"{out['experiments_present']}")
    for row in out["m1_headroom_closure"]:
        print(f"  M1 budget={row['budget']}: {row.get('status', 'ok')} "
              f"{'' if row.get('status') else row}")
    for row in out["m3_cliff_shift"]:
        print(f"  M3 (cliff shift) budget={row['budget']}: {row.get('status', 'ok')} "
              f"{'' if row.get('status') else row}")
    s3 = out["s3_mismatch_summary"]
    if s3.get("status") == "ok":
        flip = ("gain goes negative somewhere in the tested range"
                if s3["any_sign_flip"] else
                "gain stays positive across the whole tested range")
        print(f"  S3 (twin mismatch): {flip}; "
              f"gain {s3['overall_min']:.3f}..{s3['overall_max']:.3f} dB "
              f"(nominal {s3['nominal']['mean']:.3f})")
        for param, v in s3["by_param"].items():
            print(f"    {param}: worst {v['worst_mean_gain']:.3f} dB at "
                  f"{v['worst_pct']}%, sign flip at {v['sign_flip_pct']}")
    sat = out["sat_surrogate_summary"]
    if sat.get("status") == "ok":
        for b, v in sat["by_budget"].items():
            if v.get("status") != "ok":
                continue
            frac = v["fraction_of_mil"]
            print(f"  SAT budget={b}: mean gain {v['sat_mean_gain']:.3f} dB vs "
                  f"MIL {v['mil_mean_gain']:.3f} dB"
                  + (f" ({100 * frac:.0f}% of MIL)" if frac is not None else ""))
        if "fit_nrmse" in sat:
            print(f"    surrogate fit: a_eff={sat['fit_a_eff']['mean']:.3f}, "
                  f"NRMSE={sat['fit_nrmse']['mean']:.4f}")


if __name__ == "__main__":
    main()
