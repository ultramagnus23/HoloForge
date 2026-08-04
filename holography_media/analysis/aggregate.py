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
import json
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


def group_by_config(results: list[dict]) -> dict:
    """{(experiment_id, config_hash): {method_id: [result, ...]}}"""
    g = defaultdict(lambda: defaultdict(list))
    for r in results:
        g[(r["experiment_id"], r["config_hash"])][r["method_id"]].append(r)
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


# --------------------------------------------------------------- E1 cliff estimators
def find_zero_crossing_K(K_gain_pairs: list[tuple]) -> float | None:
    """Linear-interpolation zero-crossing where mean paired gain goes
    positive -> non-positive as K increases. K_gain_pairs sorted by K."""
    for i in range(len(K_gain_pairs) - 1):
        K0, g0 = K_gain_pairs[i]
        K1, g1 = K_gain_pairs[i + 1]
        if g0 > 0 and g1 <= 0:
            frac = g0 / (g0 - g1)
            return K0 + frac * (K1 - K0)
    return None


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


def gain_curve(grouped: dict, experiment_id: str, budget: float) -> list[tuple]:
    """[(K, mean_gain, ci_lo, ci_hi), ...] sorted by K, for one
    (experiment_id, budget) -- MIL-BSGD paired gain in PSNR. Works for M1
    (iteration-matched) or M2 (compute-matched) identically; which arm
    you're looking at is just which experiment_id you pass."""
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
        mil, bsgd = by_method.get("MIL", []), by_method.get("BSGD", [])
        pairs = paired_gain(mil, bsgd, key="psnr")
        gains = [g for _, g in pairs]
        stat = mean_std_median_ci95(gains)
        if stat["mean"] is not None:
            entries.append((cfg["K_nominal"], stat["mean"], stat["ci95_lo"], stat["ci95_hi"]))
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

        # measured contrast C: mean of MIL's realized max/mean across all
        # (K, seed) for this budget
        contrasts = [r["contrast"]["max_over_mean"]
                    for _, by_method in configs_and_methods
                    for r in by_method.get("MIL", [])
                    if r["contrast"]["max_over_mean"] is not None]

        if not curve or not contrasts:
            table.append(dict(budget=budget, status="no_data"))
            continue

        measured_C = statistics.fmean(contrasts)
        # predicted Kc(measured_C): rebuild a recorder from ANY matching
        # config's medium/grid to evaluate the analytic Eq. 5 inversion
        # (grid/medium are the same across every job in a budget group by
        # construction of build_M1_jobs/build_M2_jobs, so any one config
        # is representative)
        any_cfg = configs_and_methods[0][0]
        medium = MediumParams(**any_cfg["medium"])
        rec = NPDDRecorder(any_cfg["n_x"], any_cfg["dx"], params=medium)
        predicted_Kc = float(rec.predicted_cliff(budget=measured_C))

        K_gain_pairs = [(k, g) for k, g, _, _ in curve]
        table.append(dict(
            budget=budget, measured_contrast_C=measured_C,
            predicted_Kc_from_measured_C=predicted_Kc,
            observed_Kstar_interp=find_zero_crossing_K(K_gain_pairs),
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


if __name__ == "__main__":
    main()
