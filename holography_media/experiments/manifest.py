"""
Job manifest builders: V1-V3 (validation, blocking), M1-M2 (main cliff/
budget comparison, two arms), S1-S2 (supporting: physics ablation,
parameter sensitivity). Renamed from the earlier E1-E7 scheme per the
execution spec's explicit instruction to avoid collision and restructure
into three tiers -- see docs/legacy_results_audit.md for what the old E1-E7
data maps to under this structure.

Execution-readiness by tier (see each builder's docstring for detail):
  M1, M2, S1, S2 -- fully execution-ready NOW. They reuse the SAME method
    registry (experiments/methods.py: GS/BSGD/LPC/MIL/ORC/ORU) and the
    same run_job() path as the old E1-E4 did; only the job CONFIGS differ
    (different medium-param sweeps / iteration-matching rules). No new
    runner code needed.
  V3 -- already execution-ready, but via a DIFFERENT script
    (experiments/rcwa_crosscheck.py), not this manifest/run_job path at
    all (RCWA cross-checks have no seed/method-registry structure -- see
    that script's own docstring). Real data already exists:
    results_rcwa.json (3-case) + results_rcwa_e7.json (90-case grid).
  V1, V2 -- job CONFIGS are defined below (real, not placeholders), but
    NEITHER has an execution path through run_job()/methods.run_method()
    yet -- both are solver/regime characterizations, not optimizer-method
    comparisons, and need new runner code. V1 overlaps substantially with
    the existing (already-written) experiments/f1_validate_twin.py; V2
    needs a genuine 3-way Kogelnik/BPM/RCWA comparison that does not exist
    yet (rcwa_crosscheck.py's E7 grid only compares Kogelnik vs RCWA, not
    BPM). Flagged, not silently faked.

A manifest is a flat list of job dicts:
    {experiment_id, method_id, seed, config, config_hash}
`config` fully resolves everything needed to reconstruct the target,
MediumParams, NPDDRecorder, and SlabBPM for that job -- nothing implicit.
`config_hash` is a short sha256 of the canonicalized (sorted-key) JSON of
`config`, used as the results-directory key so two configs that differ in
any field never collide and identical configs always resolve to the same
path (resume = same command -> same hashes -> same skip set).
"""
from __future__ import annotations
import hashlib
import json
import math

# PVA/AA-like defaults, matching holomedia.npdd.MediumParams() exactly --
# duplicated here (not imported) so a manifest's config is self-contained
# JSON with no import-time dependency on holomedia's current field
# defaults silently drifting the meaning of an already-generated manifest.
DEFAULT_MEDIUM = dict(D0=0.1, sigma=0.08, kappa=2.0, gamma=1.0, dn_max=3.5e-3,
                      k_bleach=0.2, alpha_D=1.0, shrinkage=0.005,
                      thickness=30.0, n0=1.5)

# GS/LPC (media_blind_gs, linear_precomp) are closed-form, no optimization
# loop -- see experiments/methods.py for the full registry and why these
# short codes replaced the earlier M1-M5b naming (collision with these
# new experiment-tier names).
ALL_METHODS = ["GS", "BSGD", "LPC", "MIL", "ORC", "ORU"]

# 5 -> 3 seeds (compute-budget reduction, see the run-cost audit that
# prompted this change): analysis/aggregate.py's CI already uses a
# t-distribution, which handles n=3 correctly -- this just means M1/M2
# report n=3 with correspondingly wider intervals, honestly, not n=5.
PAPER_SEEDS = [0, 1, 2]

# sub/near/post-cliff K's (moved up from the S1 section below so
# build_M2_jobs can use them as its default K subset -- Python evaluates
# default-argument expressions at function-definition time, so this has
# to be defined before build_M2_jobs, not just before S1's build_S1_jobs).
S1_K_POINTS = [1.308996938995747, 3.9269908169872414, 5.235987755982988]

# Measured (not assumed) on CPU at n_x=1024: MIL (media_in_the_loop, full
# NPDD forward per iteration) costs ~21.7x more wall-clock per iteration
# than BSGD (media_blind_sgd, a single linear multiply + BPM readout) --
# see the commit introducing M2 for the measurement script. PROVISIONAL:
# re-measure on the actual GPU before relying on this for real M2 runs:
# relative FFT-vs-elementwise cost can differ meaningfully between CPU and
# GPU (and between GPU models), so this default should be treated as a
# starting point, not a settled constant.
COMPUTE_MATCH_RATIO = 21.7


def config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _job(experiment_id, method_id, seed, config):
    return dict(experiment_id=experiment_id, method_id=method_id, seed=seed,
               config=dict(config), config_hash=config_hash(config))


def _bars_target_spec(period_px: int) -> dict:
    return dict(kind="bars", period_px=period_px)


def K_from_period(period_px: int, dx: float) -> float:
    return 2.0 * math.pi / (period_px * dx)


def period_from_K(K: float, dx: float) -> int:
    return max(2, round(2.0 * math.pi / (K * dx)))


# The 14-point K grid from the original cliff/budget design (7 historical
# period-px values union a dense insert across the collapse region) --
# unchanged science, just the experiment_id it feeds is renamed below.
def _cliff_K_grid(dx: float) -> list[float]:
    existing_periods = [8, 12, 16, 24, 32, 48, 64]
    existing_K = sorted(K_from_period(p, dx) for p in existing_periods)
    dense_insert_K = [3.5, 4.25, 4.6, 5.0, 5.6, 6.0, 6.5]
    # Second dense insert, 7.0-7.6. The analytic cliff prediction at the
    # 4x and 8x contrast budgets lands near Kc ~ 6.6 and ~ 7.3, but the
    # grid above jumped 6.5 -> 7.854 -> 10.472, so the two highest-budget
    # predictions fell in an unsampled gap: there was no measurement
    # anywhere near where the theory said the cliff should be, which alone
    # made the budget-dependence of K* unresolvable at those budgets.
    high_budget_bracket_K = [7.0, 7.3, 7.6]
    return sorted(set(round(k, 6) for k in
                      existing_K + dense_insert_K + high_budget_bracket_K))


# =====================================================================
# M1/M2: main cliff x budget comparison, two arms (per your instruction:
# "matched on scenes, seeds, optimizer, iteration budget, and separately
# on compute"). M3 is NOT a third manifest -- it's the derived statistic
# (shift in cliff location between the M1/M2 arms, with seed uncertainty),
# computed by analysis/aggregate.py from M1+M2's data. See the PR/commit
# message for the explicit interpretation call this rests on.
# =====================================================================
def build_M1_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, methods=None) -> list[dict]:
    """Cliff x budget grid, ITERATION-matched arms: BSGD (media-unaware)
    and MIL (media-aware) both get the same n_iters budget. This is the
    original cliff/budget design (formerly build_E1_jobs), unchanged
    science -- only the experiment_id changed (E1 -> M1) and the method
    registry codes changed (M2/M4 -> BSGD/MIL) to avoid the tier-name
    collision.
    """
    seeds = seeds if seeds is not None else PAPER_SEEDS
    methods = methods if methods is not None else ALL_METHODS
    dx = 51.2 / n_x  # fixed physical window, matches gpu_npdd_mesh_convergence_sweep.py convention
    all_K = _cliff_K_grid(dx)
    budgets = [2.0, 4.0, 8.0]

    jobs = []
    for K in all_K:
        period_px = period_from_K(K, dx)
        for budget in budgets:
            for method_id in methods:
                base_config = dict(
                    n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                    converge_tol=converge_tol, contrast_cap=budget,
                    dose_budget=1.0, medium=DEFAULT_MEDIUM,
                    target=_bars_target_spec(period_px), K_nominal=K,
                    arm="iteration_matched",
                )
                if method_id in ("GS", "LPC"):
                    for_seeds = [0]  # closed-form, no seed dependence worth repeating
                else:
                    for_seeds = seeds
                for seed in for_seeds:
                    jobs.append(_job("M1", method_id, seed, base_config))
    return jobs


def build_M2_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, compute_match_ratio: float = COMPUTE_MATCH_RATIO,
                  K_points=None) -> list[dict]:
    """Same cliff x budget grid, COMPUTE-matched arms: BSGD gets
    n_iters * compute_match_ratio iterations so its total wall-clock/FLOP
    cost approximately matches MIL's (which does a full NPDD forward pass
    per iteration; BSGD does one linear multiply). Only BSGD and MIL are
    compared here (GS/LPC/ORC/ORU aren't part of the arm-matching question
    -- they don't have a "budget" to match in the same sense).

    compute_match_ratio defaults to a CPU-measured value (see
    COMPUTE_MATCH_RATIO's docstring) -- pass a GPU-measured ratio once
    available; this parameter exists specifically so that re-measurement
    doesn't require editing this function.

    K_points defaults to S1_K_POINTS (sub/near/post-cliff, 3 points) rather
    than M1's full 14-point grid: M2 exists to answer one question -- is
    MIL's gain just more compute, at fixed budget? -- not to re-map the
    cliff (M1 already does that). 3 K's x 3 budgets = 9 cells is a
    complete answer to that confound at ~1/5th the job count. Pass
    K_points=_cliff_K_grid(dx) explicitly for the full grid if you want it.
    """
    seeds = seeds if seeds is not None else PAPER_SEEDS
    dx = 51.2 / n_x
    all_K = K_points if K_points is not None else S1_K_POINTS
    budgets = [2.0, 4.0, 8.0]
    bsgd_n_iters = round(n_iters * compute_match_ratio)

    jobs = []
    for K in all_K:
        period_px = period_from_K(K, dx)
        for budget in budgets:
            target = _bars_target_spec(period_px)
            mil_config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                             converge_tol=converge_tol, contrast_cap=budget,
                             dose_budget=1.0, medium=DEFAULT_MEDIUM,
                             target=target, K_nominal=K, arm="compute_matched")
            bsgd_config = dict(mil_config, n_iters=bsgd_n_iters)
            for seed in seeds:
                jobs.append(_job("M2", "MIL", seed, mil_config))
                jobs.append(_job("M2", "BSGD", seed, bsgd_config))
    return jobs


# =====================================================================
# S1: physics-component ablation (NOT the same as the pre-existing
# experiments/ablation_gradients.py, which ablates GRADIENT COMPUTATION
# PATHWAYS -- an engineering question. This ablates NPDD MODEL TERMS -- a
# physics question. See docs/legacy_results_audit.md's flagged naming
# collision.) Each condition is implemented via existing MediumParams
# fields alone (no new holomedia code): setting a parameter to its
# limiting value approximates removing that physical mechanism.
# =====================================================================
# S1_K_POINTS moved up to the top-of-file constants section (needed there
# as build_M2_jobs's default K subset).
S1_BUDGET = 2.0

S1_CONDITIONS = {
    "baseline": {},
    "no_nonlocality": dict(sigma=0.0),  # Ghat(K) = exp(-0.5 K^2 sigma^2) -> 1: no blur
    "no_diffusion": dict(D0=0.0),  # D_eff = D0 exp(-alpha_D N) -> 0: no transport
    "no_dye_depletion": dict(k_bleach=0.0),  # d(t) stays 1: no sensitivity falloff
    # dn_max -> 100x default: tanh(1.5 N) stays in its linear regime for
    # realistic N, approximating an unsaturating (linear) index response.
    # This is an APPROXIMATION of removing saturation (tanh is still
    # technically present), not an exact ablation -- documented as such,
    # not silently treated as exact.
    "no_saturation_approx": dict(dn_max=DEFAULT_MEDIUM["dn_max"] * 100),
}


def build_S1_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None) -> list[dict]:
    seeds = seeds if seeds is not None else [0, 1, 2]
    dx = 51.2 / n_x
    jobs = []
    for K in S1_K_POINTS:
        period_px = period_from_K(K, dx)
        for cond_name, overrides in S1_CONDITIONS.items():
            medium = dict(DEFAULT_MEDIUM, **overrides)
            config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                         converge_tol=converge_tol, contrast_cap=S1_BUDGET,
                         dose_budget=1.0, medium=medium,
                         target=_bars_target_spec(period_px), K_nominal=K,
                         ablation_condition=cond_name)
            for method_id in ["BSGD", "MIL"]:
                for seed in seeds:
                    jobs.append(_job("S1", method_id, seed, config))
    return jobs


# =====================================================================
# S2: parameter sensitivity applied to cliff location specifically.
# Perturbs each of the 3 key NPDD parameters (D0, sigma, kappa) by
# +/-10/25/50%, across a K range spanning the collapse region (so K* can
# be re-estimated per perturbation and an uncertainty band on cliff
# position assembled by analysis/aggregate.py). Scoped to ONE budget
# (2x) to keep job count bounded for a "supporting" tier -- extend to
# all 3 budgets if compute allows.
# =====================================================================
S2_PARAMS = ["D0", "sigma", "kappa"]

# Reduced from the original 7-point +/-10/25/50% grid (S2_PERTURBATIONS_PCT_FULL
# below): drop +/-25%, keep the ends (+/-10%, +/-50%) and baseline. This
# still yields two distinct, interpretable claims (a "small" and a "large"
# perturbation regime) rather than collapsing to one point estimate, at
# ~1.8x the job count of a +/-25%-only cut. See docs/s2_sensitivity_notes.md
# for the two write-up caveats this tier needs regardless of which grid is
# used (one-at-a-time perturbation with no interaction terms; perturbation
# magnitudes need justifying against the literature spread once V1's
# digitized curves exist, not asserted). Full 7-point grid still available
# via perturbations_pct=S2_PERTURBATIONS_PCT_FULL.
S2_PERTURBATIONS_PCT_FULL = [-50, -25, -10, 0, 10, 25, 50]
S2_PERTURBATIONS_PCT = [-50, -10, 0, 10, 50]
S2_BUDGET = 2.0
# Collapse-region K's from the M1 grid -- where a shift in K* is actually
# observable; the same rationale as M1's "dense insert" for this range.
# Trimmed to the innermost 4 (dropping the two outermost on each side) as
# part of the same compute-budget cut -- full 8-point grid kept below for
# explicit opt-in.
S2_K_POINTS_FULL = [3.5, 4.25, 4.6, 5.0, 5.24, 5.6, 6.0, 6.5]
S2_K_POINTS = [4.6, 5.0, 5.24, 5.6]


def build_S2_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, perturbations_pct=None, K_points=None) -> list[dict]:
    seeds = seeds if seeds is not None else [0, 1, 2]
    perturbations_pct = perturbations_pct if perturbations_pct is not None else S2_PERTURBATIONS_PCT
    K_points = K_points if K_points is not None else S2_K_POINTS
    dx = 51.2 / n_x
    jobs = []
    for param in S2_PARAMS:
        base_value = DEFAULT_MEDIUM[param]
        for pct in perturbations_pct:
            if pct == 0 and param != S2_PARAMS[0]:
                continue  # baseline (0%) is param-independent; only emit it once
            value = base_value * (1.0 + pct / 100.0)
            medium = dict(DEFAULT_MEDIUM, **{param: value})
            for K in K_points:
                period_px = period_from_K(K, dx)
                config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                             converge_tol=converge_tol, contrast_cap=S2_BUDGET,
                             dose_budget=1.0, medium=medium,
                             target=_bars_target_spec(period_px), K_nominal=K,
                             sensitivity_param=param, sensitivity_pct=pct)
                for method_id in ["BSGD", "MIL"]:
                    for seed in seeds:
                        jobs.append(_job("S2", method_id, seed, config))
    return jobs


# =====================================================================
# V1: NPDD solver validation vs. published data. JOB CONFIGS ONLY --
# NOT execution-ready through run_job()/methods.run_method() (no
# optimizer/method-registry concept applies; this characterizes the
# forward solver directly, matching experiments/f1_validate_twin.py's
# existing approach). Needs a dedicated validation runner (follow-up).
# Still blocked, independent of this restructure, on you digitizing the
# published curves (data/literature/README.md) to compare against.
# =====================================================================
V1_K_GRID = [2.0, 6.0, 12.0, 20.0]  # rad/um, matches f1_validate_twin.py
V1_T_GRID = [1, 2, 4, 6, 8, 10, 14, 18]  # exposure times, matches f1_validate_twin.py


def build_V1_jobs(n_x: int = 1024, dx: float = 0.05) -> list[dict]:
    """K x exposure-time grid for DE-growth-curve validation. method_id
    is the fixed sentinel "TWIN" (deterministic forward simulation, not
    an optimizer method) so these jobs still fit the schema's
    {experiment_id}/{config_hash}/{method_id}_seed{N}.json path without
    a parallel path-naming scheme; seed is fixed at 0 (no randomness in
    a forward-only growth-curve simulation)."""
    jobs = []
    for K in V1_K_GRID:
        for t in V1_T_GRID:
            config = dict(n_x=n_x, dx=dx, lam_um=0.405, medium=DEFAULT_MEDIUM,
                         K_nominal=K, t_total=t, kind="validation_growth")
            jobs.append(_job("V1", "TWIN", 0, config))
    return jobs


# =====================================================================
# V2: Kogelnik / BPM / RCWA regime map. JOB CONFIGS ONLY -- NOT
# execution-ready: needs a genuine 3-way comparison (Kogelnik closed-form
# vs. SlabBPM split-step vs. RCWA) that does not exist yet.
# rcwa_crosscheck.py's E7 grid only compares Kogelnik vs. RCWA (2-way);
# adding BPM is real new code (follow-up), not a config change. Job
# configs mirror E7's grid (same K/dn/geometry values) for continuity
# with data already collected under that grid.
# =====================================================================
V2_K_GRID = [2.0, 4.0, 6.0, 8.0, 12.0]  # matches results_rcwa_e7.json's grid
V2_DN_GRID = [1.0e-3, 3.5e-3, 6.0e-3]  # matches E4's dn_max sweep values
V2_GEOMETRIES = ["unslanted_bragg", "unslanted_normal", "slanted20_bragg"]


def build_V2_jobs(n_x: int = 1024, dx: float = 0.05) -> list[dict]:
    jobs = []
    for K in V2_K_GRID:
        for dn in V2_DN_GRID:
            for geom in V2_GEOMETRIES:
                config = dict(n_x=n_x, dx=dx, lam_um=0.405, K_nominal=K, dn=dn,
                             geometry=geom, kind="validation_regime_map")
                jobs.append(_job("V2", "TWIN", 0, config))
    return jobs


# V3 (RCWA cross-check) is NOT a run_job()-shaped manifest -- see
# experiments/rcwa_crosscheck.py's own docstring. Real data already
# exists: results_rcwa.json (3-case) + results_rcwa_e7.json (90-case
# grid, formerly labeled "E7" -- same data, relabeled V3 in prose/docs
# going forward, file names unchanged since raw results are append-only
# and renaming a committed file isn't a "new run").
def build_V3_jobs() -> list[dict]:
    """No manifest jobs -- returns []. V3's real execution path is
    `python experiments/rcwa_crosscheck.py` (3-case) and
    `python experiments/rcwa_crosscheck.py e7` (90-case grid), both
    already run. This function exists only so V3 has an entry in
    BUILDERS for uniform tooling (probe/build_all_jobs iterate over
    BUILDERS); it contributes 0 jobs and 0 compute to those.
    """
    return []


def build_all_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4) -> list[dict]:
    """M1/M2/S1/S2 only -- the execution-ready tiers. V1/V2/V3 are
    excluded (V1/V2 have no runner yet; V3 runs via a separate script)."""
    jobs = []
    jobs += build_M1_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_M2_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_S1_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_S2_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    return jobs


BUILDERS = {
    "M1": build_M1_jobs, "M2": build_M2_jobs,
    "S1": build_S1_jobs, "S2": build_S2_jobs,
}

# V1/V2/V3 deliberately NOT in BUILDERS: BUILDERS feeds run_manifest.py's
# --manifest CLI choices and probe(), which both assume the
# run_job()/methods.run_method() execution path. Listing V1/V2 there
# would let --manifest V1 "succeed" by running 0 jobs silently -- worse
# than a clear NotImplementedError. See VALIDATION_BUILDERS below for the
# config-only builders, used directly (not through run_manifest.py) until
# their runners exist.
VALIDATION_BUILDERS = {"V1": build_V1_jobs, "V2": build_V2_jobs, "V3": build_V3_jobs}


if __name__ == "__main__":
    print("Execution-ready (run_manifest.py):")
    for name, fn in BUILDERS.items():
        print(f"  {name}: {len(fn())} jobs")
    print(f"  ALL (M1+M2+S1+S2): {len(build_all_jobs())} jobs")
    print("\nConfig-only (no runner yet, or runs via a separate script):")
    for name, fn in VALIDATION_BUILDERS.items():
        n = len(fn()) if name != "V3" else "n/a (separate script, see build_V3_jobs docstring)"
        print(f"  {name}: {n}")
