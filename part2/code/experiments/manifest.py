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
    a twin-validation script that has since been retired; V2
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
import os

DEFAULT_MEDIUM = dict(D0=0.1, sigma=0.08, kappa=2.0, gamma=1.0, dn_max=3.5e-3,
                      k_bleach=0.2, alpha_D=1.0, shrinkage=0.005,
                      thickness=30.0, n0=1.5)

ALL_METHODS = ["GS", "BSGD", "RSGD", "LPC", "GPC", "MIL", "SAT", "ORC", "ORU"]

PAPER_SEEDS = [0, 1, 2]

S1_K_POINTS = [1.308996938995747, 3.9269908169872414, 5.235987755982988]

COMPUTE_MATCH_RATIO = 21.7


def config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


FROZEN_N_Z = 128
FROZEN_SLANT_DEG = 0.0


def _job(experiment_id, method_id, seed, config):
    config = dict(config)
    config.setdefault("n_z", FROZEN_N_Z)
    config.setdefault("slant_deg", FROZEN_SLANT_DEG)
    return dict(experiment_id=experiment_id, method_id=method_id, seed=seed,
               config=config, config_hash=config_hash(config))


def _bars_target_spec(period_px: int) -> dict:
    return dict(kind="bars", period_px=period_px)


def K_from_period(period_px: int, dx: float) -> float:
    return 2.0 * math.pi / (period_px * dx)


def period_from_K(K: float, dx: float) -> int:
    """Nearest EVEN pixel period for a requested K.

    Even, because build_target renders bars as (x // (period_px // 2)) % 2,
    whose true period is 2*(period_px//2). For ODD period_px that floor
    silently discards the last pixel: period_px=17 and 16 both render
    half-period 8, i.e. the SAME target. On the previous grid that
    collapsed 17 nominal K points onto 13 distinct targets (5.0 with
    5.236; 6.5 with 7.0; 7.3, 7.6 and 7.854 all together) and made the
    realized K differ from the nominal one by up to +7.6% -- so the cliff
    plot's x-axis was wrong precisely at the high-K end where the budget
    comparison happens, and aggregate.py treated duplicated experiments as
    independent points on the gain curve.

    Rounding to even makes period_px//2 exact, so the rendered period is
    period_px and the realized K is exactly 2*pi/(period_px*dx).
    """
    p = round(2.0 * math.pi / (K * dx))
    p = 2 * round(p / 2)
    return max(4, p)


def K_from_period_exact(period_px: int, dx: float) -> float:
    """The K a given EVEN pixel period actually realizes. Inverse of
    period_from_K on the realizable lattice."""
    return 2.0 * math.pi / (period_px * dx)


_CLIFF_PERIODS_PX = [64, 48, 36, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 8]


def _cliff_K_grid(dx: float) -> list[float]:
    return sorted(round(K_from_period_exact(p, dx), 6) for p in _CLIFF_PERIODS_PX)


def build_M1_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, methods=None,
                  rsgd_tv_weight: dict[float, float] | None = None,
                  periods_px: list[int] | None = None) -> list[dict]:
    """Cliff x budget grid, ITERATION-matched arms: BSGD (media-unaware)
    and MIL (media-aware) both get the same n_iters budget. This is the
    original cliff/budget design (formerly build_E1_jobs), unchanged
    science -- only the experiment_id changed (E1 -> M1) and the method
    registry codes changed (M2/M4 -> BSGD/MIL) to avoid the tier-name
    collision.

    rsgd_tv_weight: {budget: tv_weight}, the per-budget regularization
    strength selected by experiments/tune_regularized_sgd.py's small
    grid search (WP3 item 1, tuned ON THE TWIN -- disclosed, not hidden).
    Only consulted for method_id="RSGD". If not passed explicitly
    (None), auto-loaded from results/summary/rsgd_tv_weight.json if that
    file exists -- so BUILDERS["M1"] (run_manifest.py's generic
    dispatch, which does not know about this kwarg) still picks up the
    real tuned values once tune_regularized_sgd.py has been run, without
    every call site needing to be updated. Falls back to tv_weight=0.0
    (plain BSGD) per budget if the file does not exist yet.
    """
    if rsgd_tv_weight is None:
        _tv_path = os.path.join(os.path.dirname(__file__), "..",
                                "results", "summary", "rsgd_tv_weight.json")
        if os.path.exists(_tv_path):
            with open(_tv_path) as _f:
                _loaded = json.load(_f)
            rsgd_tv_weight = {float(k): v for k, v in _loaded["best_per_budget"].items()}
    seeds = seeds if seeds is not None else PAPER_SEEDS
    methods = methods if methods is not None else ALL_METHODS
    dx = 51.2 / n_x
    all_K = _cliff_K_grid(dx)
    if periods_px is not None:
        keep = {round(K_from_period_exact(p, dx), 6) for p in periods_px}
        all_K = [K for K in all_K if round(K, 6) in keep]
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
                if method_id == "RSGD":
                    base_config["tv_weight"] = (
                        (rsgd_tv_weight or {}).get(budget, 0.0))
                if method_id in ("GS", "LPC", "GPC"):
                    for_seeds = [0]
                else:
                    for_seeds = seeds
                for seed in for_seeds:
                    jobs.append(_job("M1", method_id, seed, base_config))
    return jobs


M1A_PERIODS_PX = [64, 36, 30, 26, 22, 18, 14, 8]
M1B_PERIODS_PX = [p for p in _CLIFF_PERIODS_PX if p not in M1A_PERIODS_PX]


def build_M1A_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4) -> list[dict]:
    return build_M1_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol, periods_px=M1A_PERIODS_PX)


def build_M1B_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4) -> list[dict]:
    return build_M1_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol, periods_px=M1B_PERIODS_PX)


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

    SAT is included here for COVERAGE, not for arm-matching, and the
    distinction matters. The sub-cliff point K = 1.31 rad/um lies BELOW
    M1's grid minimum of 1.96, so a SAT arm run only on M1 has no
    sub-cliff data -- precisely where media-in-the-loop's advantage is
    largest and the surrogate is therefore most interesting. SAT runs at
    MIL's iteration budget, not BSGD's compute-matched one: it is a
    cheaper MODEL, not a cheaper budget, so there is nothing to
    compute-match it to. That means the M2 SAT-vs-BSGD comparison hands
    BSGD ~21.7x more iterations than SAT gets, which makes it a
    conservative comparison for the surrogate rather than a flattering
    one.
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
                jobs.append(_job("M2", "SAT", seed, mil_config))
    return jobs


S1_BUDGET = 2.0

S1_CONDITIONS = {
    "baseline": {},
    "no_nonlocality": dict(sigma=0.0),
    "no_diffusion": dict(D0=0.0),
    "no_dye_depletion": dict(k_bleach=0.0),
    "no_saturation": dict(linearize_index_map=True),
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



S1X_CONDITIONS = {
    "baseline": {},
    "no_monomer_depletion": dict(fixed_monomer=True),
    "no_dye_depletion": dict(k_bleach=0.0),
    "no_saturation": dict(linearize_index_map=True),
    "only_tanh": dict(fixed_monomer=True, k_bleach=0.0),
    "only_dye": dict(fixed_monomer=True, linearize_index_map=True),
    "only_monomer": dict(k_bleach=0.0, linearize_index_map=True),
    "linear_recording": dict(fixed_monomer=True, k_bleach=0.0,
                             linearize_index_map=True),
    "linear_slope_matched": dict(fixed_monomer=True, k_bleach=0.0,
                                 linearize_index_map=True, kappa=1.0 / 15.0),
    "no_monomer_depletion_matched": dict(fixed_monomer=True,
                                         kappa=0.2 / (1.0 - math.exp(-2.0))),
    "only_dye_matched": dict(fixed_monomer=True, linearize_index_map=True,
                             kappa=0.2 / (1.0 - math.exp(-2.0))),
    "only_tanh_matched": dict(fixed_monomer=True, k_bleach=0.0, kappa=0.1),
}


def build_S1X_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                   seeds=None) -> list[dict]:
    seeds = seeds if seeds is not None else [0, 1, 2]
    dx = 51.2 / n_x
    jobs = []
    for seed in seeds:
        for cond_name, overrides in S1X_CONDITIONS.items():
            medium = dict(DEFAULT_MEDIUM, **overrides)
            for K in S1_K_POINTS:
                period_px = period_from_K(K, dx)
                config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                              converge_tol=converge_tol, contrast_cap=S1_BUDGET,
                              dose_budget=1.0, medium=medium,
                              target=_bars_target_spec(period_px), K_nominal=K,
                              ablation_condition=cond_name)
                for method_id in ["BSGD", "MIL"]:
                    jobs.append(_job("S1X", method_id, seed, config))
    return jobs

NZ_VALUES = [32, 256, 128, 64]
NZ_K_POINTS = [1.963495, 3.926991, 5.235988]
NZ_BUDGET = 2.0


def build_NZ_jobs(n_x: int = 512, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, n_z_values=None, K_points=None,
                  experiment_id: str = "NZ") -> list[dict]:
    seeds = seeds if seeds is not None else PAPER_SEEDS
    n_z_values = n_z_values if n_z_values is not None else NZ_VALUES
    K_points = K_points if K_points is not None else NZ_K_POINTS
    dx = 51.2 / n_x
    jobs = []
    for seed in seeds:
        for n_z in n_z_values:
            for K in K_points:
                period_px = period_from_K(K, dx)
                config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                              converge_tol=converge_tol, contrast_cap=NZ_BUDGET,
                              dose_budget=1.0, medium=DEFAULT_MEDIUM,
                              target=_bars_target_spec(period_px), K_nominal=K,
                              arm="iteration_matched", n_z=n_z)
                for method_id in ["BSGD", "MIL"]:
                    jobs.append(_job(experiment_id, method_id, seed, config))
    return jobs


NZ1024_K_POINTS = [1.963495, 3.926991]
NZ1024_N_Z = [32, 128]


def build_NZ1024_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                      seeds=None) -> list[dict]:
    return build_NZ_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol,
                         seeds=seeds, n_z_values=NZ1024_N_Z,
                         K_points=NZ1024_K_POINTS, experiment_id="NZ_1024")


NZ0_K_POINTS = [1.963495, 3.926991, 5.235988]
NZ0_N_Z = [32, 128, 256]


def build_NZ0_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                   seeds=None) -> list[dict]:
    return build_NZ_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol,
                         seeds=seeds if seeds is not None else [0],
                         n_z_values=NZ0_N_Z, K_points=NZ0_K_POINTS,
                         experiment_id="NZ0")


S8_SLANTS_DEG = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
S8_K_POINTS = [1.963495, 3.926991]
S8_BUDGET = 2.0


def build_S8_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None) -> list[dict]:
    seeds = seeds if seeds is not None else PAPER_SEEDS
    dx = 51.2 / n_x
    jobs = []
    for slant in S8_SLANTS_DEG:
        for K in S8_K_POINTS:
            period_px = period_from_K(K, dx)
            config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                          converge_tol=converge_tol, contrast_cap=S8_BUDGET,
                          dose_budget=1.0, medium=DEFAULT_MEDIUM,
                          target=_bars_target_spec(period_px), K_nominal=K,
                          arm="iteration_matched", slant_deg=slant)
            for method_id in ["BSGD", "MIL"]:
                for seed in seeds:
                    jobs.append(_job("S8", method_id, seed, config))
    return jobs


S2_PARAMS = ["D0", "sigma", "kappa"]

S2_PERTURBATIONS_PCT_FULL = [-50, -25, -10, 0, 10, 25, 50]
S2_PERTURBATIONS_PCT = [-50, -10, 0, 10, 50]
S2_BUDGET = 2.0
S2_PERIODS_FULL_PX = [36, 32, 30, 28, 26, 24, 22, 20]
S2_PERIODS_PX = [28, 26, 24, 22]
S2_K_POINTS_FULL = [round(K_from_period_exact(p, 51.2 / 1024), 6) for p in S2_PERIODS_FULL_PX]
S2_K_POINTS = [round(K_from_period_exact(p, 51.2 / 1024), 6) for p in S2_PERIODS_PX]


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
                continue
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


def build_M1C_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4) -> list[dict]:
    return [j for j in build_M1B_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
            if abs(j["config"]["K_nominal"] - 3.926991) < 1e-3]


def build_M2R_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4) -> list[dict]:
    return [j for j in build_M2_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
            if j["config"]["contrast_cap"] == 2.0]


S2R_K_POINTS = [round(K_from_period_exact(24, 51.2 / 1024), 6)]


def build_S2R_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4) -> list[dict]:
    return build_S2_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol, K_points=S2R_K_POINTS)


S3_PARAMS = ["D0", "sigma", "kappa", "dn_max"]

S3_PERTURBATIONS_PCT = [-50, -25, -10, 0, 10, 25, 50]

def _dn_max_disagreement_factor(default: float = 2.7) -> float:
    path = os.path.join(os.path.dirname(__file__), "..", "results_literature_fit.json")
    if not os.path.exists(path):
        return default
    with open(path) as f:
        fits = json.load(f).get("fits", [])
    bayfol = [fit for fit in fits if "bruder2017" in fit.get("file", "")
             and fit.get("second_param") == "dn_max"]
    if len(bayfol) != 2:
        return default
    vals = sorted(fit["second_param_fit"] for fit in bayfol)
    return vals[1] / vals[0] if vals[0] > 0 else default


DN_MAX_LITERATURE_DISAGREEMENT_FACTOR = _dn_max_disagreement_factor()
_dn_max_hi_pct = round((DN_MAX_LITERATURE_DISAGREEMENT_FACTOR - 1.0) * 100)
_dn_max_lo_pct = round((1.0 / DN_MAX_LITERATURE_DISAGREEMENT_FACTOR - 1.0) * 100)
S3_PERTURBATIONS_PCT_DN_MAX = sorted(set(
    [_dn_max_lo_pct, -50, -25, -10, 0, 10, 25, 50, 100, _dn_max_hi_pct]))

S3_BUDGET = 2.0
S3_K_POINTS = S1_K_POINTS


def s3_perturbations_for(param: str) -> list[int]:
    return (S3_PERTURBATIONS_PCT_DN_MAX if param == "dn_max"
            else S3_PERTURBATIONS_PCT)


def build_S3_conditions(n_x: int = 1024) -> list[dict]:
    """The EVALUATION grid: one entry per (param, pct) mismatch condition.

    Returns condition dicts, not run_job()-shaped jobs, because a single
    optimized exposure is evaluated against every one of these -- the
    design work is shared across the whole list, which is what makes this
    tier cost forward passes rather than optimizations.
    """
    conds = []
    for param in S3_PARAMS:
        for pct in s3_perturbations_for(param):
            if pct == 0 and param != S3_PARAMS[0]:
                continue
            value = DEFAULT_MEDIUM[param] * (1.0 + pct / 100.0)
            conds.append(dict(mismatch_param=param, mismatch_pct=pct,
                              medium=dict(DEFAULT_MEDIUM, **{param: value})))
    return conds


S6_PARAMS = S3_PARAMS
S6_N_DRAWS = 30
S6_PCT_RANGE = 25.0


def build_S6_joint_conditions(n_draws: int = S6_N_DRAWS, pct_range: float = S6_PCT_RANGE,
                              seed: int = 0) -> list[dict]:
    """N_DRAWS independent joint perturbations, one Uniform(-pct_range,
    +pct_range)% draw per parameter per trial, fully reproducible (fixed
    numpy seed) -- rerunning this function always returns the identical
    set of conditions."""
    import numpy as np
    rng = np.random.RandomState(seed)
    conds = []
    for draw_id in range(n_draws):
        pct_by_param = {p: float(rng.uniform(-pct_range, pct_range)) for p in S6_PARAMS}
        medium = dict(DEFAULT_MEDIUM)
        for p, pct in pct_by_param.items():
            medium[p] = DEFAULT_MEDIUM[p] * (1.0 + pct / 100.0)
        conds.append(dict(draw_id=draw_id, pct_by_param=pct_by_param, medium=medium))
    return conds


def s6_result_config(design_config: dict, cond: dict) -> dict:
    """Same shape as s3_result_config: the design config with the joint-
    perturbed evaluation medium substituted in and the draw's per-
    parameter percentages attached (as a sorted-key-stable string, since
    config_hash needs a JSON-stable, not a Python dict-ordering-dependent,
    representation)."""
    pct_str = ",".join(f"{p}={cond['pct_by_param'][p]:.4f}" for p in sorted(S6_PARAMS))
    return dict(design_config, medium=cond["medium"], draw_id=cond["draw_id"],
               pct_by_param_str=pct_str, design_medium="nominal", arm="joint_mismatch_eval")


def build_S3_designs(n_x: int = 1024, n_iters: int = 800,
                     converge_tol: float = 1e-4, seeds=None,
                     methods=None, K_points=None) -> list[dict]:
    """The DESIGN stage: what must actually be optimized, once each, at
    theta_nominal. Deliberately tiny (len(methods) x len(K) x len(seeds));
    everything else in S3 is forward evaluation of these exposures."""
    seeds = seeds if seeds is not None else PAPER_SEEDS
    methods = methods if methods is not None else ["BSGD", "MIL"]
    K_points = K_points if K_points is not None else S3_K_POINTS
    dx = 51.2 / n_x
    jobs = []
    for K in K_points:
        period_px = period_from_K(K, dx)
        config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                      converge_tol=converge_tol, contrast_cap=S3_BUDGET,
                      dose_budget=1.0, medium=DEFAULT_MEDIUM,
                      target=_bars_target_spec(period_px), K_nominal=K,
                      design_medium="nominal", arm="mismatch_design")
        for method_id in methods:
            for seed in seeds:
                jobs.append(_job("S3", method_id, seed, config))
    return jobs


def s3_result_config(design_config: dict, cond: dict) -> dict:
    """Config recorded on an S3 RESULT row: the design config with the
    evaluation medium substituted in and the mismatch labels attached.
    n_iters is kept (it describes how the exposure was produced) but
    aggregate.py's pairing key already excludes it, so MIL and BSGD pair
    correctly even if their design budgets ever differ."""
    return dict(design_config, medium=cond["medium"],
                mismatch_param=cond["mismatch_param"],
                mismatch_pct=cond["mismatch_pct"],
                design_medium="nominal", arm="mismatch_eval")


S4_K_POINT = S1_K_POINTS[1]
S4_BUDGETS = [2.0, 8.0]
S4_TARGET_KINDS = ["spots", "random_binary"]


def build_S4_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, methods=None) -> list[dict]:
    seeds = seeds if seeds is not None else PAPER_SEEDS
    methods = methods if methods is not None else ["BSGD", "MIL"]
    dx = 51.2 / n_x
    jobs = []
    for budget in S4_BUDGETS:
        for target_kind in S4_TARGET_KINDS:
            target_spec = dict(kind=target_kind, seed=13)
            config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                         converge_tol=converge_tol, contrast_cap=budget,
                         dose_budget=1.0, medium=DEFAULT_MEDIUM,
                         target=target_spec, K_nominal=S4_K_POINT,
                         target_kind=target_kind, arm="target_ensemble")
            for method_id in methods:
                for seed in seeds:
                    jobs.append(_job("S4", method_id, seed, config))
    return jobs


S5_K_POINT = S1_K_POINTS[1]
S5_BUDGET = 2.0
S5_NOISE_STD = 0.05


def build_S5_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, methods=None) -> list[dict]:
    seeds = seeds if seeds is not None else PAPER_SEEDS
    methods = methods if methods is not None else ["BSGD", "MIL"]
    dx = 51.2 / n_x
    period_px = period_from_K(S5_K_POINT, dx)
    config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                 converge_tol=converge_tol, contrast_cap=S5_BUDGET,
                 dose_budget=1.0, medium=DEFAULT_MEDIUM,
                 target=_bars_target_spec(period_px), K_nominal=S5_K_POINT,
                 noise_std=S5_NOISE_STD, arm="noise_robustness")
    jobs = []
    for method_id in methods:
        for seed in seeds:
            jobs.append(_job("S5", method_id, seed, config))
    return jobs


V1_K_GRID = [2.0, 6.0, 12.0, 20.0]
V1_T_GRID = [1, 2, 4, 6, 8, 10, 14, 18]


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


V2_K_GRID = [2.0, 4.0, 6.0, 8.0, 12.0]
V2_DN_GRID = [1.0e-3, 3.5e-3, 6.0e-3]
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
    "M1": build_M1_jobs, "M1A": build_M1A_jobs, "M1B": build_M1B_jobs, "M2": build_M2_jobs,
    "S1": build_S1_jobs, "S1X": build_S1X_jobs, "M1C": build_M1C_jobs, "M2R": build_M2R_jobs,
    "S2": build_S2_jobs, "S2R": build_S2R_jobs,
    "S4": build_S4_jobs, "S5": build_S5_jobs,
    "NZ": build_NZ_jobs, "NZ_1024": build_NZ1024_jobs, "NZ0": build_NZ0_jobs, "S8": build_S8_jobs,
}

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
