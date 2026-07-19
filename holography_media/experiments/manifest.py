"""
Phase 1.1: job manifest builders for E1-E7.

A manifest is a flat list of job dicts:
    {experiment_id, method_id, seed, config, config_hash}
`config` fully resolves everything needed to reconstruct the target,
MediumParams, NPDDRecorder, and SlabBPM for that job -- nothing implicit.
`config_hash` is a short sha256 of the canonicalized (sorted-key) JSON of
`config`, used as the results-directory key so two configs that differ in
any field never collide and identical configs always resolve to the same
path (resume = same command -> same hashes -> same skip set).

Paper-scale defaults (master prompt Phase 3): n_x=1024, 800 Adam iters,
converge_tol=1e-4 (hard cap 1500), 5 seeds, PVA/AA-like defaults at 405nm
unless swept. These are DEFAULTS on the builder functions, not hardcoded,
so a smoke pass at n_x=256 (or smaller) is one keyword away for local
CPU verification before anything goes to Colab.
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

ALL_METHODS = ["M1", "M2", "M3", "M4", "M5a", "M5b"]
PAPER_SEEDS = [0, 1, 2, 3, 4]


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


# ---------------------------------------------------------------- E1: cliff x budget
def build_E1_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, methods=None) -> list[dict]:
    """Cliff x budget grid, the headline experiment.

    K grid: the existing 7 points union a dense insert across the collapse
    region, per master prompt Phase 3 exactly:
        {0.98, 1.31, 1.96, 2.62, 3.5, 3.93, 4.25, 4.6, 5.0, 5.24, 5.6, 6.0,
         6.5, 7.85} -- 14 unique K values (union of the 7 existing period-px
        values' K's and the 7 new dense-insert K values in rad/um, sorted).
    Budgets: contrast_cap in {2.0, 4.0, 8.0}, matching predicted_cliff's
    B_c exactly per docs/definitions.md (d).
    """
    seeds = seeds if seeds is not None else PAPER_SEEDS
    methods = methods if methods is not None else ALL_METHODS
    dx = 51.2 / n_x  # fixed physical window, matches gpu_npdd_mesh_convergence_sweep.py convention
    existing_periods = [8, 12, 16, 24, 32, 48, 64]
    existing_K = sorted(K_from_period(p, dx) for p in existing_periods)
    dense_insert_K = [3.5, 4.25, 4.6, 5.0, 5.6, 6.0, 6.5]
    all_K = sorted(set(round(k, 6) for k in existing_K + dense_insert_K))
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
                )
                if method_id in ("M1", "M3"):
                    for_seeds = [0]  # closed-form, no seed dependence worth repeating
                else:
                    for_seeds = seeds
                for seed in for_seeds:
                    jobs.append(_job("E1", method_id, seed, base_config))
    return jobs


# ---------------------------------------------------------------- E2: sigma probe
def build_E2_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None) -> list[dict]:
    """Sigma probe at K=7.85 rad/um, methods M2+M4, 5 seeds."""
    seeds = seeds if seeds is not None else PAPER_SEEDS
    dx = 51.2 / n_x
    K_target = 7.853981633974483  # period8 at this window convention
    period_px = period_from_K(K_target, dx)
    sigmas = [0.02, 0.05, 0.08, 0.12, 0.20, 0.30]

    jobs = []
    for sigma in sigmas:
        medium = dict(DEFAULT_MEDIUM, sigma=sigma)
        config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                     converge_tol=converge_tol, contrast_cap=None,
                     dose_budget=1.0, medium=medium,
                     target=_bars_target_spec(period_px), K_nominal=K_target)
        for method_id in ["M2", "M4"]:
            for seed in seeds:
                jobs.append(_job("E2", method_id, seed, config))
    return jobs


# ---------------------------------------------------------------- E3: shrinkage
def build_E3_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None) -> list[dict]:
    """Shrinkage sweep at K=3.93 rad/um, methods M2+M4, 5 seeds."""
    seeds = seeds if seeds is not None else PAPER_SEEDS
    dx = 51.2 / n_x
    K_target = 3.9269908169872414  # period16
    period_px = period_from_K(K_target, dx)
    shrinkages = [0.0, 0.01, 0.02, 0.03]

    jobs = []
    for s in shrinkages:
        medium = dict(DEFAULT_MEDIUM, shrinkage=s)
        config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                     converge_tol=converge_tol, contrast_cap=None,
                     dose_budget=1.0, medium=medium,
                     target=_bars_target_spec(period_px), K_nominal=K_target)
        for method_id in ["M2", "M4"]:
            for seed in seeds:
                jobs.append(_job("E3", method_id, seed, config))
    return jobs


# ---------------------------------------------------------------- E4: material sweeps
def build_E4_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None) -> list[dict]:
    """Delta_n_max / thickness / D0 sweeps, methods M2+M4, 3 seeds (labeled)."""
    seeds = seeds if seeds is not None else [0, 1, 2]
    dx = 51.2 / n_x
    K_target = 3.9269908169872414
    period_px = period_from_K(K_target, dx)

    axes = {
        "dn_max": [1e-3, 3.5e-3, 6e-3],
        "thickness": [10.0, 30.0, 100.0],
        "D0": [0.01, 0.1, 1.0],
    }
    jobs = []
    for field, values in axes.items():
        for v in values:
            medium = dict(DEFAULT_MEDIUM, **{field: v})
            config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                         converge_tol=converge_tol, contrast_cap=None,
                         dose_budget=1.0, medium=medium,
                         target=_bars_target_spec(period_px), K_nominal=K_target,
                         swept_field=field)
            for method_id in ["M2", "M4"]:
                for seed in seeds:
                    jobs.append(_job("E4", method_id, seed, config))
    return jobs


# ---------------------------------------------------------------- E5: targets beyond bars
def build_E5_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None, image_slice_paths=None) -> list[dict]:
    """Sparse-spot + natural-image-slice targets, all 6 methods, 3 seeds.

    image_slice_paths: list of paths to 1D-sliceable image assets. NOT
    fetched or fabricated by this builder -- per the master prompt, an
    actual CC-licensed (or DIV2K-if-licensing-permits) image must be added
    to the repo by the user first, with its source/license documented
    (data/images/README.md, not yet created). Called with image_slice_paths
    unset, this returns ONLY the sparse-spot jobs and prints a warning
    rather than silently fabricating image data or downloading anything
    (downloading files requires explicit user approval, out of scope for
    an unattended builder function).
    """
    seeds = seeds if seeds is not None else [0, 1, 2]
    dx = 51.2 / n_x
    budgets_by_operating_point = {"below_cliff": 8.0, "at_cliff": 2.0, "above_cliff": 1.0}

    jobs = []
    for op_name, budget in budgets_by_operating_point.items():
        config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                     converge_tol=converge_tol, contrast_cap=budget,
                     dose_budget=1.0, medium=DEFAULT_MEDIUM,
                     target=dict(kind="spots", n_spots=5, seed=7),
                     operating_point=op_name)
        for method_id in ALL_METHODS:
            for seed in seeds:
                jobs.append(_job("E5", method_id, seed, config))

    if not image_slice_paths:
        print("[build_E5_jobs] WARNING: no image_slice_paths supplied -- "
              "natural-image-slice targets NOT included. Add licensed image "
              "assets under data/images/ (with data/images/README.md "
              "documenting source+license) and pass their paths before "
              "this experiment is complete per the master prompt.")
    else:
        for img_path in image_slice_paths:
            for op_name, budget in budgets_by_operating_point.items():
                config = dict(n_x=n_x, dx=dx, lam_um=0.405, n_iters=n_iters,
                             converge_tol=converge_tol, contrast_cap=budget,
                             dose_budget=1.0, medium=DEFAULT_MEDIUM,
                             target=dict(kind="image_slice", path=img_path, row=None),
                             operating_point=op_name)
                for method_id in ALL_METHODS:
                    for seed in seeds:
                        jobs.append(_job("E5", method_id, seed, config))
    return jobs


# ---------------------------------------------------------------- E6: wavelength (optional)
def build_E6_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                  seeds=None) -> list[dict]:
    """Wavelength-detuning rerun at 3 seeds -- ONLY if Gate 1 budget allows.
    Not included in build_all_jobs() by default; call explicitly."""
    seeds = seeds if seeds is not None else [0, 1, 2]
    dx = 51.2 / n_x
    K_target = 3.9269908169872414
    period_px = period_from_K(K_target, dx)
    wavelengths = [0.400, 0.405, 0.420, 0.435, 0.450]

    jobs = []
    for lam in wavelengths:
        config = dict(n_x=n_x, dx=dx, lam_um=lam, n_iters=n_iters,
                     converge_tol=converge_tol, contrast_cap=None,
                     dose_budget=1.0, medium=DEFAULT_MEDIUM,
                     target=_bars_target_spec(period_px), K_nominal=K_target)
        for seed in seeds:
            jobs.append(_job("E6", "M4", seed, config))
    return jobs


# E7 (RCWA validity envelope) is NOT an optimizer manifest -- it extends
# experiments/rcwa_crosscheck.py's own grid (K x polarization x incidence x
# Delta-n), which has no seed/method-registry structure. Left as a
# standalone script extension, not a manifest job type; see repo_map.md.


def build_all_jobs(n_x: int = 1024, n_iters: int = 800, converge_tol: float = 1e-4,
                   include_E6: bool = False) -> list[dict]:
    jobs = []
    jobs += build_E1_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_E2_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_E3_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_E4_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    jobs += build_E5_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    if include_E6:
        jobs += build_E6_jobs(n_x=n_x, n_iters=n_iters, converge_tol=converge_tol)
    return jobs


BUILDERS = {
    "E1": build_E1_jobs, "E2": build_E2_jobs, "E3": build_E3_jobs,
    "E4": build_E4_jobs, "E5": build_E5_jobs, "E6": build_E6_jobs,
}


if __name__ == "__main__":
    for name, fn in BUILDERS.items():
        jobs = fn()
        print(f"{name}: {len(jobs)} jobs")
    print(f"ALL (E1-E5): {len(build_all_jobs())} jobs")
