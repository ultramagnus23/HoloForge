"""
Phase 2 method registry: uniform dispatch over GS/BSGD/LPC/MIL/ORC/ORU for
the manifest runner (experiments/run_manifest.py). Each entry point takes
the same config-shaped arguments and returns the same shape of result, so
the runner can compute PSNR/DE/contrast-stats and write the Phase-1.2
schema generically without a per-method special case.

Registry
--------
GS   media_blind_gs           -- phase-optimized, naive linear exposure map
BSGD media_blind_sgd          -- SGD on an ideal linear medium, eval on twin
LPC  linear_precomp           -- closed-form 1/H(K) pre-compensation
SAT  sat_sgd                  -- SGD through a saturation-only surrogate twin,
                                 eval on the real twin (cheap-model control)
MIL  media_in_the_loop        -- ours
ORC  oracle_ideal             -- constrained oracle (E>=0, dose+contrast)
ORU  oracle_unconstrained     -- free dn optimization, only dn_max-bounded

Originally named M1-M5b; renamed to avoid colliding with the experiment-
tier names M1-M3 (main cliff/budget comparison, see experiments/manifest.py)
introduced by the V1-V3/M1-M3/S1-S2 restructure -- "M1" meaning both a
method and an experiment tier in the same job dict was a real ambiguity,
not just a style choice.

GS and LPC have no optimization loop (`iterations_run: 0`, empty loss
history) -- this is intentional (closed-form methods), not a missing
feature.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from holomedia import (NPDDRecorder, SlabBPM,
                       media_in_the_loop, media_blind_sgd, media_blind_gs,
                       oracle_ideal, oracle_unconstrained, linear_precomp,
                       sat_sgd, psnr, psnr_si, diffraction_efficiency)

METHOD_IDS = ["GS", "BSGD", "LPC", "MIL", "SAT", "ORC", "ORU"]

METHOD_NAMES = {
    "GS": "media_blind_gs",
    "BSGD": "media_blind_sgd",
    "LPC": "linear_precomp",
    "MIL": "media_in_the_loop",
    "SAT": "sat_sgd",
    "ORC": "oracle_ideal",
    "ORU": "oracle_unconstrained",
}


def contrast_stats(E: torch.Tensor) -> dict:
    """Realized exposure contrast statistics -- Phase 1.2 schema field,
    needed by Phase 4's empirical headroom-closure analysis."""
    mean = float(E.mean())
    if mean <= 0:
        return dict(max_over_mean=None, p95_over_mean=None)
    p95 = float(torch.quantile(E, 0.95))
    return dict(max_over_mean=float(E.max()) / mean, p95_over_mean=p95 / mean)


def run_method(method_id: str, target: torch.Tensor, recorder: NPDDRecorder,
              bpm: SlabBPM, seed: int, n_iters: int = 800, lr: float = 5e-2,
              dose_budget: float = 1.0, contrast_cap: float | None = None,
              converge_tol: float | None = None, log_every: int = 50) -> dict:
    """Run one method, return a dict with everything the Phase-1.2 schema
    needs EXCEPT git hash / device / wall-clock (added by the caller, since
    those are orchestration concerns, not physics ones)."""
    if method_id not in METHOD_IDS:
        raise ValueError(f"unknown method_id {method_id!r}, expected one of {METHOD_IDS}")

    mask = (target > 0.05).double()
    history = []
    early_stop_reason = "n/a"
    iterations_run = n_iters
    extra = {}

    if method_id == "GS":
        E, recon = media_blind_gs(target, recorder, bpm, dose_budget=dose_budget,
                                  seed=seed, contrast_cap=contrast_cap)
        iterations_run = 0
        early_stop_reason = "closed_form_no_optimization"

    elif method_id == "BSGD":
        E, recon, history = media_blind_sgd(target, recorder, bpm, n_iters=n_iters,
                                            lr=lr, dose_budget=dose_budget, seed=seed,
                                            contrast_cap=contrast_cap, log_every=log_every)
        early_stop_reason = "n_iters_exhausted"

    elif method_id == "LPC":
        E, recon = linear_precomp(target, recorder, bpm, dose_budget=dose_budget,
                                  contrast_cap=contrast_cap)
        iterations_run = 0
        early_stop_reason = "closed_form_no_optimization"

    elif method_id == "MIL":
        E, recon, history = media_in_the_loop(target, recorder, bpm, n_iters=n_iters,
                                              lr=lr, dose_budget=dose_budget, seed=seed,
                                              log_every=log_every, verbose=False,
                                              converge_tol=converge_tol,
                                              contrast_cap=contrast_cap)
        iterations_run = history[-1][0] if history else 0
        early_stop_reason = ("converge_tol" if (converge_tol is not None and
                             iterations_run < n_iters - 1) else "n_iters_exhausted")

    elif method_id == "SAT":
        E, recon, history = sat_sgd(target, recorder, bpm, n_iters=n_iters,
                                    lr=lr, dose_budget=dose_budget, seed=seed,
                                    log_every=log_every,
                                    converge_tol=converge_tol,
                                    contrast_cap=contrast_cap)
        iterations_run = history[-1][0] if history else 0
        early_stop_reason = ("converge_tol" if (converge_tol is not None and
                             iterations_run < n_iters - 1) else "n_iters_exhausted")
        # Provenance for the surrogate's one-parameter calibration: a
        # SAT number is only interpretable next to how well the pointwise
        # model could fit the real twin in the first place, so the fit is
        # carried in the result row rather than left implicit.
        extra = dict(sat_fit=dict(a_eff=getattr(sat_sgd, "last_a_eff", None),
                                  nrmse=getattr(sat_sgd, "last_fit_nrmse", None)))

    elif method_id == "ORC":
        E, recon = oracle_ideal(target, recorder, bpm, n_iters=n_iters, lr=lr,
                                dose_budget=dose_budget, seed=seed,
                                contrast_cap=contrast_cap)
        early_stop_reason = "n_iters_exhausted"

    elif method_id == "ORU":
        E, recon = oracle_unconstrained(target, recorder, bpm, n_iters=n_iters,
                                        lr=lr, seed=seed)
        early_stop_reason = "n_iters_exhausted"

    return dict(
        method_id=method_id, method_name=METHOD_NAMES[method_id],
        # psnr is now the SCALE-INVARIANT metric (holomedia.optimize.psnr_si),
        # which is exactly the objective every optimizer minimizes -- see the
        # objective-alignment note in optimize.py. psnr_maxnorm_legacy is the
        # OLD max-normalized metric, retained per-row so pre-fix and post-fix
        # result sets can be compared directly instead of the change being an
        # unexplainable jump in the headline number.
        psnr=psnr_si(recon, target),
        psnr_maxnorm_legacy=psnr(recon, target),
        diffraction_efficiency=diffraction_efficiency(recon, mask),
        loss_history=history, iterations_run=iterations_run,
        early_stop_reason=early_stop_reason,
        contrast=contrast_stats(E),
        **extra,
    )
