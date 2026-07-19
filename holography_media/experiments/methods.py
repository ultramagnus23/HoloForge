"""
Phase 2 method registry: uniform dispatch over M1-M5b for the manifest
runner (experiments/run_manifest.py). Each entry point takes the same
config-shaped arguments and returns the same shape of result, so the
runner can compute PSNR/DE/contrast-stats and write the Phase-1.2 schema
generically without a per-method special case.

Registry
--------
M1  media_blind_gs           -- phase-optimized, naive linear exposure map
M2  media_blind_sgd          -- SGD on an ideal linear medium, eval on twin
M3  linear_precomp           -- NEW: closed-form 1/H(K) pre-compensation
M4  media_in_the_loop        -- ours
M5a oracle_ideal             -- constrained oracle (E>=0, dose+contrast)
M5b oracle_unconstrained     -- NEW: free dn optimization, only dn_max-bounded

M1 and M3 have no optimization loop (`iterations_run: 0`, empty loss
history) -- this is intentional per Phase 2's spec, not a missing feature.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from holomedia import (NPDDRecorder, SlabBPM,
                       media_in_the_loop, media_blind_sgd, media_blind_gs,
                       oracle_ideal, oracle_unconstrained, linear_precomp,
                       psnr, diffraction_efficiency)

METHOD_IDS = ["M1", "M2", "M3", "M4", "M5a", "M5b"]

METHOD_NAMES = {
    "M1": "media_blind_gs",
    "M2": "media_blind_sgd",
    "M3": "linear_precomp",
    "M4": "media_in_the_loop",
    "M5a": "oracle_ideal",
    "M5b": "oracle_unconstrained",
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

    if method_id == "M1":
        E, recon = media_blind_gs(target, recorder, bpm, dose_budget=dose_budget, seed=seed)
        iterations_run = 0
        early_stop_reason = "closed_form_no_optimization"

    elif method_id == "M2":
        E, recon, history = media_blind_sgd(target, recorder, bpm, n_iters=n_iters,
                                            lr=lr, dose_budget=dose_budget, seed=seed,
                                            contrast_cap=contrast_cap, log_every=log_every)
        early_stop_reason = "n_iters_exhausted"

    elif method_id == "M3":
        E, recon = linear_precomp(target, recorder, bpm, dose_budget=dose_budget,
                                  contrast_cap=contrast_cap)
        iterations_run = 0
        early_stop_reason = "closed_form_no_optimization"

    elif method_id == "M4":
        E, recon, history = media_in_the_loop(target, recorder, bpm, n_iters=n_iters,
                                              lr=lr, dose_budget=dose_budget, seed=seed,
                                              log_every=log_every, verbose=False,
                                              converge_tol=converge_tol,
                                              contrast_cap=contrast_cap)
        iterations_run = history[-1][0] if history else 0
        early_stop_reason = ("converge_tol" if (converge_tol is not None and
                             iterations_run < n_iters - 1) else "n_iters_exhausted")

    elif method_id == "M5a":
        E, recon = oracle_ideal(target, recorder, bpm, n_iters=n_iters, lr=lr,
                                dose_budget=dose_budget, seed=seed,
                                contrast_cap=contrast_cap)
        early_stop_reason = "n_iters_exhausted"

    elif method_id == "M5b":
        E, recon = oracle_unconstrained(target, recorder, bpm, n_iters=n_iters,
                                        lr=lr, seed=seed)
        early_stop_reason = "n_iters_exhausted"

    return dict(
        method_id=method_id, method_name=METHOD_NAMES[method_id],
        psnr=psnr(recon, target),
        diffraction_efficiency=diffraction_efficiency(recon, mask),
        loss_history=history, iterations_run=iterations_run,
        early_stop_reason=early_stop_reason,
        contrast=contrast_stats(E),
    )
