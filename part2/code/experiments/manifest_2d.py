"""
Bounded 2D study: job manifest + target generators.

Scope decision (see paper's Phase 2 decision memo): NOT a full 2D
generalization of the M1 15-K x 3-budget x 3-seed x 4-method grid (costed
at thousands of GPU-hours at real image resolution -- infeasible on this
hardware by the submission deadline). Instead: 3 real 2D image targets x 3
contrast budgets x N seeds x 2 methods (MIL, BSGD -- the paper's core
paired comparison), at a fixed resolution. K is not swept because K is not
well-defined for a broadband 2D image; the 3 targets vary in spatial-
frequency content instead, playing the analogous role.

Reuses holomedia.npdd3d.NPDDRecorder3D / holomedia.diffraction3d.SlabBPM3D
unmodified -- same governing NPDD equations as the 1D pipeline, generalized
to a 2D transverse grid, already validated structurally by
experiments/showcase_3d.py. No physics changes here, so this is not gated
by the recording-chemistry freeze.

Targets are synthetic/generated (no licensed image assets), following the
same reasoning that deferred manifest.py's E5 natural-image-slice tier.
"""
from __future__ import annotations
import hashlib
import json
import math

import torch

# Same medium defaults as the 1D pipeline (experiments/manifest.py's
# DEFAULT_MEDIUM), duplicated here for the same reason: a job's config
# should be self-contained JSON, not dependent on holomedia's current
# field defaults silently drifting a previously-generated manifest's
# meaning.
DEFAULT_MEDIUM = dict(D0=0.1, sigma=0.08, kappa=2.0, gamma=1.0, dn_max=3.5e-3,
                      k_bleach=0.2, alpha_D=1.0, shrinkage=0.005,
                      thickness=30.0, n0=1.5)

BUDGETS = [2.0, 4.0, 8.0]
SEEDS = [0, 1, 2]
N = 128            # transverse grid, both axes (n_x = n_y = N)
DX = 0.15          # um/px, matches experiments/showcase_3d.py's convention
LAM_UM = 0.405
N_STEPS = 150       # NPDD PDE integration steps (showcase used 120)
N_Z = 16            # BPM depth-propagation steps (showcase used 12)
N_ITERS = 800        # optimizer iterations -- matches the 1D M1 headline
                     # grid's n_iters exactly, so "same iteration budget as
                     # the 1D study" is a true statement, not an estimate.
LR = 5e-2

TARGET_KINDS = ["disc", "checkerboard", "reschart"]


def config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


# --------------------------------------------------------------- targets
def disc_target(n: int, radius_frac: float = 0.32) -> torch.Tensor:
    """Low-spatial-frequency content: a single filled disc."""
    x = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2)
    return (r < radius_frac).double()


def checkerboard_target(n: int, period_px: int = 16) -> torch.Tensor:
    """Mid-spatial-frequency content: a single period, both axes at once --
    tests the 2D non-local kernel against a non-separable pattern, unlike
    a separable product of two 1D bar gratings."""
    x = torch.arange(n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    half = max(period_px // 2, 1)
    return (((X // half) + (Y // half)) % 2).double()


def reschart_target(n: int, periods_px=(24, 12, 6)) -> torch.Tensor:
    """Multi-zone resolution chart: n horizontal thirds, each a vertical-bar
    grating at a different period (coarse/medium/fine). The 2D analogue of
    the 1D study's K-sweep -- one image spans several spatial frequencies
    instead of one grating spanning one -- since a genuine per-target K
    is not well-defined for a broadband 2D image."""
    zone_h = n // len(periods_px)
    x = torch.arange(n)
    out = torch.zeros(n, n, dtype=torch.float64)
    for i, period in enumerate(periods_px):
        half = max(period // 2, 1)
        bars = ((x // half) % 2).double()
        row0, row1 = i * zone_h, (i + 1) * zone_h if i < len(periods_px) - 1 else n
        out[row0:row1, :] = bars.unsqueeze(0).expand(row1 - row0, n)
    return out


TARGET_FNS = {
    "disc": lambda: disc_target(N),
    "checkerboard": lambda: checkerboard_target(N, period_px=16),
    "reschart": lambda: reschart_target(N, periods_px=(24, 12, 6)),
}


# ------------------------------------------------------------------ jobs
def _job(method_id: str, seed: int, config: dict) -> dict:
    return dict(experiment_id="M1_2D", method_id=method_id, seed=seed,
               config=dict(config), config_hash=config_hash(config))


def build_2d_jobs(target_kinds=None, budgets=None, seeds=None,
                  methods=("BSGD", "MIL")) -> list[dict]:
    target_kinds = target_kinds if target_kinds is not None else TARGET_KINDS
    budgets = budgets if budgets is not None else BUDGETS
    seeds = seeds if seeds is not None else SEEDS
    jobs = []
    for kind in target_kinds:
        for budget in budgets:
            config = dict(n=N, dx=DX, lam_um=LAM_UM, n_steps=N_STEPS,
                         n_z=N_Z, n_iters=N_ITERS, lr=LR,
                         dose_budget=1.0, contrast_cap=budget,
                         medium=DEFAULT_MEDIUM, target_kind=kind)
            for method_id in methods:
                for seed in seeds:
                    jobs.append(_job(method_id, seed, config))
    return jobs
