"""
Media-in-the-loop optimization and media-blind baselines.

The design variable is the delivered exposure E(x) >= 0 under a total-dose
budget -- NOT a phase pattern. Nonnegativity is enforced by softplus
parameterization; the dose budget by renormalization each step (projected
gradient). Baselines convert a conventionally optimized phase hologram to an
exposure assuming a linear medium (the "media-blind" practice this paper
argues against).

Methods
-------
media_in_the_loop : SGD/Adam through NPDDRecorder + SlabBPM  (ours)
media_blind_gs    : Gerchberg-Saxton phase -> naive linear exposure map
media_blind_sgd   : SGD assuming ideal linear medium, evaluated on real twin
oracle_ideal      : SGD on an ideal medium, evaluated on the ideal medium
"""

from __future__ import annotations
import math
import torch

from .npdd import NPDDRecorder, MediumParams
from .diffraction import SlabBPM


# ---------------------------------------------------------------- utilities
def dose_project(E: torch.Tensor, budget: float) -> torch.Tensor:
    """Scale exposure so mean dose == budget (projection onto dose simplex)."""
    return E * (budget / (E.mean() + 1e-12))


def _seeded_init_theta(n_x: int, device, dtype, seed: int, eps: float = 1e-2):
    """Small seeded perturbation around zero, not exact zeros.

    Exact-zero init made every method's result independent of `seed` --
    Adam on deterministic zero-initialized state with deterministic (FFT/
    elementwise) forward ops has nothing left for the seed to affect, so a
    5-seed sweep was silently computing one trajectory 5 times. This keeps
    the "start near uniform dose" design intent (eps small) while making
    the optimization trajectory genuinely seed-dependent. A CPU generator
    is used regardless of `device` since torch.randn's CUDA generator path
    is not guaranteed bit-reproducible across GPU models; the resulting
    tensor is moved to `device` after sampling.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    theta = eps * torch.randn(n_x, generator=g, dtype=dtype)
    return theta.to(device).requires_grad_(True)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.max() + 1e-12)
    b = b / (b.max() + 1e-12)
    mse = torch.mean((a - b) ** 2)
    return float(10.0 * torch.log10(1.0 / (mse + 1e-12)))


def diffraction_efficiency(recon: torch.Tensor, target_mask: torch.Tensor) -> float:
    """Fraction of reconstructed power landing inside the target support."""
    return float((recon * target_mask).sum() / (recon.sum() + 1e-12))


# ------------------------------------------------------- batched utilities
def dose_project_batch(E: torch.Tensor, budget: float) -> torch.Tensor:
    """dose_project, per batch row (E: (B, n_x))."""
    return E * (budget / (E.mean(dim=-1, keepdim=True) + 1e-12))


def _seeded_init_theta_batch(n_x: int, device, dtype, seeds, eps: float = 1e-2):
    rows = []
    for s in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(s)
        rows.append(eps * torch.randn(n_x, generator=g, dtype=dtype))
    return torch.stack(rows).to(device).requires_grad_(True)


def psnr_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """psnr, per batch row (a, b: (B, n_x)) -> (B,) tensor."""
    a = a / (a.amax(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.amax(dim=-1, keepdim=True) + 1e-12)
    mse = torch.mean((a - b) ** 2, dim=-1)
    return 10.0 * torch.log10(1.0 / (mse + 1e-12))


def diffraction_efficiency_batch(recon: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """diffraction_efficiency, per batch row -> (B,) tensor."""
    return (recon * target_mask).sum(dim=-1) / (recon.sum(dim=-1) + 1e-12)


# ------------------------------------------------------------------ methods
def media_in_the_loop(target: torch.Tensor, recorder: NPDDRecorder,
                      bpm: SlabBPM, n_iters: int = 400, lr: float = 5e-2,
                      dose_budget: float = 1.0, seed: int = 0,
                      log_every: int = 50, verbose: bool = True,
                      converge_tol: float | None = None, patience: int = 3):
    """Optimize exposure through the differentiable recording twin (ours).

    converge_tol : if set, stop early once the relative change in loss
        between successive `log_every`-spaced checks stays below
        `converge_tol` for `patience` consecutive checks. Default None
        preserves the original always-run-n_iters behavior exactly.
        `history`'s final (it, loss) entry reports the stopping iteration.
    """
    device = target.device
    # softplus-parameterized exposure, initialized near uniform dose
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    history = []
    prev_loss = None
    stable_count = 0

    for it in range(n_iters):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = dose_project(E, dose_budget)
        dn = recorder(E)
        recon = bpm(dn, shrinkage=recorder.p.shrinkage)
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        loss.backward()
        opt.step()
        if it % log_every == 0:
            cur = float(loss.detach())
            history.append((it, cur))
            if verbose:
                print(f"  [media-in-the-loop] iter {it:4d}  loss {cur:.4e}")
            if converge_tol is not None and prev_loss is not None:
                rel_change = abs(prev_loss - cur) / (abs(prev_loss) + 1e-12)
                stable_count = stable_count + 1 if rel_change < converge_tol else 0
                if stable_count >= patience:
                    break
            prev_loss = cur

    with torch.no_grad():
        E = dose_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E.detach(), recon.detach(), history


def media_blind_sgd(target: torch.Tensor, recorder: NPDDRecorder, bpm: SlabBPM,
                    n_iters: int = 400, lr: float = 5e-2, dose_budget: float = 1.0,
                    seed: int = 0):
    """Optimize assuming a LINEAR ideal medium (dn = c * E), then evaluate the
    resulting exposure on the real twin. This is what current practice does."""
    device = target.device
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    c_lin = recorder.p.dn_max  # ideal linear map with same index budget

    for _ in range(n_iters):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = dose_project(E, dose_budget)
        dn_ideal = c_lin * (E - E.mean())          # linear, zero-mean modulation
        recon = bpm(dn_ideal, shrinkage=0.0)       # blind: no shrinkage either
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        loss.backward()
        opt.step()

    with torch.no_grad():
        E = dose_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon_real = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E.detach(), recon_real.detach()


def media_blind_gs(target: torch.Tensor, recorder: NPDDRecorder, bpm: SlabBPM,
                   n_iters: int = 200, dose_budget: float = 1.0, seed: int = 0):
    """Classic GS in the far field -> phase -> naive exposure conversion,
    evaluated on the real twin. The weakest but most common practice.

    seed controls the random initial phase (previously drawn from
    whatever the global RNG state happened to be after other calls in the
    same script -- reproducible now, but note this changes prior runs'
    exact GS numbers, which were never actually seed-controlled)."""
    device = target.device
    n = recorder.n_x
    amp_t = torch.sqrt(target / (target.sum() + 1e-12)).to(torch.complex128)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    field = torch.exp(1j * 2 * math.pi *
                      torch.rand(n, generator=g, dtype=torch.float64).to(device))
    for _ in range(n_iters):
        far = torch.fft.fft(field)
        far = amp_t * torch.exp(1j * torch.angle(far))
        near = torch.fft.ifft(far)
        field = torch.exp(1j * torch.angle(near))
    phase = torch.angle(field).real
    # naive conversion: exposure proportional to desired phase (mod 2pi),
    # assuming dn linear in dose -- exactly the assumption we critique.
    E = dose_project((phase - phase.min()) + 1e-6, dose_budget)
    with torch.no_grad():
        recon_real = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E, recon_real


def oracle_ideal(target: torch.Tensor, recorder: NPDDRecorder, bpm: SlabBPM,
                 n_iters: int = 400, lr: float = 5e-2, dose_budget: float = 1.0,
                 seed: int = 0):
    """Upper bound: linear medium optimized AND evaluated as linear."""
    device = target.device
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    c_lin = recorder.p.dn_max

    for _ in range(n_iters):
        opt.zero_grad()
        E = dose_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(c_lin * (E - E.mean()), shrinkage=0.0)
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        loss.backward()
        opt.step()

    with torch.no_grad():
        E = dose_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(c_lin * (E - E.mean()), shrinkage=0.0)
    return E.detach(), recon.detach()


# -------------------------------------------------------------- batched methods
# Batched counterparts: one (B, n_x) theta run through one Python/Adam loop
# instead of B separate (n_x,) thetas run in B sequential Python loops.
# NPDDRecorder/SlabBPM have no cross-batch-row mixing (per-row FFT + the
# per-row D_bar fix in npdd.py), and the backward scalar below is always
# loss_per_row.SUM() rather than .mean() specifically so each row's Adam
# trajectory is bit-identical to that (target, seed) run through the
# unbatched function alone -- verified by direct comparison, see
# tests/test_smoke.py::test_batched_matches_unbatched. Batching changes
# wall-clock only, never the numbers. No early-stopping (converge_tol):
# rows would need independently-timed stopping, not implemented here.

def media_in_the_loop_batched(targets: torch.Tensor, recorder: NPDDRecorder,
                              bpm: SlabBPM, seeds, n_iters: int = 400,
                              lr: float = 5e-2, dose_budget: float = 1.0,
                              log_every: int = 50, verbose: bool = True):
    device = targets.device
    theta = _seeded_init_theta_batch(recorder.n_x, device, torch.float64, seeds)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = targets / (targets.sum(dim=-1, keepdim=True) + 1e-12)
    history = []

    for it in range(n_iters):
        opt.zero_grad()
        E = dose_project_batch(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        dn = recorder(E)
        recon = bpm(dn, shrinkage=recorder.p.shrinkage)
        r_norm = recon / (recon.sum(dim=-1, keepdim=True) + 1e-12)
        loss_per_row = torch.mean((r_norm - t_norm) ** 2, dim=-1) * recorder.n_x
        loss_per_row.sum().backward()
        opt.step()
        if it % log_every == 0:
            cur = float(loss_per_row.mean().detach())
            history.append((it, cur))
            if verbose:
                print(f"  [batched ours] iter {it:4d}  mean loss {cur:.4e}")

    with torch.no_grad():
        E = dose_project_batch(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E.detach(), recon.detach(), history


def media_blind_sgd_batched(targets: torch.Tensor, recorder: NPDDRecorder,
                            bpm: SlabBPM, seeds, n_iters: int = 400,
                            lr: float = 5e-2, dose_budget: float = 1.0):
    device = targets.device
    theta = _seeded_init_theta_batch(recorder.n_x, device, torch.float64, seeds)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = targets / (targets.sum(dim=-1, keepdim=True) + 1e-12)
    c_lin = recorder.p.dn_max

    for _ in range(n_iters):
        opt.zero_grad()
        E = dose_project_batch(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        dn_ideal = c_lin * (E - E.mean(dim=-1, keepdim=True))
        recon = bpm(dn_ideal, shrinkage=0.0)
        r_norm = recon / (recon.sum(dim=-1, keepdim=True) + 1e-12)
        loss_per_row = torch.mean((r_norm - t_norm) ** 2, dim=-1) * recorder.n_x
        loss_per_row.sum().backward()
        opt.step()

    with torch.no_grad():
        E = dose_project_batch(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon_real = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E.detach(), recon_real.detach()


def media_blind_gs_batched(targets: torch.Tensor, recorder: NPDDRecorder,
                           bpm: SlabBPM, seeds, n_iters: int = 200,
                           dose_budget: float = 1.0):
    device = targets.device
    n = recorder.n_x
    amp_t = torch.sqrt(targets / (targets.sum(dim=-1, keepdim=True) + 1e-12)).to(torch.complex128)
    rows = []
    for s in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(s)
        rows.append(torch.rand(n, generator=g, dtype=torch.float64))
    phase0 = torch.stack(rows).to(device)
    field = torch.exp(1j * 2 * math.pi * phase0)
    for _ in range(n_iters):
        far = torch.fft.fft(field)
        far = amp_t * torch.exp(1j * torch.angle(far))
        near = torch.fft.ifft(far)
        field = torch.exp(1j * torch.angle(near))
    phase = torch.angle(field).real
    E = dose_project_batch((phase - phase.amin(dim=-1, keepdim=True)) + 1e-6, dose_budget)
    with torch.no_grad():
        recon_real = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E, recon_real


def oracle_ideal_batched(targets: torch.Tensor, recorder: NPDDRecorder,
                         bpm: SlabBPM, seeds, n_iters: int = 400,
                         lr: float = 5e-2, dose_budget: float = 1.0):
    device = targets.device
    theta = _seeded_init_theta_batch(recorder.n_x, device, torch.float64, seeds)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = targets / (targets.sum(dim=-1, keepdim=True) + 1e-12)
    c_lin = recorder.p.dn_max

    for _ in range(n_iters):
        opt.zero_grad()
        E = dose_project_batch(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(c_lin * (E - E.mean(dim=-1, keepdim=True)), shrinkage=0.0)
        r_norm = recon / (recon.sum(dim=-1, keepdim=True) + 1e-12)
        loss_per_row = torch.mean((r_norm - t_norm) ** 2, dim=-1) * recorder.n_x
        loss_per_row.sum().backward()
        opt.step()

    with torch.no_grad():
        E = dose_project_batch(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(c_lin * (E - E.mean(dim=-1, keepdim=True)), shrinkage=0.0)
    return E.detach(), recon.detach()
