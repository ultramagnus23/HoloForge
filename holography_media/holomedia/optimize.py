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


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.max() + 1e-12)
    b = b / (b.max() + 1e-12)
    mse = torch.mean((a - b) ** 2)
    return float(10.0 * torch.log10(1.0 / (mse + 1e-12)))


def diffraction_efficiency(recon: torch.Tensor, target_mask: torch.Tensor) -> float:
    """Fraction of reconstructed power landing inside the target support."""
    return float((recon * target_mask).sum() / (recon.sum() + 1e-12))


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
    torch.manual_seed(seed)
    device = target.device
    # softplus-parameterized exposure, initialized near uniform dose
    theta = torch.zeros(recorder.n_x, dtype=torch.float64, device=device,
                        requires_grad=True)
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
    torch.manual_seed(seed)
    device = target.device
    theta = torch.zeros(recorder.n_x, dtype=torch.float64, device=device,
                        requires_grad=True)
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
                   n_iters: int = 200, dose_budget: float = 1.0):
    """Classic GS in the far field -> phase -> naive exposure conversion,
    evaluated on the real twin. The weakest but most common practice."""
    device = target.device
    n = recorder.n_x
    amp_t = torch.sqrt(target / (target.sum() + 1e-12)).to(torch.complex128)
    field = torch.exp(1j * 2 * math.pi *
                      torch.rand(n, dtype=torch.float64, device=device))
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
    torch.manual_seed(seed)
    device = target.device
    theta = torch.zeros(recorder.n_x, dtype=torch.float64, device=device,
                        requires_grad=True)
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
