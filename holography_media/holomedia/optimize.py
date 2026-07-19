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


def contrast_project(E: torch.Tensor, budget: float,
                     contrast_cap: float | None = None,
                     n_passes: int = 20) -> torch.Tensor:
    """Project E onto {E >= 0, mean(E) == budget, max(E)/mean(E) <= contrast_cap}.

    This constraint did NOT previously exist anywhere in the codebase:
    dose_project alone only fixes the mean; peak/mean ("contrast") is
    scale-invariant under that rescaling, so nothing capped it (verified:
    dose_project(E, b) has identical max/mean for any b). The paper's
    analytic K_c(B_c) prediction (NPDDRecorder.predicted_cliff) takes a
    contrast-budget B_c as input, but no optimizer run ever enforced that
    B_c as an actual constraint on the exposure -- this closes that gap.

    contrast_cap=None preserves the exact old behavior (pure dose_project,
    no contrast constraint) for backward compatibility with every existing
    caller/test.

    Implementation: alternate clamping the peak to contrast_cap*budget and
    re-normalizing the mean back to budget. Clamping changes the mean (mass
    above the ceiling is removed), so a single pass does not exactly hit
    both constraints simultaneously; convergence is geometric (~2x error
    reduction per pass, verified) but the rate depends on the input's
    peakedness -- simple inputs converge to ~1e-7 within 6 passes, harder
    (e.g. post-frequency-boost) inputs only to ~3e-3 at 6 passes and need
    ~20 for <1e-6. Both operations (clamp, rescale) are differentiable, so
    this is safe to use inside an unrolled optimization loop; 20 passes of
    elementwise clamp+rescale is negligible cost next to one NPDD/BPM
    forward pass.
    """
    E = dose_project(E, budget)
    if contrast_cap is None:
        return E
    ceiling = contrast_cap * budget
    for _ in range(n_passes):
        E = torch.clamp(E, max=ceiling)
        E = dose_project(E, budget)
    return E


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
                      converge_tol: float | None = None, patience: int = 3,
                      contrast_cap: float | None = None):
    """Optimize exposure through the differentiable recording twin (ours).

    converge_tol : if set, stop early once the relative change in loss
        between successive `log_every`-spaced checks stays below
        `converge_tol` for `patience` consecutive checks. Default None
        preserves the original always-run-n_iters behavior exactly.
        `history`'s final (it, loss) entry reports the stopping iteration.
    contrast_cap : if set, additionally enforce max(E)/mean(E) <= contrast_cap
        (see contrast_project). Default None preserves the original
        dose-only projection exactly.
    """
    device = target.device
    # softplus-parameterized exposure, initialized near uniform dose
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    history = []
    prev_loss = None
    stable_count = 0
    broke_early = False

    for it in range(n_iters):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = contrast_project(E, dose_budget, contrast_cap)
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
                    broke_early = True
                    break
            prev_loss = cur

    # Guarantee history[-1][0] is the TRUE last iteration run whenever the
    # loop completed its full budget (not just the last log_every-aligned
    # checkpoint) -- without this, `iterations_run = history[-1][0]` looks
    # like an early stop even when converge_tol never triggered, since the
    # last checkpoint before n_iters is almost never exactly n_iters-1.
    # When the loop DID break early, the break already happened right after
    # appending the entry that triggered it, so history[-1] is already the
    # true stopping point and needs no adjustment.
    if not broke_early and n_iters > 0 and history and history[-1][0] != n_iters - 1:
        history.append((n_iters - 1, float(loss.detach())))

    with torch.no_grad():
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget, contrast_cap)
        recon = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E.detach(), recon.detach(), history


def media_blind_sgd(target: torch.Tensor, recorder: NPDDRecorder, bpm: SlabBPM,
                    n_iters: int = 400, lr: float = 5e-2, dose_budget: float = 1.0,
                    seed: int = 0, contrast_cap: float | None = None,
                    log_every: int = 50):
    """Optimize assuming a LINEAR ideal medium (dn = c * E), then evaluate the
    resulting exposure on the real twin. This is what current practice does.

    Now returns (E, recon, history) -- previously a 2-tuple with no loss
    curve. Phase 2 (M2 in the method registry) requires M2 to log its
    convergence curve "so plateau can be demonstrated," matching M4's
    convergence-parity requirement for a fair comparison. This is a
    BREAKING return-arity change; every call site in the repo was updated
    in the same commit (grep for `media_blind_sgd(` before editing further)."""
    device = target.device
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    c_lin = recorder.p.dn_max  # ideal linear map with same index budget
    history = []

    for it in range(n_iters):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = contrast_project(E, dose_budget, contrast_cap)
        dn_ideal = c_lin * (E - E.mean())          # linear, zero-mean modulation
        recon = bpm(dn_ideal, shrinkage=0.0)       # blind: no shrinkage either
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        loss.backward()
        opt.step()
        if it % log_every == 0:
            history.append((it, float(loss.detach())))

    with torch.no_grad():
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget, contrast_cap)
        recon_real = bpm(recorder(E), shrinkage=recorder.p.shrinkage)
    return E.detach(), recon_real.detach(), history


def linear_precomp(target: torch.Tensor, recorder: NPDDRecorder, bpm: SlabBPM,
                   dose_budget: float = 1.0, contrast_cap: float | None = None,
                   I_mean: float = 1.0):
    """Linear pre-compensation (Phase-2 registry: M3) -- closed-form, ZERO
    optimization. Did not exist in the codebase before this pass; the
    critical missing baseline per the master prompt: if media_in_the_loop
    (M4) cannot beat this formula, the "optimization" is not earning its
    keep.

    Method: FFT the target, boost each spatial-frequency component by
    1/H(K) where H is the linearized recording transfer function
    (NPDDRecorder.small_signal_mtf, Eq. 5/9), IFFT, clip to E>=0, then
    apply the EXACT same contrast_project used by M2/M4/M5a so all methods
    satisfy the same dose/contrast constraints identically (rather than an
    ad-hoc per-frequency boost cap, which would only approximately bound
    the realized exposure's peak/mean -- contrast_project bounds it
    exactly). `boost` itself is clamped only for numerical stability
    (H can be arbitrarily close to 0 at high K), not as the actual
    constraint mechanism.

    Unit-tested (tests/test_method_registry.py):
      - H(K) == 1 for all K (sigma=0, D0=0) => boost==1 everywhere => E is
        an unmodified (up-to-dose-rescaling) copy of target: no reshaping
        occurs when there is nothing to pre-compensate for.
      - output satisfies E >= 0 exactly and mean(E) == dose_budget exactly,
        and max(E)/mean(E) <= contrast_cap (+ 1e-6 slack) when set.
    """
    device = target.device
    dtype = torch.float64
    n_x, dx = recorder.n_x, recorder.dx
    freq = torch.fft.fftfreq(n_x, d=dx).to(device=device, dtype=dtype)  # cycles/um
    K = (2.0 * math.pi * freq).abs()  # rad/um; H depends on K^2, sign irrelevant
    H = recorder.small_signal_mtf(K, I_mean=I_mean)
    boost = torch.clamp(1.0 / (H + 1e-9), max=1e6)  # numerical safety only

    T_hat = torch.fft.fft(target.to(dtype).to(torch.complex128))
    E_hat = T_hat * boost.to(torch.complex128)
    E = torch.fft.ifft(E_hat).real
    E = torch.clamp(E, min=0.0)
    E = contrast_project(E, dose_budget, contrast_cap)

    with torch.no_grad():
        dn = recorder(E)
        recon = bpm(dn, shrinkage=recorder.p.shrinkage)
    return E, recon


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
                 seed: int = 0, contrast_cap: float | None = None):
    """CONSTRAINED oracle (Phase-2 registry: M5a). Upper bound: linear medium
    optimized AND evaluated as linear, under the SAME E>=0/dose/contrast
    constraints as media_in_the_loop (M4) -- see docs/definitions.md (b) for
    why this, not an unconstrained variant, is what "oracle" has always
    meant in this codebase. For the free (no exposure-domain constraints)
    counterpart, M5b, see oracle_unconstrained below."""
    device = target.device
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    c_lin = recorder.p.dn_max

    for _ in range(n_iters):
        opt.zero_grad()
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget, contrast_cap)
        recon = bpm(c_lin * (E - E.mean()), shrinkage=0.0)
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        loss.backward()
        opt.step()

    with torch.no_grad():
        E = contrast_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget, contrast_cap)
        recon = bpm(c_lin * (E - E.mean()), shrinkage=0.0)
    return E.detach(), recon.detach()


def oracle_unconstrained(target: torch.Tensor, recorder: NPDDRecorder, bpm: SlabBPM,
                         n_iters: int = 400, lr: float = 5e-2, seed: int = 0):
    """UNCONSTRAINED oracle (Phase-2 registry: M5b) -- did not exist in the
    codebase before this pass (docs/definitions.md (b)). Optimizes a real-
    valued index modulation dn(x) DIRECTLY against the ideal linear medium's
    readout, with no nonnegativity, no dose budget, and no contrast cap --
    the only constraint is whatever the medium's own dn_max saturation
    would allow, applied here as a soft tanh bound so the search space is
    still bounded (an entirely unbounded dn would let the optimizer cheat
    via arbitrarily large index contrast, which is not a physically
    meaningful upper bound). Comparing M5a (constrained) vs M5b decomposes
    the oracle gap in Sec. 5 into "cost of exposure-domain constraints"
    (M4 vs M5a headroom) vs "cost of medium physics" (M5a vs M5b headroom)."""
    device = target.device
    torch.manual_seed(seed)
    dn_max = recorder.p.dn_max
    dn_free = (1e-3 * torch.randn(recorder.n_x, device=device, dtype=torch.float64)
              ).requires_grad_(True)
    opt = torch.optim.Adam([dn_free], lr=lr)
    t_norm = target / (target.sum() + 1e-12)

    for _ in range(n_iters):
        opt.zero_grad()
        dn = dn_max * torch.tanh(dn_free / dn_max)  # soft-bounded, no hard E/dose constraint
        recon = bpm(dn, shrinkage=0.0)
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        loss.backward()
        opt.step()

    with torch.no_grad():
        dn = dn_max * torch.tanh(dn_free / dn_max)
        recon = bpm(dn, shrinkage=0.0)
    return dn.detach(), recon.detach()


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
