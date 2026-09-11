"""
Differentiable Non-local Polymerization-Driven Diffusion (NPDD) recording model.

Implements the two-species NPDD system (Sheridan et al.) as a differentiable
PyTorch module using a spectral IMEX (implicit diffusion / explicit reaction)
time stepper. The diffusion operator is applied exactly in Fourier space,
which is unconditionally stable and cheap to differentiate through.

State variables (1D transverse coordinate x, evolved in time):
    u(x,t)  -- free monomer concentration (normalized to u0 = 1)
    N(x,t)  -- immobile polymer concentration
    d(x,t)  -- photosensitive dye concentration (normalized, bleaches)

Governing equations (normalized):
    du/dt = d/dx( D(u) du/dx ) - F(x,t) * (G * u)(x)
    dN/dt = + F(x,t) * (G * u)(x)
    dd/dt = - k_bleach * I(x) * d(x)

with:
    F(x,t) = kappa * (I(x) * d(x,t))**gamma      local initiation rate
    G      = Gaussian non-local response kernel, width sigma (chain growth)
    D(u)   = D0 * exp(-alpha_D * (1-u))          diffusion slows as network forms

Refractive index modulation (Lorentz-Lorenz linearized, saturating):
    dn(x)  = dn_max * tanh( c_n * N(x) )

All parameters carry physically meaningful ranges; see configs/media/*.yaml
and Table 1 of the paper for literature sources.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as Fnn
import torch.utils.checkpoint as _ckpt


@dataclass
class MediumParams:
    """Physical parameters of the photopolymer medium (normalized units).

    Normalization: length unit = 1 micron, time unit = 1 s, u0 = 1.
    Literature ranges (see paper Table 1):
        D0        : 1e-3 .. 1e0  um^2/s   (monomer diffusion, PVA/AA ~ 1e-1)
        sigma     : 0.02 .. 0.3  um       (non-locality length; ~50-100 nm typical)
        kappa     : 0.1 .. 10    (dose sensitivity, folds in intensity scale)
        gamma     : 0.5 .. 1.0   (radical termination exponent; 0.5 = bimolecular)
        dn_max    : 1e-3 .. 6e-3 (index modulation budget / M/# proxy)
        k_bleach  : 0 .. 1       (dye bleaching rate; 0 disables depletion)
        alpha_D   : 0 .. 3       (diffusion slowdown from network formation)
        shrinkage : 0 .. 0.03    (fractional thickness change; detunes Bragg)
        thickness : 5 .. 100 um  (used by the diffraction stage)
        n0        : background index (1.5 typical)
    """
    D0: float = 0.1
    sigma: float = 0.08
    kappa: float = 2.0
    gamma: float = 1.0
    dn_max: float = 3.5e-3
    k_bleach: float = 0.2
    alpha_D: float = 1.0
    shrinkage: float = 0.005
    thickness: float = 30.0
    n0: float = 1.5
    # REMEDIATION (confirmed peer-review finding B1): a real "remove
    # saturation" ablation, not a dn_max rescale -- see NPDDRecorder's
    # _index_map docstring. A medium-level (not recorder-constructor-level)
    # field so experiments/manifest.py's existing
    # dict(DEFAULT_MEDIUM, **overrides) ablation-condition machinery can
    # set it exactly like every other physics toggle (sigma=0, D0=0, ...).
    linearize_index_map: bool = False

    def to_tensor_dict(self, device, dtype=torch.float64):
        return {k: torch.as_tensor(v, device=device, dtype=dtype)
                for k, v in self.__dict__.items()}


def depth_resolved_dn(recorder: "NPDDRecorder", exposure: torch.Tensor,
                      optical_density: float, n_z: int) -> torch.Tensor:
    """WP6 (Applied Optics revision, depth-resolved absorption): the
    uniform-through-depth recording assumption oe_main.tex's Discussion
    section already bounds analytically (Section on depth-resolved
    absorption) as "good only for OD <~ 0.1 over the recorded
    thickness" -- this makes that bound an EMPIRICAL one instead, by
    actually attenuating the recording exposure with depth (Beer-Lambert:
    delivered dose falls by 10^{-OD * z/thickness} at depth z) and
    running the SAME NPDD recording physics independently at each of
    n_z depth slices, then letting each slice's own (now dimmer, at
    greater depth) exposure saturate the local dn on its own.

    Returns dn of shape (n_z, n_x): one full 1D recorded index profile
    per depth slice, NOT a single profile extruded uniformly through
    depth the way every other tier in this codebase assumes. Feeds
    directly into holomedia.diffraction.SlabBPM.forward_depth_resolved.

    Implementation note: NPDDRecorder.forward already supports a leading
    batch dimension (its diffusion step means D_eff over dim=-1 only,
    keepdim=True -- see NPDDRecorder._diffuse's docstring), so all n_z
    depth slices are recorded in ONE batched forward call, not a Python
    loop over z -- this is real physics, not free, but it is one GPU
    call rather than n_z sequential ones.

    optical_density=0 must reproduce the existing uniform-depth
    assumption EXACTLY (every slice sees identical, unattenuated
    exposure) -- this is the empirical check that this function is a
    strict generalization of the existing model, not a different one.
    """
    if optical_density < 0:
        raise ValueError(f"optical_density must be >= 0, got {optical_density}")
    z_frac = (torch.arange(n_z, device=exposure.device, dtype=recorder.dtype) + 0.5) / n_z
    atten = 10.0 ** (-optical_density * z_frac)  # (n_z,), 1.0 at z_frac=0 by construction
    I_stack = exposure.unsqueeze(0) * atten.unsqueeze(1)  # (n_z, n_x)
    dn_stack = recorder(I_stack)  # batched forward, (n_z, n_x)
    return dn_stack


class NPDDRecorder(torch.nn.Module):
    """Differentiable simulator: exposure pattern -> recorded index profile.

    Parameters
    ----------
    n_x : grid points in x
    dx  : grid spacing (um)
    t_total : exposure duration (s)
    n_steps : time steps (IMEX; 200-500 adequate for the parameter ranges above)
    params : MediumParams
    """

    def __init__(self, n_x: int, dx: float, t_total: float = 10.0,
                 n_steps: int = 300, params: MediumParams | None = None,
                 dtype=torch.float64):
        super().__init__()
        self.n_x, self.dx = n_x, dx
        self.t_total, self.n_steps = t_total, n_steps
        self.dt = t_total / n_steps
        self.p = params or MediumParams()
        self.dtype = dtype

        # Fourier-space wavenumbers for spectral operators
        k = 2.0 * math.pi * torch.fft.fftfreq(n_x, d=dx)
        self.register_buffer("k2", (k ** 2).to(dtype))

        # Non-local Gaussian kernel in Fourier space: exp(-k^2 sigma^2 / 2)
        self.register_buffer(
            "G_hat", torch.exp(-0.5 * (k ** 2) * (self.p.sigma ** 2)).to(dtype)
        )

    # ---------------------------------------------------------------- helpers
    def _nonlocal(self, u: torch.Tensor) -> torch.Tensor:
        """Gaussian non-local response applied via FFT (periodic BCs)."""
        return torch.fft.ifft(torch.fft.fft(u) * self.G_hat).real

    def _index_map(self, N: torch.Tensor) -> torch.Tensor:
        """dn(N): the saturating map by default, or its exact small-signal
        LINEAR limit (tanh(x) ~= x for small x, so tanh(1.5N) -> 1.5N) when
        self.p.linearize_index_map=True.

        REMEDIATION (confirmed peer-review finding B1): the previous
        ablation approximated "removing saturation" by scaling dn_max
        instead. That does not work -- dn_max never enters the u/N/d
        PDEs, only this final conversion, so N (and tanh(1.5N)'s own
        saturation, since realistic N already sits near 1) is
        bit-for-bit IDENTICAL regardless of dn_max, verified directly
        (max|N(dn_max) - N(100*dn_max)| = 0.0 on a realistic exposure).
        This flag instead linearizes the conversion itself, holding
        every PDE dynamic (and every other medium parameter, dn_max
        included) exactly fixed -- it isolates the index-map
        nonlinearity specifically, which is what the ablation claims to
        test."""
        if self.p.linearize_index_map:
            return self.p.dn_max * 1.5 * N
        return self.p.dn_max * torch.tanh(1.5 * N)

    def _diffuse(self, u: torch.Tensor, D_eff: torch.Tensor) -> torch.Tensor:
        """Implicit (exact) diffusion step with spatially averaged D.

        Uses the spatial ARITHMETIC mean of D_eff (D_eff.mean() below --
        corrected label; this docstring previously said "harmonic-mean",
        which does not match what the code computes) as a single scalar
        D_bar, applied via an exact spectral (constant-coefficient)
        diffusion step -- a real approximation to the stated governing
        PDE's spatially-varying operator d/dx(D_eff(x) d/dx u), not an
        exact solution of it (confirmed peer-review finding I3).

        REMEDIATION: this docstring previously cited
        "experiments/ablation_variableD.py" as already having validated
        that this approximation is "second-order for the regimes
        studied" -- that file does not exist in this codebase (checked
        directly). The claim was unsupported by any actual evidence.
        Replaced with a real, freshly-run comparison instead of either
        removing the claim or leaving a dangling reference: at a
        realistic late-trajectory optimizer state (D_eff varying ~2.4x
        across the field, 0.0285-0.0697), one diffusion sub-step under
        this mean-D spectral approximation differs from a true
        variable-coefficient conservative finite-difference reference
        (d/dx(D(x) du/dx), 200 explicit sub-steps for numerical
        accuracy) by ~0.11% relative to u's own scale. This is a
        single-step comparison, not a bound on error accumulated over a
        full 300-step trajectory -- a genuine remaining gap, not claimed
        to be closed here.

        Mean is taken over the last (spatial) axis only, keepdim=True, so
        a batched (B, n_x) input gets one D_bar per batch row rather than
        one pooled across the whole batch -- for unbatched (n_x,) input
        this reduces to the original scalar mean exactly.
        """
        D_bar = D_eff.mean(dim=-1, keepdim=True)
        decay = torch.exp(-D_bar * self.k2 * self.dt)
        return torch.fft.ifft(torch.fft.fft(u) * decay).real

    # ---------------------------------------------------------------- forward
    def forward(self, exposure: torch.Tensor, return_history: bool = False):
        """Simulate recording.

        exposure : (n_x,) nonnegative intensity pattern I(x), normalized so
                   that mean(I) ~ 1 corresponds to nominal dose at t_total.
        returns  : dn (n_x,) recorded index modulation profile
                   (optionally full state history for diagnostics).
        """
        I = exposure.to(self.dtype)
        u = torch.ones_like(I)
        N = torch.zeros_like(I)
        d = torch.ones_like(I)
        hist = []

        for _ in range(self.n_steps):
            # explicit reaction half-step
            # Non-local chain growth: polymer initiated at x' deposits at x,
            # so the Gaussian kernel acts on the FULL production term F*u,
            # not on u alone (Sheridan NPDD; see paper Eq. 2).
            F_loc = self.p.kappa * torch.clamp(I * d, min=0.0) ** self.p.gamma
            poly_rate = self._nonlocal(F_loc * torch.clamp(u, min=0.0))
            # REMEDIATION (confirmed peer-review finding B2): dt*poly_rate is
            # an EXPLICIT-Euler estimate of how much monomer this step
            # consumes; at large dt*F_loc products (e.g. large literature-fit
            # kappa at large dose, dt~0.16, F_loc~30) it can exceed the
            # monomer actually available, driving u negative. The old code
            # updated u and N by the SAME dt*poly_rate (exactly conservative
            # by construction: d(u+N)/dt=0 pointwise) and only THEN clamped u
            # to >=0 -- discarding the negative excess from u without ever
            # removing the corresponding (physically impossible) amount from
            # N, so N silently absorbed monomer that was never there.
            # Reproduced directly on this codebase's real literature-fit
            # inputs (K=8.98, kappa=16.34, dose=80, the n_steps=500 the
            # pipeline actually used): mean(u+N) came out to 2.63 instead of
            # the periodic-BC conservation law's required 1.0, and even
            # 32,000 steps only reached ~1.019, not exact -- confirming this
            # is a genuine conservation bug, not merely under-resolution.
            # Fix: cap the CONSUMED amount at what actually exists (`u`)
            # before applying it to both fields, so u >= 0 and u+N is
            # conserved to floating-point precision at ANY step size --
            # this is the standard positivity-preserving fix for an explicit
            # reaction step, not a finer-timestep workaround.
            consumed = torch.minimum(self.dt * poly_rate, u)
            u = u - consumed
            N = N + consumed
            d = d * torch.exp(-self.p.k_bleach * I * self.dt)
            u = torch.clamp(u, min=0.0)  # safety net; consumed<=u makes this a no-op now

            # implicit diffusion step (network-slowed diffusivity)
            D_eff = self.p.D0 * torch.exp(-self.p.alpha_D * N)
            u = self._diffuse(u, D_eff)

            if return_history:
                hist.append((u.detach().clone(), N.detach().clone()))

        dn = self._index_map(N)
        return (dn, hist) if return_history else dn

    # -------------------------------------------------- checkpointed forward
    def _step(self, u, N, d, I):
        # Same B2 conservation fix as forward() -- cap consumed monomer at
        # what exists, applied identically to u and N, rather than clamping
        # u alone after an unconstrained explicit update.
        F_loc = self.p.kappa * torch.clamp(I * d, min=0.0) ** self.p.gamma
        poly_rate = self._nonlocal(F_loc * torch.clamp(u, min=0.0))
        consumed = torch.minimum(self.dt * poly_rate, u)
        u = u - consumed
        N = N + consumed
        d = d * torch.exp(-self.p.k_bleach * I * self.dt)
        u = torch.clamp(u, min=0.0)
        D_eff = self.p.D0 * torch.exp(-self.p.alpha_D * N)
        u = self._diffuse(u, D_eff)
        return u, N, d

    def _block(self, u, N, d, I, n_sub):
        for _ in range(n_sub):
            u, N, d = self._step(u, N, d, I)
        return u, N, d

    def forward_checkpointed(self, exposure: torch.Tensor, block: int = 25):
        """Discrete-adjoint-equivalent forward pass.

        Reverse-mode AD through an unrolled loop already computes the exact
        discrete adjoint; the practical distinction ablated here is memory:
        instead of retaining every one of n_steps intermediate activations,
        `torch.utils.checkpoint` retains state only every `block` steps and
        recomputes the sub-trajectory during the backward pass. Gradients are
        analytically identical to `forward()`'s, and this now measures as
        cosine similarity 1.000000 on real optimization probes -- not the
        ~0.96-0.98 previously reported here.

        CORRECTION (confirmed peer-review finding I10): the ~0.96-0.98
        figure previously stated in this docstring was never a real
        floating-point divergence between the checkpointed and unrolled
        gradients. It was a bug in experiments/ablation_gradients.py's
        cosine-similarity helper: its safety epsilon (1e-12), meant to
        guard against dividing by a zero-norm vector, was NOT negligible
        relative to the real denominator when comparing these specific
        gradients (norm ~1e-6, so norm(a)*norm(b) ~1e-12 -- the SAME
        order as the epsilon meant to be negligible next to it),
        artificially depressing the reported similarity by a couple of
        percent even for numerically identical vectors -- reproduced
        directly: two literally-identical gradient vectors at this norm
        scale gave a "cosine similarity" of 0.977127 under the old
        formula. With the epsilon fixed (1e-30, still a zero-division
        guard, now genuinely negligible), the real answer is that
        checkpointing changes nothing about the computed gradient at
        this block size, only memory/wall-clock (see
        experiments/ablation_gradients.py for the measured numbers).
        """
        I = exposure.to(self.dtype)
        u = torch.ones_like(I)
        N = torch.zeros_like(I)
        d = torch.ones_like(I)
        n_blocks, rem = divmod(self.n_steps, block)
        for _ in range(n_blocks):
            u, N, d = _ckpt.checkpoint(self._block, u, N, d, I, block,
                                        use_reentrant=False)
        if rem:
            u, N, d = self._block(u, N, d, I, rem)
        return self._index_map(N)

    # ------------------------------------------------------- analytic helpers
    def small_signal_mtf(self, K: torch.Tensor, I_mean: float = 1.0):
        """Linearized NPDD transfer function H(K) (paper Eq. 9).

        H(K) = G_hat(K) / (1 + D0 K^2 / F0)
        with F0 = kappa * I_mean^gamma.  Predicts the recordable-contrast
        rolloff and, combined with the dose/nonnegativity budget, the
        compensation cliff K_c where required boost 1/H exceeds budget B.
        """
        F0 = self.p.kappa * (I_mean ** self.p.gamma)
        G = torch.exp(-0.5 * K ** 2 * self.p.sigma ** 2)
        return G / (1.0 + self.p.D0 * K ** 2 / F0)

    def predicted_cliff(self, budget: float = 4.0, I_mean: float = 1.0) -> float:
        """Spatial frequency K_c beyond which compensation exceeds `budget`x boost."""
        K = torch.linspace(0.1, 2 * math.pi / (2 * self.dx), 4096, dtype=self.dtype)
        H = self.small_signal_mtf(K, I_mean)
        mask = (1.0 / H) > budget
        return float(K[mask][0]) if mask.any() else float("inf")


class SaturationOnlyTwin(NPDDRecorder):
    """Cheap pointwise saturation-only surrogate for the NPDD twin.

    WHY THIS EXISTS. S1's mechanism ablation shows that removing
    saturation changes MIL's gain by ~7.5x, while removing non-locality,
    diffusion or dye depletion each changes it by <=28%. The obvious
    follow-up -- "then why pay for the full PDE at all?" -- is answered
    with data by optimizing against THIS model and evaluating the
    resulting exposure on the real twin (holomedia.optimize.sat_sgd,
    method code SAT), not with prose.

    THE MODEL. Delete every transport term from the NPDD system:
    sigma -> 0 (G = identity), D0 -> 0 (no diffusion), k_bleach -> 0
    (d == 1). What survives is a purely local ODE pair

        du/dt = -F u,   dN/dt = +F u,   F = kappa * I(x)**gamma

    whose solution needs no time stepping at all:

        N(x)  = 1 - exp(-a * I(x)**gamma)
        dn(x) = dn_max * tanh(1.5 * N(x))

    i.e. ONE monotone saturating map from exposure to index in a handful
    of elementwise ops -- no n_steps IMEX unroll, no FFTs, and an
    O(1)-memory backward pass instead of an unrolled adjoint.

    THE SENSITIVITY `a`, AND WHY IT MUST BE CALIBRATED. Setting
    a = kappa * t_total makes this map the EXACT zero-transport limit of
    the full recorder (verified to 3.5e-5 relative in
    tests/test_method_registry.py::
    test_saturation_only_twin_matches_npdd_in_zero_transport_limit), and
    that is this class's default. But at the paper's default medium that
    exact limit is USELESS as a design model: a = kappa * t_total = 20,
    so at the nominal dose (mean E = 1) the pointwise reaction has
    already run to completion everywhere, N = 1 - e^-20, and the map is
    flat. Measured at the optimizer's uniform initialization, d(dn)/dE is
    3.9e-11 against the full twin's 2.6e-5 -- six orders down. An
    optimizer started there never leaves it (measured: 800 iterations
    move realized contrast from 1.00 to 1.04, and the loss not at all).

    That is a real property of the zero-transport limit and is worth
    stating once -- what keeps the real medium responsive at working dose
    is precisely the transport this model deletes. But a baseline that
    lost for that reason would be a straw man: it would be failing from
    optimization conditioning, not from lack of modeling power, and would
    answer a question nobody asked. So the SAT baseline calibrates `a`
    first (see fit_saturation_only), which is what anyone actually
    shipping a cheap saturating model would do, and is the same
    fit-once-offline / amortize-forever protocol the neural surrogate in
    holomedia/surrogate.py already uses. On the default medium the fit
    returns a ~ 5.3 at NRMSE ~ 0.06 against the full twin: a genuinely
    good pointwise fit, and a fair opponent.

    WHY NOT LITERALLY dn = dn_max * tanh(kappa * E). Because the form
    above is not a guess: it is what the full model BECOMES in the
    zero-transport limit, so SAT-vs-MIL is a clean ablation of the
    transport terms rather than a comparison against an arbitrarily
    chosen sigmoid. Same cost class, same single free parameter; this one
    is just the honest one.

    Everything else -- dn_max, gamma, shrinkage, thickness, n0 -- is the
    medium's own. SAT is a cheaper MODEL of the same medium, not a
    differently calibrated medium.
    """

    def __init__(self, n_x: int, dx: float, t_total: float = 10.0,
                 n_steps: int = 300, params: MediumParams | None = None,
                 dtype=torch.float64, a_eff: float | None = None):
        super().__init__(n_x, dx, t_total=t_total, n_steps=n_steps,
                         params=params, dtype=dtype)
        # None => the exact zero-transport limit. See the class docstring
        # for why the SAT baseline does not use that default.
        self.a_eff = (self.p.kappa * self.t_total) if a_eff is None else float(a_eff)

    def forward(self, exposure: torch.Tensor, return_history: bool = False):
        I = torch.clamp(exposure.to(self.dtype), min=0.0)
        N = 1.0 - torch.exp(-self.a_eff * (I ** self.p.gamma))
        dn = self.p.dn_max * torch.tanh(1.5 * N)
        return (dn, []) if return_history else dn

    def forward_checkpointed(self, exposure: torch.Tensor, block: int = 25):
        """No unrolled trajectory to checkpoint -- the forward pass is
        already O(1) in memory, so this is just forward()."""
        return self.forward(exposure)


def _sample_band_limited_exposures(n_samples: int, n_x: int, dtype, seed: int,
                                   dose_budget: float = 1.0):
    """Random nonnegative band-limited exposures at the given mean dose.

    Same generator family as holomedia.surrogate.train_surrogate's, for
    the same reason: the fit should see the statistics the optimizer
    actually explores. (A spatially UNIFORM exposure would be useless as
    fit data here -- a constant profile drives the full twin to complete
    conversion regardless of its level, so it carries no information
    about the dose response at all.)
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    raw = torch.randn(n_samples, n_x, generator=g, dtype=dtype)
    f = torch.fft.fft(raw, dim=-1)
    freqs = torch.fft.fftfreq(n_x).to(dtype)
    cutoff = 0.05 + 0.4 * torch.rand(n_samples, 1, generator=g, dtype=dtype)
    f = f * torch.exp(-(freqs / cutoff) ** 2).to(f.dtype)
    smooth = torch.fft.ifft(f, dim=-1).real
    E = Fnn.softplus(2.0 * smooth) + 1e-3
    return E * (dose_budget / E.mean(dim=-1, keepdim=True))


def fit_saturation_only(recorder: NPDDRecorder, n_samples: int = 48,
                        seed: int = 0, n_steps_fit: int = 600,
                        lr: float = 0.05, dose_budget: float = 1.0
                        ) -> tuple[float, float]:
    """Calibrate SaturationOnlyTwin's single sensitivity `a` to `recorder`.

    Offline one-parameter least squares of the pointwise map against the
    full twin's response on `n_samples` random band-limited exposures.
    Costs n_samples forward passes of the real twin, paid once and
    amortized over every subsequent optimization -- the same bargain
    holomedia/surrogate.py's neural surrogate makes, with one parameter
    instead of a CNN.

    Fitting in log-space keeps `a` positive with no constraint machinery.

    Returns (a_eff, nrmse), with NRMSE normalized by the full twin's own
    dn range over the fit set. Report that number: it is the residual a
    purely pointwise model cannot explain -- i.e. exactly the information
    the transport terms carry -- and it is the honest measure of the
    handicap SAT runs under before any optimization happens.
    """
    dtype = recorder.dtype
    device = recorder.k2.device
    Es = _sample_band_limited_exposures(n_samples, recorder.n_x, dtype, seed,
                                        dose_budget=dose_budget).to(device)
    with torch.no_grad():
        dns = torch.stack([recorder(Es[i]) for i in range(n_samples)])

    log_a = torch.zeros((), dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.Adam([log_a], lr=lr)
    dn_max, gamma = recorder.p.dn_max, recorder.p.gamma
    loss = None
    for _ in range(n_steps_fit):
        opt.zero_grad()
        a = torch.exp(log_a)
        pred = dn_max * torch.tanh(1.5 * (1.0 - torch.exp(-a * Es ** gamma)))
        loss = torch.mean((pred - dns) ** 2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        rng = (dns.max() - dns.min()).clamp(min=1e-12)
        nrmse = float(torch.sqrt(loss.detach()) / rng)
    return float(torch.exp(log_a.detach())), nrmse
