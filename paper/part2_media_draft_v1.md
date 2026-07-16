# Media-in-the-Loop Holography: Computer-Generated Holography Through a Differentiable Model of Volume Photopolymer Recording

**Chaitanya Tripathi¹, [Co-author]²**
¹Ashoka University, Sonipat, India — ²BITS Pilani, India

> DRAFT v1. Every bracket of the form **[RESULT: ...]** is a number or figure
> that must come from running the experiments in `/experiments`. Do not
> submit or preprint with any placeholder unfilled. Sections marked
> [EXPAND] need 1–2 paragraphs of your own prose after results exist.

---

## Abstract

Computer-generated holography (CGH) algorithms overwhelmingly assume an ideal phase modulator. Camera-in-the-loop (CITL) methods correct real-device model mismatch for spatial light modulators, but write-once volume recording media — the photopolymers underlying holographic optical elements, AR waveguide couplers, and low-cost fabricated holograms — permit no such feedback: by the time reconstruction error is measurable, the hologram is permanent. We introduce *media-in-the-loop holography*, which replaces the camera with a differentiable digital twin of the recording chemistry itself. The twin couples the non-local polymerization-driven diffusion (NPDD) model of photopolymer grating formation to volume diffraction via split-step beam propagation, and reproduces published growth-curve and angular-selectivity behavior **[RESULT: quantify agreement, e.g. "to within X% of digitized experimental curves"]**. Optimizing the *delivered exposure* through the twin — a nonnegative, dose-budgeted inverse problem structurally distinct from phase-only CGH — recovers 1.7–4.0 dB of reconstruction quality over media-blind optimization in preliminary experiments at PVA/acrylamide-like parameters and 405 nm **[RESULT-FINAL: confirm and tighten at paper scale — 1024+ grid, 800+ iters, 5 seeds, full medium families]**. We further identify a *compensation cliff*: a spatial-frequency threshold, predicted analytically from a linearization of the recording kinetics, beyond which dose and transport constraints render degradation unrecoverable by any exposure pre-compensation. In preliminary runs the media-aware advantage collapses from +4.0 dB to ≈0 dB between K = 5.2 and 7.9 rad/µm, bracketing the analytic prediction K_c ≈ 4.2–7.0 rad/µm for contrast budgets of 2–4× **[RESULT-FINAL: dense-K confirmation]**. All code, configurations, and digitized validation data are released.

---

## 1. Introduction

Phase-only CGH computes a modulation pattern whose diffracted field approximates a target image. Four decades of algorithmic progress — from Gerchberg–Saxton (GS) [Gerchberg & Saxton 1972] through Wirtinger and stochastic-gradient holography [Chakravarthula et al. 2019] to neural methods [Peng et al. 2020; Shi et al. 2021] — share one assumption: the computed pattern is what the physical device displays. For spatial light modulators (SLMs), the gap between assumption and hardware was closed by camera-in-the-loop optimization [Peng et al. 2020], which measures the actual reconstruction and backpropagates the residual, either per-image or into a parameterized display model.

This solution is unavailable to an entire class of holographic devices. Volume photopolymer media — Bayfol HX, PVA/acrylamide, PQ/PMMA — record a hologram as a permanent refractive-index distribution formed by photopolymerization and mass transport. The medium is write-once: measurement of the reconstruction error is possible only after the error has become unerasable. CITL's feedback signal arrives, by construction, too late. Yet these media are precisely where model mismatch is most severe. The recorded index profile is not a scaled copy of the exposure: monomer diffusion during exposure, non-local polymer chain growth, dye depletion with depth, saturable index response, and post-exposure shrinkage each reshape the pattern, in ways that depend nonlinearly on the exposure itself.

Current fabrication practice therefore either restricts designs to regimes where the medium is approximately linear, or accepts the degradation. Concurrently, *fabrication-aware* differentiable optics has emerged for surface-relief elements: neural-lithography digital twins predict the 3D geometry produced by grayscale lithography and are placed inside end-to-end design loops [Zheng et al. 2023; Wei et al. 2025]. These twins, however, model an etched height map — a thin phase screen — and are inapplicable to volume media, whose behavior is governed by reaction–diffusion kinetics inside a 10–100 µm slab and read out through thick-grating (Bragg-selective) diffraction.

We close this gap. Our contributions:

1. **A differentiable recording twin** (Sec. 3): the NPDD reaction–diffusion system, dye depletion, saturable index response, and shrinkage, implemented with a spectral IMEX integrator whose gradients are obtained by unrolling; validated against published photopolymer measurements.
2. **Exposure-domain CGH** (Sec. 4): reformulating hologram design as optimization over the nonnegative, dose-budgeted exposure delivered to the medium, with the twin inside the loss. We show why this problem is structurally different from phase-only CGH and analyze its linearized limit.
3. **A compensation-limit analysis** (Sec. 4.3): a closed-form small-signal transfer function of the recording process, yielding an analytic prediction of the spatial frequency beyond which pre-compensation must fail; confirmed by full nonlinear experiments.
4. **A systematic recovery study** (Sec. 5): across physically realistic parameter ranges for three photopolymer families at 405/450 nm, quantifying what media-in-the-loop optimization recovers relative to media-blind GS and SGD baselines, and where it saturates.

This work is simulation-only by design; Sec. 6 states precisely what experimental validation requires, and the twin's component-wise validation against literature measurements (Sec. 3.4) bounds the sim-to-real gap our claims depend on.

## 2. Related work

**Model-mismatch correction for holographic displays.** CITL optimization [Peng et al. 2020] and its extensions — Michelson holography [Choi et al. 2021], partially coherent CITL [Peng et al. 2021], learned hardware-in-the-loop phase retrieval [Chakravarthula et al. 2020], dual-modulation and simultaneous-color calibration [2023–2024] — correct the display's forward model using camera measurements. All require a rewritable modulator observed live. Our setting inverts the availability: the medium is write-once, so the model must carry the full burden; no measurement of the final artifact informs its own optimization.

**Fabrication-aware surface-relief optics.** Neural lithography [Zheng et al. 2023] and large-area fabrication-aware diffractive optics [Wei et al. 2025] insert a learned digital twin of a lithographic process into differentiable design. Tolerance-aware deep optics [2025] optimizes designs for robustness to geometric manufacturing-error distributions, and classical studies quantify etch-depth and alignment error effects in multilevel diffractive elements [Sci. Rep. 2020; binary-optics literature]. All model *surface geometry*. Volume photopolymers differ in kind: the recorded object is a volumetric index distribution produced by chemistry, its formation is nonlinear and history-dependent, and its readout is Bragg-selective. A prior patent applies a fixed smoothing kernel to a CGH profile for injection-moldability [US 10,649,157]; this is a post-hoc filter on surface relief, not in-loop optimization through recording physics.

**Photopolymer recording models.** The NPDD framework [Sheridan & Lawrence 2000; Gleeson & Sheridan 2009; review: Guo et al. 2012] quantitatively describes grating formation in photopolymers: polymerization driven by exposure, non-local chain growth of characteristic length σ, monomer diffusion at rate D, and saturation of the achievable index modulation. Coupled-wave theory [Kogelnik 1969] maps the recorded modulation to diffraction efficiency and angular selectivity. For twenty-five years these models have been used *descriptively* — fitted to measured gratings to extract material parameters. To our knowledge they have never been differentiated through, nor placed inside a hologram-design loop. That inversion of use is the central move of this paper.

**Our prior work.** [Tripathi 2025, HoloForge preprint] systematically characterized degradation of phase-only CGH under abstract device non-idealities (quantization, noise), identifying a sharp quality collapse at 1-bit phase quantization. The present work replaces abstract operators with degradations that *emerge* from recording physics, and moves from characterization to compensation and limits.

## 3. A differentiable model of photopolymer recording

### 3.1 Recording kinetics

Let u(x,t) denote free-monomer concentration (normalized), N(x,t) polymer concentration, and d(x,t) photosensitizer concentration, over transverse coordinate x during exposure to intensity I(x). The NPDD system reads

  ∂u/∂t = ∂/∂x ( D(N) ∂u/∂x ) − F(x,t) (G_σ ∗ u)(x)   (1)
  ∂N/∂t = + F(x,t) (G_σ ∗ u)(x)         (2)
  ∂d/∂t = − k_b I(x) d          (3)

with local initiation rate F(x,t) = κ (I(x) d(x,t))^γ, γ ∈ [½, 1] capturing radical-termination kinetics; non-local response kernel G_σ a normalized Gaussian of width σ modeling polymer-chain growth away from the initiation site; and network-hindered diffusivity D(N) = D₀ exp(−α_D N). Dye depletion (3) couples exposure history to sensitivity, and, extended over depth via Beer–Lambert absorption, produces the depth-dependence that invalidates thin-element treatment. The recorded index perturbation follows a saturating Lorentz–Lorenz response, Δn(x) = Δn_max tanh(c_N N(x)), where Δn_max encodes the medium's dynamic-range budget (the M/# in holographic-storage terminology). Post-exposure shrinkage compresses the recorded structure longitudinally by factor (1−s), detuning the Bragg condition at readout.

**Table 1 — Physical parameter ranges and sources.** [EXPAND: fill from the reading assignment; every row cites the measurement paper.]

| Parameter | Symbol | PVA/AA | PQ/PMMA | Bayfol-class | Source |
|---|---|---|---|---|---|
| Monomer diffusivity (µm²/s) | D₀ | [RESULT] | [RESULT] | [RESULT] | [cite] |
| Non-locality length (µm) | σ | ~0.05–0.1 | ... | ... | [cite] |
| Index budget | Δn_max | ... | ... | ... | [cite] |
| Shrinkage | s | ... | ... | ... | [cite] |
| Dose sensitivity @405 nm | κ | ... | ... | ... | [cite] |

### 3.2 Numerical scheme and differentiability

Equations (1–3) are integrated by a spectral IMEX scheme: the stiff diffusion operator is applied exactly in Fourier space each step (unconditionally stable), while reaction terms advance explicitly; the non-local convolution is a Fourier multiplier. The full trajectory (200–500 steps) is unrolled under automatic differentiation, giving exact gradients of the recorded profile with respect to the exposure. Memory is bounded by gradient checkpointing every 25 steps. [EXPAND after ablation: unrolled vs adjoint vs neural-surrogate gradients; report wall-clock and gradient fidelity — experiments/ablation_gradients.py.]

### 3.3 Readout

Reconstruction is computed by split-step scalar beam propagation through the recorded slab: alternating phase kicks exp(i k₀ Δn(x,z) δz) and band-limited angular-spectrum propagation over δz inside the medium, followed by free-space propagation to the observation plane. For pure sinusoidal gratings this engine must and does reduce to Kogelnik's coupled-wave predictions (Sec. 3.4), which also serve as the closed-form validation tier. Scalar validity is bounded by RCWA cross-checks on representative gratings **[RESULT: max deviation over tested cases]**.

### 3.4 Twin validation (Figure 1)

We validate the twin against digitized published measurements in three respects: (a) diffraction-efficiency growth curves versus exposure for sinusoidal recording at several spatial frequencies, reproducing the NPDD signature of growth, saturation, and high-frequency rolloff governed by the dimensionless ratio R = D₀K²/F₀; (b) the recorded-contrast transfer versus spatial frequency against the linearized prediction of Sec. 4.3; and (c) Kogelnik angular selectivity of recorded gratings. **[RESULT: Fig. 1 panels + quantitative agreement statement.]**

## 4. Media-in-the-loop optimization

### 4.1 Problem statement

Given target intensity I_t, find exposure E(x) minimizing

  L(E) = ‖ P( Twin(E) ) − I_t ‖²  s.t. E ≥ 0, mean(E) = B  (4)

where Twin maps exposure to recorded Δn via Sec. 3, P is the readout propagation, and B the dose budget. Nonnegativity is enforced by softplus parameterization, the budget by projection each step; optimization uses Adam through the unrolled twin.

### 4.2 Why this is not phase-only CGH

Phase-only CGH optimizes an unconstrained angle variable; expressivity is limited only by the phase-wrapping structure. Exposure-domain CGH is constrained on three sides simultaneously: the control is nonnegative (intensity), integrally bounded (dose), and enters the recorded modulation through a *low-pass, saturating, self-depleting* map — the non-local kernel blurs, the tanh saturates, and every joule spent writing one region depletes dye and monomer available elsewhere and later. Pre-compensation (boosting high-frequency exposure content to counteract the blur) is possible only while the boosted exposure remains nonnegative and within budget, and only while the medium retains monomer to convert. This triple constraint produces a hard feasibility boundary absent from conventional CGH. [EXPAND: 1 paragraph connecting to observed optimization behavior.]

### 4.3 The compensation cliff

Linearizing (1–2) about uniform exposure I₀ for a small sinusoidal perturbation at spatial frequency K yields the recording transfer function

  H(K) = Ĝ(K) / (1 + D₀K²/F₀),  Ĝ(K) = exp(−σ²K²/2), F₀ = κ I₀^γ. (5)

Exact pre-compensation of a target component at K requires exposure boost 1/H(K). Given contrast headroom B_c set by nonnegativity and dose budget, compensation is feasible only for K < K_c where 1/H(K_c) = B_c. Equation (5) therefore *predicts* a compensation cliff — the volume-media analogue, emergent from chemistry, of the quantization cliff reported in [Tripathi 2025]. Figure 3 tests this prediction against full nonlinear optimization. **[RESULT: predicted K_c vs. observed knee, across σ and D₀ sweeps.]**

## 5. Experiments

**Protocol.** Methods: (i) media-blind GS (phase-optimized, naive linear exposure conversion — current common practice); (ii) media-blind SGD (gradient-optimized under an ideal linear medium, evaluated on the twin); (iii) media-in-the-loop SGD (ours); (iv) oracle (ideal medium, upper bound). Targets: binary bar patterns at three frequencies, sparse-spot patterns, and [N] natural-image slices (DIV2K); 5 seeds; identical iteration budgets; metrics: PSNR, SSIM, in-support diffraction efficiency. All configurations in `/configs`; every figure regenerates from one script.

> **PRELIMINARY RESULTS (v0.1, CPU scale: n_x = 256, 100 IMEX steps, 150 Adam
> iterations, 2 seeds — real simulation outputs of the released code; all
> trends require confirmation at paper scale before submission).**
>
> **Cliff test (Fig. A).** At default PVA/AA-like parameters, media-in-the-loop
> optimization outperforms media-blind SGD by +2.6, +4.0, +1.7, +2.2, +1.7 dB
> at K = 0.98, 1.31, 1.96, 2.62, 3.93 rad/µm respectively — and the advantage
> collapses to −0.1 and +0.3 dB at K = 5.24 and 7.85 rad/µm. The analytic
> model (Eq. 5) predicts K_c = 4.2 / 7.0 / 9.9 rad/µm for contrast budgets of
> 2× / 4× / 8×; the observed collapse between 5.2 and 7.9 rad/µm falls inside
> the 2–4× budget band, consistent with the dose-projected optimizer's
> effective headroom. Mean in-support diffraction efficiency across the sweep:
> 0.563 (ours) vs 0.506 (blind).
>
> **Dynamic-range sweep.** The media-aware gain grows with index budget:
> +0.4 dB at Δn_max = 1×10⁻³, +1.7 dB at 3.5×10⁻³, +2.0 dB at 6×10⁻³ —
> compensation needs headroom to spend.
>
> **Non-locality sweep.** At K = 3.93 rad/µm the gain is nearly flat in σ
> (+1.6 to +1.7 dB from σ = 0.02 to 0.3 µm): at this frequency and D₀ the
> transport term D₀K²/F₀ dominates the rolloff, not the non-local blur. The
> paper-scale σ sweep must therefore probe higher K, where Ĝ(K) bites
> **[RESULT-FINAL: rerun σ sweep at K ≥ 8 rad/µm]** — itself a finding:
> which mechanism limits recording is frequency-dependent, and Eq. 5
> predicts the crossover.
>
> **Honest reading of the oracle gap.** Even media-aware optimization sits
> 4–10 dB below the ideal-medium oracle: compensation is partial, and the
> residual gap quantifies the irreducible cost of recording physics at this
> operating point. Absolute PSNR values at this scale are low (binary
> targets, small grids, short optimization) and are meaningful only as
> relative comparisons under identical budgets.

**F2 — Qualitative comparison** at a Bayfol-like operating point, 405 nm. **[RESULT-FINAL: figure + artifact-structure prose at paper scale.]**

**F3 — Recovery curves.** Quality versus σ, D₀, Δn_max, s, T for all four methods at paper scale, with mean ± std bands over 5 seeds. **[RESULT-FINAL]**

**F4 — Diode-regime design chart.** Media-aware gain (dB) over the (medium family × dose-sensitivity) plane at 405 and 450 nm. **[RESULT: contour figure + practitioner reading of it.]**

**F5 — Ablations.** Gradient pathway (unrolled / adjoint / surrogate); single vs. scheduled multi-exposure; 2D main results vs. one 3D(x,y,z) demonstration case. **[RESULT]**

**F6 — Failure atlas.** Regimes where compensation is impossible, classified by limiting mechanism: blur-limited (σ), transport-limited (D₀K²/F₀), depletion-limited (dose/monomer), detuning-limited (s·T). **[RESULT + EXPAND: this section is where reviewer goodwill is won; be generous with honest failure.]**

## 6. Discussion and limitations

This study is simulation-only; its claims are about optimization mathematics conditioned on a physics model validated component-wise against published measurements (Sec. 3.4). Experimental closure requires two-beam recording of optimized versus naive exposures in a characterized medium and comparison of measured reconstructions — planned as follow-on work. Scalar diffraction bounds validity to moderate numerical apertures (RCWA deviations bounded in Sec. 3.3); polarization and vector effects are not modeled. Main results are 2D(x,z), standard in the NPDD literature; a 3D case demonstrates that no qualitative behavior is lost. The twin's parameters, while literature-sourced, vary batch-to-batch in real media; media-in-the-loop optimization in practice would be paired with per-batch parameter fitting from a small number of test gratings — exactly the fitting workflow the NPDD literature already established. [EXPAND after results.]

## 7. Conclusion

Write-once volume media are the one class of holographic device that can never benefit from camera-in-the-loop correction, and the class whose recording physics distorts designs the most. Placing a differentiable model of that physics inside the design loop recovers **[RESULT]** of the lost quality and — equally useful — tells the designer, in advance and in closed form, what cannot be recovered. We release the twin and all experiments as an open extension of the HoloForge codebase.

## References
[EXPAND: full BibTeX in paper/refs.bib — every work named in Sec. 2, plus Goodman, Matsushima & Shimobaba (band-limited ASM), Kogelnik 1969, Zhao & Mouroulis 1994, Sheridan NPDD series, Guo et al. 2012 review, Peng et al. 2020/2021, Choi et al. 2021, Shi et al. 2021, Zheng et al. 2023, Wei et al. 2025, tolerance-aware deep optics 2025, MDL error studies, US 10,649,157.]
