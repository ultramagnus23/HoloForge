# Changelog

## Unreleased
- **New experiment tier S3 (twin-miscalibration robustness).**
  `experiments/run_s3_mismatch.py` + `build_S3_*` in
  `experiments/manifest.py`. Optimizes the exposure ONCE against the twin
  at nominal parameters, then evaluates that same frozen exposure on
  perturbed twins with no re-optimization. This is the experiment that
  can support a robustness claim; S2 cannot, because it re-optimizes both
  arms on the perturbed medium, keeping the perturbation common to both
  arms -- the exact condition under which the paper's own
  paired-comparison argument says a systematic twin error cancels. The
  manuscript previously stated a robustness claim citing S2; that claim
  is now stated against S3 and the S2 subsection says explicitly what it
  does and does not establish.
  Result: gain survives +/-50% error in D0, sigma, kappa and dn_max; the
  only failure found is over-estimating dn_max by ~2.7x (+170%), where
  gain goes negative. dn_max is swept wider than the others because our
  own literature fits disagree on it by exactly that factor.
- **New baseline SAT (saturation-only surrogate).**
  `holomedia.npdd.SaturationOnlyTwin` + `holomedia.optimize.sat_sgd`,
  registered in `experiments/methods.py` and run on M1's full grid.
  Answers S1's unanswered follow-up ("if saturation dominates, why the
  full PDE?") with data: it recovers 28/42/48% of media-in-the-loop's
  mean gain at 2x/4x/8x, so the transport terms do carry real
  information despite each being individually non-critical in S1.
  The surrogate is the EXACT closed-form zero-transport limit of the full
  twin (regression-tested to 3.4e-5 relative), not an arbitrary sigmoid.
  Its dose sensitivity is calibrated to the real twin by a one-parameter
  offline fit before use: the uncalibrated limit is fully saturated at
  working dose (gradient 3.9e-11 vs the twin's 2.6e-5, six orders down)
  and an optimizer started there never moves, which would have made the
  baseline lose on optimizer conditioning rather than modeling power.
- **FIX (figures, affects a published figure):** `_render_F8` (S2
  sensitivity band) pooled every K point into one bucket per
  (param, pct, method) before calling `analysis.aggregate.paired_gain`.
  That function matches arms by SEED via a dict, so with 3 seeds and N K
  points it kept only the LAST K's BSGD value per seed and paired every
  MIL row against it -- comparing MIL at one spatial frequency against
  BSGD at another. F8's plotted curve consequently did not match
  `s2_sensitivity_summary`, which pairs correctly inside each config
  group. Both F8 and the new F10 now pair within K and pool afterwards.
- **FIX (figures):** the S2/S3 error bands were 95% CIs over pooled
  (seed, K) gains, so they were dominated by between-K spread -- a real
  effect being averaged over, not uncertainty about the mean -- and read
  as enormous seed uncertainty (0.2-3.4 dB around a 1.8 dB mean on S3).
  Now computed per seed after K-averaging, which is what the axis labels
  already claimed. Means are unchanged on a balanced grid; only the
  intervals move.


## v0.4
- First GPU results for this project (Colab T4; prior passes had no GPU
  access). Two solver-characterization sweeps, single seed each — NOT the
  full 5-seed F2/F3 method-comparison rerun (`f2_f3_recovery.py` at paper
  scale is still unrun; see README TODO).
- Added a real convergence-threshold early-stop option (`converge_tol`) to
  `media_in_the_loop` (`holomedia/optimize.py`) — previously the optimizer
  had no such concept, so there was nothing to sweep. Non-breaking: default
  `None` preserves the old fixed-iteration behavior exactly.
- Fixed a GPU-readiness gap: `experiments/f2_f3_recovery.py` and
  `run_confirm.py` never call `.to(device)` on the `NPDDRecorder`/`SlabBPM`
  modules, so passing CUDA target tensors would previously not have moved
  the actual computation to GPU. The new `experiments/gpu_*.py` scripts move
  both modules explicitly.
- `experiments/gpu_npdd_mesh_convergence_sweep.py`: mesh-density sweep
  (n_x=512/1024/2048, fixed 51.2µm window) found PSNR mesh-independent
  within 0.04dB and wall-clock flat (~450s) across the range — the GPU run
  is overhead-bound, not compute-bound, at this problem size. Convergence-
  threshold sweep (tol=1e-3/1e-4/1e-5) found tol=1e-4 (630 iters, 351s)
  already matches tol=1e-5's PSNR (1350 iters, 749s) — diminishing returns
  well before the 800-iter fixed budget used elsewhere.
  Results: `results/gpu_reruns/npdd_mesh_sweep/results.json`.
- `experiments/gpu_bpm_wavelength_sweep.py`: one recording at 405nm design,
  read out at 400-450nm (brackets the paper's stated 405/450nm band). PSNR/DE
  peak on-design and fall off non-monotonically moving away from it
  (420nm worse than 435nm) -- consistent with Kogelnik's oscillatory
  sin^2(nu) detuning response, reported as measured rather than smoothed.
  Results: `results/gpu_reruns/bpm_wavelength_sweep/results.json`.

## v0.3
- Table 1 filled with real literature-cited parameter ranges (PVA/AA, PQ/PMMA,
  Bayfol-class); added `configs/media/pq_pmma_405nm.yaml` and
  `bayfol_hx_405nm.yaml`. Some cells flagged as order-of-magnitude anchors or
  "not independently confirmed" rather than over-claimed precision — see
  Table 1's caveat paragraph.
- `data/literature/`: documented (not fabricated) the exact papers/figures to
  digitize and wired an auto-loading overlay into `f1_validate_twin.py`.
- Gradient-pathway ablation: `holomedia/npdd.py::forward_checkpointed`
  (checkpointed discrete adjoint) + `holomedia/surrogate.py` (neural
  surrogate) + `experiments/ablation_gradients.py`. Measured, not assumed:
  checkpointed gradients reach 0.977 cosine similarity to unrolled autodiff on
  a realistic probe (not bit-exact -- see code comments on why); surrogate
  gradients are noisier (0.84) but still useful for a short optimization run.
- RCWA cross-check via `torcwa`: `experiments/rcwa_crosscheck.py`. Scalar
  Kogelnik vs. full-vector RCWA agree to within 0.038 absolute DE across
  K = 2-12 rad/µm at 405 nm / 30 µm thickness (Bragg-matched, Snell-refracted
  incidence angle).
- 3D (x,y,z) showcase: `holomedia/npdd3d.py`, `holomedia/diffraction3d.py`,
  `experiments/showcase_3d.py`. Media-in-the-loop beats media-blind by
  +0.63 dB on a small (48x48) ring target -- same qualitative direction as
  the 2D(x,z) results, smaller magnitude at this reduced scale.
- `experiments/make_figures.py` now reads confirmation/paper-scale results
  (`results_confirm.json`) when present, falls back to CPU-scale
  `results_prelim.json`, and adds Figs E (gradient ablation) and F (RCWA).
- Confirmation-scale rerun (`experiments/run_confirm.py`, n_x=512, 500 iters,
  3 seeds): a genuine increase over v0.1/v0.2 CPU-scale numbers, but **not**
  the full 1024/800-iter/5-seed paper scale the README originally asked for --
  this environment has no GPU, and the full sweep benchmarks at ~26h CPU time.
  See README.md for the exact GPU command to finish the job.
- New smoke tests: checkpointed-forward-matches-forward, 3D twin runs +
  gradients flow (`tests/test_smoke.py`).

## v0.2
- **PHYSICS FIX (breaking):** NPDD non-local term corrected from
  F * (G conv u) to G conv (F * u). The kernel must smear the full
  production term (chains initiated at x' deposit polymer at x). The v0.1
  form left the exposure pattern unblurred, silently disabling sigma.
  Detected because sigma sweeps were implausibly flat at high K; post-fix
  the solver reproduces the analytic Ghat(K) rolloff. All v0.1 results
  retracted and rerun.
- Shrinkage model v2: slanted-fringe Bragg detuning via per-slice lateral
  FFT shift (physical; unslanted transmission fringes correctly near-immune).
- Added F2 qualitative panel (natural-spectrum target): +7.6 dB media-aware gain.
- Added sweeps: thickness, D0, high-K sigma crossover probe.

## v0.1
- Initial twin (NPDD spectral IMEX), Kogelnik + slab BPM, media-in-the-loop
  optimizer + 3 baselines, cliff/sigma/dn_max sweeps, smoke tests, paper draft.
