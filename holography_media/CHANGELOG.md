# Changelog

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
