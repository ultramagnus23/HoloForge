# Changelog

## Applied Optics submission (2026-10-21)

JOSA A declined manuscript 620231 on 15 October 2026 at the editorial stage,
without peer review, on scope grounds and that journal's limit on papers
relying on computer-simulated results. No scientific objections were raised.
Retargeted to Applied Optics. The recording model is unchanged; everything
below is validation, uncertainty and presentation.

- `analysis/sensitivity.py` (new): ranks the four recording parameters by how
  much the design gain depends on each, two independent ways that agree.
  One-at-a-time over +/-50% (tier S3) and a first-order attribution across the
  30 joint draws (tier S6) with bootstrap intervals and a reported linear R^2.
  Delta n_max dominates, outranking kappa by 3.4x; sigma is the only parameter
  whose joint coefficient does not exclude zero. Writes
  `results/applied_optics/sensitivity.json`.
- `experiments/rcwa_slant_sweep.py` (new): fills in the intermediate slants
  the committed RCWA grid skips, 336 cases over both incidence conventions,
  the two S8 spatial frequencies and three index-modulation levels. Shows the
  slant collapse is Bragg detuning (first transmitted order 0.460 at 0 deg,
  0.023 at 2.5 deg, undiffracted order rising to 0.85) and that the
  closed-form reference loses validity at the smallest nonzero tested angle.
- Figures renumbered: new Fig. 3 (operating regime map: the K-by-budget grid,
  the slant sweep, and the RCWA mechanism panel) and Fig. 4 (sensitivity
  tornado plus joint attribution); twin validation becomes Fig. 5; the
  one-parameter miscalibration curves move to Fig. S6.
- Manuscript restructured for Applied Optics: the introduction opens on the
  practitioner's question, the small-signal transfer function is derived in
  condensed form in the main text, the compute-matched intervals are inlined,
  and a new Section 6 gives a step-by-step calibration protocol with the
  minimum measurement set and the parameters worth spending effort on.
  Limitations are a section of their own. The main text no longer depends on
  the supplement. Abstract trimmed to about 100 words.
- `scripts/make_submission_package.py` now gates on published pages (the
  two-column journal layout, limit 10) rather than the universal template,
  which runs longer for the same body. Currently 9 published pages.
- `data/literature/DIGITIZE_TODO.md` (new): verified specifications, with
  figure numbers and axes, for the multi-spatial-frequency datasets the
  cross-K held-out validation needs. That validation is not yet run; the data
  does not exist in the repository.
- `experiments/bpm_vs_rcwa.py` (new): compares the split-step beam propagation
  used INSIDE the design loop against RCWA directly, which no previous check
  did; every earlier comparison bounded only the closed-form Kogelnik
  reference. 84 cases over both spatial frequencies, three index-modulation
  levels, the full slant range and both incidence conventions. The engines
  agree on the first-order fraction to a median of 0.001 and at worst 0.027 at
  normal incidence, the geometry the design loop reads out in, with slanted
  cases no worse than unslanted; 0.037 is the worst over the whole grid.
  `rcwa_crosscheck.rcwa_order_powers` is an additive companion to
  `rcwa_de_general`, which is untouched so committed results stay reproducible.
- That result changed a conclusion. The earlier draft of this revision
  reported the gain enhancement at 2.5 to 5 degrees of slant as a
  model-dependent artifact, on the grounds that the closed-form reference
  fails there. With the propagator itself validated, the enhancement is real
  and has a closed-form explanation: a tilt of arcsin(K*lambda/(4*pi*n0)), the
  Bragg angle itself, brings a normally illuminated grating onto Bragg. That
  predicts 2.42 and 4.84 degrees at the two tested K with nothing fitted, and
  the measured gain peaks at the two tested angles nearest those, 2.5 and 5.0,
  where the first-order fraction peaks as well. Section 4.6, the limitations
  and the conclusion were rewritten, and Fig. 3(c) now shows both engines at
  normal incidence instead of RCWA alone at Bragg incidence.
- Bibliography: DOI coverage raised from 6 of 28 entries to 25, each matched
  against the Crossref registry record on title, first author, year,
  container, volume and pages. The three without one carry a comment saying
  what was checked and why none was added. Verifying them turned up two more
  errors in `sheridan2012reply`, an entry already corrected once before: it
  had issue 2 and pages 1--3, against the publisher's own "Vol. 29, Issue 3,
  pp. 460-462 (2012)".

## JOSA A submission (2026-10-03)
- Manuscript retargeted to JOSA A and fitted to 10 pages in the Optica
  universal template (7 in the two-column journal layout): new title,
  ~100-word abstract, Conclusion, Optica back matter including the required
  generative-AI disclosure, merged ablation table, four redrawn figures.
  Files renamed `oe_main.tex` -> `manuscript.tex`, `oe_supplement.tex` ->
  `supplement.tex`; new `cover_letter.tex` reads `numbers.tex`.
- Twin validation corrected: the k_bleach-held fit to the measured Bayfol
  series is degenerate (a flat line), which the old text described as
  reproducing the curve's shape. Fig. 4 now shows the k_bleach-free fits and
  the held-out cross-series predictions; the held fits move to Fig. S5.
- Supplement rebuilt as Supplement 1 (S-numbered sections, figures, tables)
  without legacy diagnostics or the previous-draft errata. A new
  finite-difference gradient check (`experiments/gradient_check.py`)
  replaces the pre-freeze checkpointed-gradient number.
- `figures/make_all.py` rewritten to produce exactly the paper's figures,
  drawn at the template text width (`Fig1`..`Fig4`, `FigS1`..`FigS5`).
- `scripts/check_consistency.py` no longer misses numbers written with a
  LaTeX thin space (`2\,dB`) and checks all three documents;
  `scripts/make_submission_package.py` (replaces `make_prism_package.py`)
  builds the upload-ready package and refuses to build on any error,
  undefined reference, `[PENDING]` macro or a manuscript over 10 pages.
- Removed legacy code and data not used by the paper: the 2D study, the 3D
  showcase, prelim/confirm/seed-bump runs, the neural-surrogate gradient
  ablation, GPU mesh/wavelength reruns, pre-freeze NZ runs, the pre-revision
  `results_archive/`, superseded figures, notebooks and audit docs. All of it
  remains in git history; no reported number changed.

## Unreleased
- **Analysis pinned to 3 seeds (`ANALYSIS_SEEDS`).** An M1 seed bump
  (3 -> 8 seeds) was started and stopped at ~16% (146/900 jobs, 8/45
  configurations). Its results are kept on disk but excluded from the
  analysis: a partial bump leaves M1 unbalanced, so per-point CIs are not
  comparable across the grid and no single seed count is true of the
  study. The filter is explicit and prints what it drops, because a
  silently narrowed input set is indistinguishable from missing data.
  This also corrected diffraction-efficiency numbers that the partial bump
  had contaminated (DENPairs 157 -> 135 = 45 configs x 3 seeds;
  DEMeanGainTwoX 2.31 -> 1.45). Set ANALYSIS_SEEDS = None after completing
  the bump for all 45 configurations.


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
