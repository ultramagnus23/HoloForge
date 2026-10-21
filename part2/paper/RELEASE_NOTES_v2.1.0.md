# Release notes: v2.1.0-applied-optics

**Prepared, not published.** Nothing here has been tagged or pushed. Create the
tag and the GitHub release yourself once `applied-optics` is merged into `main`,
so that Zenodo archives the exact code behind the submitted paper. The Data
availability statement cites the concept DOI `10.5281/zenodo.22202762`, which
always resolves to the newest release, so the manuscript needs no edit when this
one appears.

Suggested release title: **HoloForge v2.1.0: Applied Optics submission**

---

## Summary

Part 2 was declined by JOSA A at the editorial stage on 15 October 2026
(manuscript 620231), without peer review, on scope grounds and that journal's
limit on papers relying on computer-simulated results. No scientific objection
was raised. This release is the version resubmitted to Applied Optics.

The recording model is unchanged from v2.0.0. Every difference is validation,
uncertainty quantification or presentation, and no previously reported number
has moved.

## New analysis

- **Parameter sensitivity ranking** (`analysis/sensitivity.py`). Two
  independent routes, which agree on the order. One at a time over plus or
  minus 50 percent (tier S3), the K-averaged gain spans 0.64 dB for
  `dn_max`, 0.19 dB for `kappa`, 0.19 dB for `D0` and 0.06 dB for `sigma`,
  from a nominal 1.81 dB. Across the 30 joint draws (tier S6), a first-order
  attribution with bootstrap intervals gives +0.071 dB per standard deviation
  of error for `dn_max`, against +0.024 for `kappa`, -0.019 for `D0` and
  +0.010 for `sigma`; three of the four intervals exclude zero and `sigma` is
  the one that does not. Linear terms account for an R-squared of 0.88. The
  index budget therefore outranks the next parameter by 3.4 times.

- **Intermediate-slant RCWA sweep** (`experiments/rcwa_slant_sweep.py`). The
  committed grid tested slant 0 and 20 degrees and nothing between, so it could
  not say where the readout stops being trustworthy. The new sweep covers
  0, 2.5, 5, 7.5, 10, 15 and 20 degrees across both incidence conventions, the
  two slant-sweep spatial frequencies, three index-modulation levels and both
  polarizations: 336 cases, about two minutes on CPU.

  It identifies the mechanism behind the slant failure. At Bragg incidence the
  first transmitted order falls from 0.460 at 0 degrees to 0.023 at 2.5 degrees
  and 0.002 at 20 degrees, while the undiffracted order rises from 0.44 to
  0.91. Tilting the fringes takes the grating out of the symmetric two-wave
  Bragg geometry, so at a fixed readout angle it stops diffracting and no
  exposure design has anything left to act on. It also shows the closed-form
  reference losing validity at the smallest nonzero angle tested, which is why
  the apparent gain enhancement between 2.5 and 5 degrees is now reported as
  model-dependent rather than as an effect.

- **The design loop's propagator validated against RCWA directly**
  (`experiments/bpm_vs_rcwa.py`). Every earlier check in this repository
  compared the closed-form Kogelnik reference with RCWA and left the
  split-step beam propagation that actually sits inside the optimizer
  unchecked. Over 84 cases the two engines agree on the first-order fraction
  to a median of 0.001 and at worst 0.027 at normal incidence, the geometry
  the design loop reads out in, with the slanted cases no worse than the
  unslanted ones.

  This reversed a conclusion. The gain enhancement at 2.5 to 5 degrees of
  slant was reported earlier in this revision as a model-dependent artifact.
  It is real: a tilt of arcsin(K*lambda/(4*pi*n0)), the Bragg angle itself,
  brings a normally illuminated grating onto Bragg, predicting 2.42 and 4.84
  degrees at the two tested spatial frequencies with nothing fitted. The
  measured gain peaks at the two nearest tested angles, and so does the
  first-order diffracted fraction.

These scripts read result files already in `results/` or use the existing RCWA
machinery; none re-runs the design campaign. New outputs live in
`results/applied_optics/`.

## Manuscript

- Restructured for Applied Optics. The introduction opens on the practitioner's
  question; the small-signal transfer function is derived in condensed form in
  the main text; the compute-matched intervals are inlined; limitations become
  a section of their own.
- New **Section 6, "Design guidelines and calibration protocol"**: a five-step
  recipe covering the minimum measurement set, the order to measure in, which
  parameters repay careful calibration, and the conditions for not using the
  method at all.
- The main text no longer depends on the supplement. Supplement 1 keeps full
  derivations and extra sweeps only.
- Abstract trimmed to about 100 words, leading with the practical problem.
- Nine pages in the two-column journal layout, under the Applied Optics
  10-page limit.

## Figures

- **Fig. 3** (new): operating regime map. The gain over the full
  spatial-frequency by contrast-budget grid, the slant sweep, and the RCWA
  panel that explains the slant collapse.
- **Fig. 4** (new): parameter sensitivity. Tornado over plus or minus 50
  percent, and the joint-draw attribution with bootstrap intervals.
- Twin validation becomes **Fig. 5**; the one-parameter miscalibration curves
  move to **Fig. S6**.

## Repository

- `scripts/make_submission_package.py` now gates on published pages, the
  two-column journal layout, rather than the universal template, which runs
  longer for the same body.
- `data/literature/DIGITIZE_TODO.md` (new): verified specifications, with
  figure numbers and axes, for the multi-spatial-frequency datasets needed for
  a cross-K held-out validation. **That validation is not in this release**:
  the data is not in the repository.
- JOSA A references removed from the README, the submission guide, the
  changelog and the build scripts.
- `paper/SUGGESTED_REVIEWERS.md` (new), with the contact of each suggestion
  verified against an institutional page.

## Known gaps, stated plainly

- Still simulation only. No hologram has been recorded.
- The recording model is still compared against one commercial medium at one
  spatial frequency, in two views of the same condition, one of which is a
  model output, with points read visually rather than with a calibrated
  digitizer. The cross-spatial-frequency held-out validation is specified but
  not run.
- The recording model itself is still compared only against the closed-form
  reference and published growth curves. The propagator is now validated
  against RCWA, but that says nothing about the photochemistry.
