# Part 2 -- plan to a submittable paper by 2026-09-25

Written 2026-09-19. Working backwards from the deadline. Status is updated in
the checklist at the bottom; timings marked (est) are estimates, not measurements.

## Why the earlier loop happened, and how this plan breaks it

Every stored M1/M2/S4-S7 result (and most of S1-S3) predates the round-1
physics fixes (commit f23ec9f). `config_hash` resume logic cannot see a physics
change, so stale results looked "done". Attribution on one M1 cell (K=1.96, 2x
budget, n_x=1024, seed 0):

| code | BSGD dB | MIL dB | paired gain |
|---|---|---|---|
| pre-fix (reproduces stored M1) | 6.079 | 7.161 | 1.083 |
| + B2 only (NPDD conservation) | 6.079 | 7.162 | 1.083 |
| + B3 only (contrast cap enforced) | 6.160 | 7.178 | 1.018 |
| + I2 only (real slant shear, 20 deg) | 6.133 | 6.179 | 0.047 |

Slant sweep, current code, same cell: 0 deg 1.02 dB, 5 deg 2.20, 10 deg 0.11,
20 deg 0.06 (early-stopping ruled out). The stored "20 deg" headline was in
effect an unslanted result.

Breaking the loop: (1) freeze the geometry ONCE (below), (2) put the geometry in
every job's config so stale results can never collide (verified: 0 of 1458
main-grid jobs match an old hash), (3) archive stale results out of the
aggregation path, (4) one campaign, one aggregation, one rewrite.

## Frozen decisions (no further code changes to physics after this)

* Geometry: unslanted transmission grating, `slant_deg = 0` (recorded per job).
  Slant dependence is reported as its own experiment (S8), not hidden.
* Readout slices: `n_z = 128` (was 32), pending NZ0 confirmation.
* Simulation reading (B4 scoping): an infinite periodic grating -- the FFT
  pipeline is exact under that reading, matching how the recording model and
  Kogelnik theory already treat the grating. Stated explicitly in Sec. 3.
* Everything stays in the paper; every number regenerates from the frozen code.

## Compute plan

Hardware: one RTX 3050 laptop GPU, 6 parallel workers (GPU saturates ~98%).
Sequential cost estimate for the full campaign ~150 GPU-hours; measured
parallel speed-up ~3x (est) -> ~50 wall-hours. Queue, in priority order, with a
HARD CUTOFF Tue 2026-09-22 22:00:

1. NZ0 (n_z study at frozen geometry) -- decides n_z. ~1.5 h.
2. RSGD re-tune (its baseline hyper-parameters were tuned under the old physics).
3. M1 full grid (15 K x 3 budgets x 9 methods) -- main figure. ~51 h seq.
4. M2 compute-matched arm. ~22 h seq.
5. S3 designs -> S6, S7 (twin miscalibration, joint, depth absorption).
6. S4, S5 (target ensemble, readout noise).
7. S1 (physics ablation).
8. S8 (slant dependence, 84 jobs).
9. S2 (parameter sensitivity, reduced grid) -- first to be cut if behind.
10. 2D check (subset) -- compare current code vs stored 2D at one K.

Anything unfinished at the cutoff is reported at its reduced scale in the
supplement with the reduction stated; nothing is kept from the archive.

## Manuscript plan (runs in parallel with the campaign)

* Now (no numbers needed): geometry scoping in Model/Readout and the parameter
  table; B4 periodic-grating statement; B5 Appendix (closed-form K_c does not hold
  as derived); I6 disclosure/refit of the twin fit; related-work TODOs.
* After aggregation (Sep 22 night): `aggregate.py` -> `make_numbers_tex.py` ->
  `figures/make_all.py`; rewrite abstract, Results, Discussion against the
  regenerated numbers. Claims are conditioned on the data, with decision rules
  fixed in advance: CI excluding zero -> "positive gain"; CI crossing zero ->
  "no reliable gain in that region". `[PENDING]` macros must reach zero.
* Sep 24: full build, `check_consistency.py`, `check_refs.py`, page count,
  supplement, cover letter, data/code release notes.
* Sep 25: preprint + submission package handed to the author to post.

## What "done" means

Every number in the PDF traces to a post-freeze result file; zero `[PENDING]`;
tests pass; the archived pre-revision results are not referenced. This is a
simulation-only paper: the honest ceiling is "submittable and defensible", not
"experimentally validated".

## Checklist

- [x] Attribution of the M1 collapse (I2 slant shear)
- [x] Geometry frozen in code + config; stale results archived
- [x] NZ0 launched
- [ ] NZ0 confirms n_z = 128
- [ ] RSGD re-tuned
- [ ] Campaign queue launched (M1 -> M2 -> ...)
- [ ] Aggregation + numbers.tex + figures regenerated
- [ ] Manuscript rewrite, zero [PENDING]
- [ ] Final build + consistency checks
- [ ] Package handed off
