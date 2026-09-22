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

Hardware: one RTX 3050 laptop GPU shared by parallel workers. The GPU
saturates (~96-98% utilization) at 4-6 workers, so the parallel speed-up is
~2-3x (est, from BSGD/MIL timings under contention), not linear. Sequential
cost of the full campaign is ~170 GPU-hours (est) -> ~60-80 wall-hours, which
does NOT fit comfortably before the cutoff; hence the priority queue and the
cut rules below. HARD CUTOFF Tue 2026-09-22 22:00.

Queue order (each worker runs it over its own balanced shard):

1. NZ0 (n_z study at frozen geometry, seed 0 decides) -- confirms n_z = 128.
2. RSGD re-tune (its regularization strengths were tuned under the old physics).
3. M1A: main grid, coarse pass (8 K x 3 budgets x 9 methods, 504 jobs).
4. S8: slant dependence (84 jobs) -- the finding that changed the paper's scope.
5. M2: compute-matched arm.
6. S4, S5 (target ensemble, readout noise), S1 (physics ablation).
7. M1B: main grid fill-in (the other 7 K points, same configs as the full M1).
(S2 and the 2D study were dropped from scope, see below.)
Separate single-process jobs: S3 designs -> S6 -> S7 (twin miscalibration, joint,
depth absorption); R1/R3 figure data.

Cut rules (decided now, so the cutoff is not a judgement call under pressure):
anything unfinished at the cutoff is reported at its reduced scale with the
reduction stated in the text (e.g. "8 of 15 K points", "one K point", "n=2
seeds"); nothing is carried over from the archive; a claim whose supporting
run did not finish is removed rather than softened.

## Manuscript plan (runs in parallel with the campaign)

* Now (no numbers needed): geometry scoping in Model/Readout and the parameter
  table; B4 periodic-grating statement; B5 Appendix (closed-form K_c does not hold
  as derived); I6 disclosure/refit of the twin fit; related-work TODOs.
* After aggregation (Sep 24 08:00-20:00): `aggregate.py` -> `make_numbers_tex.py`
  -> `figures/make_all.py`; rewrite abstract, Results, Discussion against the
  regenerated numbers, including the S8 slant subsection. Claims are
  conditioned on the data: CI excluding zero -> "positive gain"; CI crossing
  zero -> "no reliable gain in that region". `[PENDING]` macros must reach zero.
* Sep 24 evening: full build, `check_consistency.py`, `check_refs.py`, page
  count, supplement, cover letter, data/code release notes.
* Sep 25: preprint + submission package handed to the author to post.

## Venue and length (decided 2026-09-19)

Target: **Applied Optics** (JOSA A is the fallback; same template, same policy).
Both are Optica hybrid journals with **no mandatory author charges** (voluntary
page charge $125/page; optional open access $2,300; print-colour figures are
charged, online colour is free) [opg.optica.org/content/author/portal/item/review-pub-charge].
**Overlength fee: $300/page beyond 10 published pages**, so the constraint is
<= 10 pages in the two-column journal format (`oe_main_lengthcheck.tex`, 9pt).
Optics Express is fully open access with a mandatory APC and is dropped.
Verify these figures on the live Optica author page at submission time.

## Scope cut (decided 2026-09-19, to fit 10 pages and the compute budget)

Removed from the paper, not re-run: the bounded 2D study, the NPDD parameter-
sensitivity study (S2), and the older diagnostics are kept only as clearly labelled
pre-revision material in the supplement. Kept and re-run: NZ0, M1 (A then B), S8,
M2, S4/S5 (supplement), S1, and S3 -> S6 -> S7. Two-column length after the cuts:
9 pages + 2 lines, before adding the slant subsection.

## What "done" means

Every number in the PDF traces to a post-freeze result file; zero `[PENDING]`;
tests pass; the archived pre-revision results are not referenced. This is a
simulation-only paper: the honest ceiling is "submittable and defensible", not
"experimentally validated".

## Checklist

- [x] Attribution of the M1 collapse (I2 slant shear)
- [x] Geometry frozen in code + config; stale results archived
- [x] NZ0 launched (6 workers, 2026-09-19 ~20:25)
- [x] Code fully frozen: 1D and 3D recorders (B2), contrast projection (B3), slant + n_z in config
- [x] Geometry / B5 / I6 / B4-scoping manuscript edits applied
- [x] S8 + NZ0 aggregation, numbers.tex macros and figure F11 wired
- [ ] NZ0 confirms n_z = 128
- [ ] RSGD re-tuned (running)
- [ ] Campaign queue launched (auto-starts when RSGD tuning finishes)
- [ ] Aggregation + numbers.tex + figures regenerated
- [ ] Manuscript rewrite, zero [PENDING]
- [ ] Final build + consistency checks
- [ ] Package handed off
