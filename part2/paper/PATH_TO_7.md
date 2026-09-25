# Path from ~5 to 7 out of 10 (written 2026-09-26)

Scale: 1 = rough idea, 10 = exemplary paper. Scores are my judgment, not
measurements. "7" means: sound, well-scoped, reproducible, limitations disclosed,
contribution modest but clear -- publishable in Applied Optics / JOSA A after
minor-to-moderate revision. Not 8-10: that needs a physical recording and a second
medium, which are out of reach in the near term.

## Where it stands (after reading the finished results)

Science ~5, manuscript ~3 (stale numbers, unwritten slant section, abstract claims
contradicted by the data, M2 not run, main grid at 9 of 15 K points).

## Must-do for a 7 (in this order)

1. **Write up what exists, consistently (no GPU, ~2 days).** 3 -> ~5. Run
   `aggregate.py` -> `make_numbers_tex.py` -> `figures/make_all.py`; rewrite abstract
   and Results against regenerated macros. Fix claims the data contradicts:
   "saturation, not transport kinetics" (K-resolved S1 shows no single mechanism
   removal collapses the gain; only K=5.24 changes materially), "never negative"
   (K=4.19, 2x is -0.03 dB), "15 tested K", SAT fraction (now ~18-36% of MIL's gain,
   very K-dependent), and delete the M2 subsection unless a reduced M2 finishes.
   Report S3/S6/S7 as K-averaged over K = 1.31/3.93/5.24 at 2x, not a single point.
2. **Replace the vague mechanism claim with a decisive ablation (~1 code change,
   ~3 GPU-h est).** The current "no_saturation" removes only the tanh index map;
   monomer depletion (finite u) still saturates N. Add an opt-in ablation that holds
   u fixed (infinite reservoir) and one that removes both nonlinearities. This can
   yield a genuine finding about WHERE the gain comes from (+0.4).
3. **Structural uncertainty (~5 GPU-h est).** Seed spread is ~0.001-0.002 dB, so the
   n=3 CIs measure optimizer initialization, not real uncertainty. Run S2R (one-K
   parameter sensitivity, 78 jobs, identical hashes to full S2) and report K-resolved
   results, not K-averages (+0.3).
4. **Fix the twin fit (I6) and add out-of-sample validation (largest lever, +0.7-1.0).**
   Refit with `k_bleach` free (or restrict fits to the pre-depletion dose range), then
   validate OUT OF SAMPLE: fit on some K, predict held-out K, and use the two paywalled
   NPDD multi-K growth-curve families (needs the author's library access to digitize;
   the fitting pipeline already exists). Turns "mechanism-level, one medium" into a
   stronger claim.
5. **Outside expert read before submission (~3-7 days, their side, +0.4).** 1-2
   photopolymer-modelling or CGH researchers; `premortem_referee_reports.md` is the
   list of expected attacks.

## Nice-to-have (drop first if time runs short)

- Finish M1B (~11 GPU-h est): full 15-point grid, +0.2. A coarse 9-point grid
  reported honestly is acceptable.
- Reduced M2 (~4 GPU-h est: 3 K x one budget x 3 seeds): answers "MIL is just more
  compute", +0.2.
- Minimal 2D check (1 target class, 2 budgets, 2 seeds; runtime uncertain): removes
  the "1D only" objection, +0.2.
- Release: tagged repo + Zenodo DOI + `REPRODUCE.md`, +0.2.

## Sequence

- Day 0: commit all results (S3/S6/S7/M1B/NZ0 are currently untracked), run the
  aggregator on the real data to shake out bugs in the new S8/NZ0 code.
- Day 0-3: GPU queue in this order: mechanism ablation, S2R, reduced M2, M1B.
  Meanwhile: writing (#1), twin refit (#4, CPU).
- Day ~3-4: preprint v1 (honest scope, ~6). Preprints are versioned, so v2 replaces it.
- Day ~4-10: outside read, second-medium validation, v2 (~7), then journal submission.

## Preconditions and risks

- The laptop has silently slept twice (39 h and ~12.7 h lost). Sleep must be disabled
  at the OS level and the machine kept plugged in; otherwise plan on 2-3x the wall time
  or move the GPU queue elsewhere. A watchdog on the same machine cannot catch a sleep.
- Item #4 depends on the author obtaining the two paywalled papers.
- All GPU-hour figures are estimates from observed 4-worker job times (~25 min per MIL
  job, ~4-5 min per BSGD/oracle job).
