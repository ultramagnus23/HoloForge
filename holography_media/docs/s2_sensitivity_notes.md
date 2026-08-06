# S2 (parameter sensitivity on cliff location) -- write-up caveats

Two things a reviewer will raise about S2 regardless of which perturbation
grid ends up being run. Write both into the manuscript's S2 discussion;
don't try to expand S2's scope to preempt them.

## 1. One-at-a-time perturbation, no interaction terms

`build_S2_jobs` (experiments/manifest.py) perturbs D0, sigma, and kappa
**independently** -- each job varies exactly one parameter while holding
the other two at their default value. This is a standard one-at-a-time
(OAT) sensitivity design. It does NOT capture:

- interaction effects between parameters (e.g. whether a +50% D0 error
  combined with a +50% sigma error is worse or better than the sum of
  their individual effects on cliff location)
- the joint uncertainty region if all three parameters are simultaneously
  uncertain (which is the realistic case -- D0, sigma, and kappa are all
  fit from the same limited calibration data)

State this as an explicit limitation of S2's design, not an oversight.
Fixing it properly would mean a factorial or Sobol-type design across all
three parameters simultaneously, which is a different (much larger) tier,
not a grid tweak to this one.

## 2. Perturbation magnitudes need an external justification

The perturbation grid (`S2_PERTURBATIONS_PCT` / `_FULL`) is currently
picked by round numbers (+/-10/25/50%), not derived from anything. That's
defensible as "a reasonable sensitivity sweep" but not as "the range these
parameters could plausibly be wrong by in reality" -- a reviewer can
reasonably ask why +/-50% and not +/-30% or +/-80%.

Once V1's literature curves are digitized (`data/literature/`,
`experiments/fit_literature_curves.py`), the disagreement between
independently-fit NPDD parameters across different published sources
(e.g. if two papers' fitted D0 differ by ~40%) becomes a natural,
citable justification for the perturbation range actually used: pick the
grid to bracket the observed cross-source spread, and say so with a
citation, rather than asserting round numbers. This is a follow-up once
V1 data exists, not a blocker on running S2 now.
