# Section 6 validation standard -- decision record

Gate B deliverable. Records which of the "Remaining Work Specification
(v2)" Phase B branches applies, and why, so the choice is traceable rather
than implicit in the prose.

## The three branches (from the work spec)

1. NRMSE ≲ 0.3: write Section 6 as straightforward validation.
2. NRMSE 0.3-0.5: write to the mechanism-validity standard (paired gain
   survives absolute-fidelity mismatch because it's driven by the
   saturating index response, established in S1).
3. NRMSE > 0.5, harness verified clean: bounded negative validation
   result; the model-change question becomes live but is out of scope for
   this submission cycle (see the work spec's "never-do" list -- no
   recording-model change after Sept 10).

## What Gate A actually found

Corrected fit (Gate A.1: `{kappa, dn_max}` instead of `{kappa, D0}`,
`dn_max` no longer fixed at a mis-scoped Table-3 citation; Gate A.4:
best-of-10-random-starts, not a single start):

| Source | K (rad/µm) | In tested range (1.96-15.71)? | NRMSE | Multi-start spread |
|---|---|---|---|---|
| Bruder 2017, Bayfol HX, experimental | 8.98 | yes | **0.286** | [0.286, 0.346], std 0.023 |
| Bruder 2017, Bayfol HX, their-own-simulation | 8.98 | yes | **0.174** | [0.174, 0.436], std 0.105 |
| Hsieh 2022, PQ/PMMA | 24.94 | **no** (1.6x above the top of the grid) | 0.100 | [0.100, 0.693], std 0.224 |

**Decision: Branch 1 (NRMSE ≲ 0.3) for the two in-regime sources.** Both
Bayfol series clear the 0.3 threshold with reasonably tight multi-start
spread (the experimental series especially: std 0.023 across 10 starts
means this isn't a lucky single init). Section 6 is written as a
straightforward, if imperfect, validation -- not the mechanism-validity
fallback, and not a bounded-negative-result framing.

## What does NOT support this conclusion, stated explicitly

- The PQ/PMMA source (Hsieh 2022) is excluded from this decision per Gate
  A.5 (out-of-tested-range) and the parameter-provenance finding that its
  curve never saturates within the plotted data, so `dn_max` isn't
  identifiable from it regardless of NRMSE. It stays in Figure F2 (shown,
  not hidden) but is not counted as a second in-regime confirmation.
- The two Bayfol series' fitted `dn_max` disagree with each other by
  roughly 2.7x (0.050 vs 0.136) despite nominally describing the same
  material at two similar exposure intensities. This is named directly in
  Section 6, not smoothed over -- a real residual inconsistency, even
  though both NRMSEs individually clear the "good" threshold.
- Two of the fit's calibration parameters (`D0`, `sigma` for both media;
  Table 3-derived context for Bayfol's original `dn_max` before this
  correction) were not independently re-scoped to the exact figure/table
  condition this pass -- see `docs/parameter_provenance.md`'s per-
  parameter table for what was and wasn't re-audited.
- Only one paper family is confirmed in-regime (Bayfol HX, two series
  from the same figure). The two paywalled Optica NPDD papers (Kelly &
  Sheridan 2011, Gleeson et al. 2008) would be a genuinely independent
  second in-regime source and remain unavailable as of this pass (source
  acquisition, Gate A.6, in progress in parallel).

## Why this doesn't require a recording-model change

Per the work spec's "never-do" list and the priority framing: a model
change was only on the table if fits stayed poor (>0.5) after the harness
was verified clean. The harness was verified clean (Gate A.2 self-
consistency test, NRMSE 0.03 on synthetic data) and the corrected fits
are not poor. There is no live question to revisit here -- the original
poor fits were a parameterization/citation-scoping bug, not evidence
against the twin's model structure.
