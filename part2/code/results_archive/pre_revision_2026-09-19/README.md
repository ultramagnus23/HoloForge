# Pre-revision results (archived 2026-09-19)

Everything here was produced by code from BEFORE the round-1 peer-review physics
fixes (commit f23ec9f: B2 NPDD conservation, B3 contrast projection, I2 slant
shear) and at the old implicit geometry (n_z=32, slant effectively 0 -> the
"20 deg" default had no geometric effect). Attribution on one M1 cell showed the
I2 fix alone collapses the paired gain from 1.08 dB to 0.05 dB at 20 deg slant.

These results are kept for provenance only. They must NOT be aggregated into
paper numbers. The revised campaign re-runs everything at the frozen geometry
(n_z=128, slant_deg=0, recorded in every job's config).
