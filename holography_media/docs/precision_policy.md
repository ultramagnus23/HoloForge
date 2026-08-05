# Precision policy (spec Sec. 1.1)

## Decision
The production manifest pipeline (`experiments/run_manifest.py`, i.e. every
Phase 3 E1-E7 job) defaults to **float32** (complex64 fields). This does
**not** change `holomedia`'s library-wide defaults (still float64) or any
already-run legacy experiment script (`run_prelim.py`, `run_confirm.py`,
`gpu_*.py`, `ablation_gradients.py`, `f1_validate_twin.py`,
`fit_literature_curves.py`, `rcwa_crosscheck.py`, `showcase_3d.py`) --
those either pin `torch.float64` explicitly or rely on `NPDDRecorder`'s
own constructor default (unchanged), so their already-committed results
stay exactly reproducible if re-run.

## Evidence
Measured (not assumed) on a representative config: `media_in_the_loop`
(M4), `n_x=256`, `n_steps=100`, `n_iters=100`, PVA/AA-like defaults,
CPU (this dev environment has no GPU; the *numerical* comparison is
device-independent, only the speed motivation is GPU-specific):

| | value |
|---|---|
| float64 PSNR, seeds 0/1/2 | 4.8772 / 4.8807 / 4.8731 dB |
| seed-to-seed std (float64, n=3) | 0.0038 dB |
| float32 PSNR, seed 0 | 4.877222537994385 dB |
| float64 PSNR, seed 0 | 4.877223287171710 dB |
| \|float32 − float64\| | 7.5e-7 dB |

**float32 changes the answer by ~5000x less than seed noise already
does.** Per the spec's own stated criterion ("if float32 changes PSNR by
less than the seed-to-seed standard deviation, float32 is defensible"),
this is a clean pass, not a borderline call.

Reproduce: `python -c "..."` -- see the git history of this file's
introducing commit for the exact script; not checked in as a standalone
script since it's a one-time justification check, not a recurring test
(the *outcome* -- float32 is enabled in `run_manifest.py`'s `DTYPE`
constant -- is what downstream code and tests depend on, not this
specific comparison being re-run every time).

## What this doesn't cover yet
- **Not GPU-measured.** The speed/memory motivation (consumer NVIDIA
  cards run FP64 at 1/64 of FP32 rate; T4 at 1/32) is well-established
  hardware fact, not something this environment can measure directly. The
  accuracy justification above is real; the speed payoff will only be
  directly confirmed once Phase 3 actually runs on a T4.
- **RCWA stays float64 unconditionally** (`experiments/rcwa_crosscheck.py`
  sets its own local `torch.set_default_dtype(torch.float64)`,
  untouched) -- torcwa's own numerical behavior at float32 was not
  evaluated here, and the RCWA cross-check's whole purpose is a tight
  independent accuracy check, so keeping its highest available precision
  is the conservative choice regardless of this policy.
- **No solver step has yet been "demonstrated numerically unstable in
  single precision"** in the sense the spec allows as a float64
  carve-out. If E1-E7 running at scale surfaces one (e.g. a
  high-iteration-count IMEX integration diverging or losing gradient
  signal at float32), that specific path should be pinned back to
  float64 explicitly (via `run_job`'s `dtype` parameter) and documented
  here, not silently reverted pipeline-wide.

## Mechanism
`holomedia.optimize`'s per-method functions previously hardcoded
`torch.float64`/`torch.complex128` for their internal init tensors
(theta, GS phase, `oracle_unconstrained`'s `dn_free`) regardless of what
dtype the `NPDDRecorder`/`SlabBPM` passed in were actually constructed
at -- a real inconsistency, found while implementing this policy, fixed
in the same pass (`_complex_dtype_for`, and every method now derives its
working dtype from `recorder.dtype` instead of a hardcoded literal).
Verified bit-for-bit unchanged at float64 (full test suite) and verified
functioning correctly at float32 (all 6 methods, `tests/test_method_registry.py`
pattern extended informally during this pass).
