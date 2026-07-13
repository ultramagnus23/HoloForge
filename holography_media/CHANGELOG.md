# Changelog

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
