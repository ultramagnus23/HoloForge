# holoforge-media

**Media-in-the-Loop Holography** — CGH optimization through a differentiable
digital twin of volume photopolymer recording (NPDD + split-step BPM).
Extension of the HoloForge project.

## Why
Camera-in-the-loop holography needs a rewritable display. Photopolymer
holograms are write-once — feedback arrives after the error is permanent.
So we put a differentiable model of the recording *chemistry* in the loop
instead, and optimize the delivered exposure (nonnegative, dose-budgeted)
rather than an idealized phase pattern.

## Layout
```
holomedia/          core library
  npdd.py           differentiable NPDD recording twin (+ analytic MTF/cliff)
  diffraction.py    Kogelnik closed form + split-step BPM readout
  optimize.py       media-in-the-loop optimizer + media-blind baselines
experiments/
  f1_validate_twin.py    twin validation vs literature (paper Fig 1)
  f2_f3_recovery.py      method comparison + parameter sweeps (Figs 2-3)
paper/
  draft_v1.md       full paper draft ([RESULT] placeholders = run the code)
tests/
  test_smoke.py     end-to-end pipeline check
configs/            medium parameter files (fill from Table 1 reading)
```

## Quickstart
```bash
pip install torch numpy
python tests/test_smoke.py          # ~2 min CPU, verifies everything
python experiments/f1_validate_twin.py
python experiments/f2_f3_recovery.py   # long; run on GPU, bump SEEDS/N_ITERS
```

## Status / TODO
- [ ] Digitize literature growth curves into data/literature/ (WebPlotDigitizer)
- [ ] Fill Table 1 parameter ranges with citations
- [ ] Bump sweeps to 5 seeds / 800+ iters on GPU
- [ ] Adjoint + neural-surrogate gradient ablations
- [ ] RCWA cross-check (torcwa) on 3 grating cases
- [ ] 3D (x,y,z) showcase case
- [ ] Plotting scripts -> paper figures
- [x] v0.2 corrected results (results_prelim*.json, figures/) — NPDD non-local bug fixed, see CHANGELOG
- [ ] GPU-scale rerun (1024+ grid, 800+ iters, 5 seeds) to replace [RESULT-FINAL] markers
