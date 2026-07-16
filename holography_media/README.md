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
- [~] Digitize literature growth curves into data/literature/ (WebPlotDigitizer) —
      **not digitized** (paywalled figures); real papers/protocol documented in
      `data/literature/README.md`, and `f1_validate_twin.py` will overlay CSVs
      automatically once added.
- [x] Fill Table 1 parameter ranges with citations — real values from Kelly &
      Sheridan 2011, Gleeson/Guo 2008, Fomenko & Berneth 2017 (Bayfol),
      Jeong et al. 2022 (PQ/PMMA); see paper Table 1 and its precision caveat.
- [~] Bump sweeps to 5 seeds / 800+ iters on GPU — **no GPU in this environment**;
      ran a confirmation-scale rerun instead (n_x=512, 500 iters, 3 seeds,
      `experiments/run_confirm.py`, ~2-4h CPU). Full paper-scale command for a
      GPU box:
      ```bash
      # edit experiments/f2_f3_recovery.py: N_X=1024, N_ITERS=800, SEEDS=[0,1,2,3,4]
      python experiments/f2_f3_recovery.py
      # estimated ~26h on a single CPU core at that scale (measured: ~180s per
      # (target, seed) for media_in_the_loop alone x 26 settings x 4 targets x 5
      # seeds); a modern GPU should bring this to well under an hour.
      ```
- [x] Adjoint + neural-surrogate gradient ablations — `experiments/ablation_gradients.py`,
      `holomedia/npdd.py::forward_checkpointed`, `holomedia/surrogate.py`.
- [x] RCWA cross-check (torcwa) on 3 grating cases — `experiments/rcwa_crosscheck.py`,
      max |Kogelnik − RCWA| = 0.038 absolute DE over K = 2-12 rad/µm.
- [x] 3D (x,y,z) showcase case — `experiments/showcase_3d.py`,
      `holomedia/npdd3d.py`, `holomedia/diffraction3d.py`.
- [x] Plotting scripts -> paper figures — `experiments/make_figures.py` now reads
      `results_confirm.json` (preferred) or `results_prelim.json` (fallback),
      plus `results_ablation_gradients.json` / `results_rcwa.json` for Figs E/F.
- [x] v0.2 corrected results (results_prelim*.json, figures/) — NPDD non-local bug fixed, see CHANGELOG
- [~] GPU-scale rerun (1024+ grid, 800+ iters, 5 seeds) to replace [RESULT-FINAL]
      markers — confirmation-scale (not full paper-scale) rerun done this pass;
      see the bullet above for the exact full-scale command/runtime.
