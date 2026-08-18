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
      markers — confirmation-scale (not full paper-scale) rerun done in the
      CPU-only pass; the FULL 5-seed method-comparison sweep (`f2_f3_recovery.py`
      at paper scale, driving Figs 2/3's [RESULT-FINAL] markers) still has not
      been run on GPU — see the bullet above for the exact command. **What HAS
      been run on GPU** (single seed, solver-characterization scope, not the
      F2/F3 method comparison): mesh-density + convergence-threshold sweep for
      the NPDD solver, and a wavelength sweep for the BPM readout — see below.
- [x] GPU-scale NPDD mesh-density + convergence-threshold sweep —
      `experiments/gpu_npdd_mesh_convergence_sweep.py`, run on a Colab T4,
      n_x=1024 paper-scale mesh, 51.2 µm fixed physical window.
      Results: `results/gpu_reruns/npdd_mesh_sweep/results.json`.
      - Mesh density (n_x = 512/1024/2048, dx scaled to keep the window fixed,
        800-iter fixed budget): PSNR is mesh-independent within 0.04 dB
        (6.65/6.63/6.61 dB) and wall-clock is essentially flat (459/452/445 s)
        across a 4x resolution range — at this problem size the GPU run is
        overhead-bound, not compute-bound, so finer mesh is close to free.
      - Convergence threshold (new `converge_tol` early-stop option added to
        `media_in_the_loop`; tol swept at fixed n_x=1024): tol=1e-3 stops at
        iter 320 (174 s, 6.53 dB); tol=1e-4 stops at iter 630 (351 s, 6.63 dB);
        tol=1e-5 runs to iter 1350 (749 s, 6.63 dB) — tol=1e-4 already matches
        tol=1e-5's quality at under half the wall-clock, i.e. diminishing
        returns set in well before the fixed 800-iter budget.
- [x] GPU-scale BPM wavelength sweep — `experiments/gpu_bpm_wavelength_sweep.py`,
      run on a Colab T4: one recording at the 405 nm design wavelength
      (n_x=1024, 800 iters), then readout of the SAME recorded profile at
      400/405/420/435/450 nm (brackets the paper's stated 405/450 nm band).
      Results: `results/gpu_reruns/bpm_wavelength_sweep/results.json`.
      PSNR/DE peak at the 405 nm design point (6.63 dB / 0.589) and fall off
      away from it, but **non-monotonically** (420 nm: 3.28 dB/0.433 is worse
      than 435 nm: 3.79 dB/0.516) — consistent with Kogelnik's oscillatory
      sin²(ν) detuning response rather than a smooth rolloff; reported as
      observed rather than smoothed into a monotonic story.
