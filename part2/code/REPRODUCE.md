# Reproducing the Part 2 manuscript

Every number in `part2/paper/oe_main.tex` and `oe_supplement.tex` is a LaTeX
macro generated from the per-job result files in `results/`. Nothing is typed
by hand; `scripts/check_consistency.py` fails if a hard-coded number appears
next to "dB", "seed" or "rad/".

All commands run from `part2/code/`. On Windows set `PYTHONUTF8=1`.

## 0. Environment

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q tests
```

## 1. Regenerate numbers and figures from the committed results (minutes, CPU)

```bash
python analysis/aggregate.py
```

```bash
python scripts/make_numbers_tex.py
```

```bash
python analysis/de_confirmation.py
```

```bash
python analysis/wasted_media.py
```

```bash
python figures/make_all.py
```

```bash
python scripts/check_consistency.py
```

```bash
python scripts/make_lengthcheck.py
```

Then build `part2/paper/oe_main.tex` and `oe_supplement.tex` with
`pdflatex` / `bibtex` / `pdflatex` / `pdflatex`. `oe_main_lengthcheck.tex`
is the same body in Optica's two-column class, for page counting.

Order matters: `make_numbers_tex.py` rewrites `numbers.tex`, and the two
`analysis/*.py` scripts append their blocks to it afterwards.

## 2. Rerun the simulations (hours to days)

Each tier is a manifest in `experiments/manifest.py`; the runner resumes by
skipping any job whose result file exists, so it can be stopped and
restarted.

| Tier | What | Jobs |
| --- | --- | --- |
| M1A / M1B | main grid (coarse 9 K / fill-in) | 504 / 441 |
| M1C | the K = 3.93 cell of M1B (all budgets) | 63 |
| M2 / M2R | compute-matched arm (full / budget 2x only) | 81 / 27 |
| S1 | one-at-a-time mechanism ablation | 90 |
| S1X | saturation-mechanism factorial + controls | 216 |
| S2R | one-K parameter sensitivity | 78 |
| S4, S5 | target families, detector noise | 24, 6 |
| S8 | slant dependence | 84 |
| NZ0 | slice-count convergence | 18 |

S3, S6, S7 (design/evaluate splits) run through
`experiments/run_s3_mismatch.py`, `run_s6_joint_mismatch.py` and
`run_s7_depth_absorption.py`.

GPU (four workers sharing one GPU):

```bash
QUEUE=M1A,M1B,S8,S1,S4,S5 scripts/launch_campaign.sh 0 1 2 3
```

CPU (used for S1X, M1C, M2R, S2R when CUDA was unavailable; a MIL job at
n_x = 1024 takes roughly 6 min on 4 threads):

```bash
scripts/launch_cpu_queue.sh 0 1 2 3
```

Twin fits and the held-out cross-series prediction (CPU, about 1 h):

```bash
python experiments/fit_literature_curves.py
```

```bash
python experiments/fit_twin_holdout.py
```

Single-seed illustration figures (R1, R3):

```bash
python -m experiments.make_r1_reconstructions
```

```bash
python -m experiments.make_r1_profiles
```

## Notes

- Result files record `git_commit`, `device` and `dtype`. The geometry is
  frozen (`n_z = 128`, `slant_deg = 0`) and both keys are written into every
  job config, so results from older geometries cannot collide by hash; those
  live in `results_archive/`.
- CPU and GPU runs are both float32 and differ slightly in rounding; the S1X
  full-model cell reproduces the GPU S1 value (the difference is reported in
  the supplement, macro `\SXDeviceMaxDiff`).
- Only M1 cells in which every method finished every seed enter the headline
  statistics (`analysis/aggregate.py::split_complete_m1`).
- Long unattended runs on a laptop: disable OS sleep and keep the machine on
  power. The runner's per-process sleep inhibition does not stop a closed
  lid or a forced update reboot.
