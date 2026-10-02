# Reproducing the Part 2 manuscript

Every number in `part2/paper/manuscript.tex`, `supplement.tex` and
`cover_letter.tex` is a LaTeX macro in `part2/paper/numbers.tex`, generated
from the per-job result files in `results/` and the committed `results_*.json`
files. Nothing is typed by hand: `scripts/check_consistency.py` fails if a
number appears next to "dB", "seed" or "rad/" outside a macro, and
`scripts/make_submission_package.py` refuses to build if any `[PENDING]` macro
reaches a PDF.

All commands run from `part2/code/`. On Windows set `PYTHONUTF8=1`.

## 0. Environment

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q tests
```

## 1. Regenerate numbers, figures and the submission package (minutes, CPU)

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

```bash
python scripts/make_submission_package.py
```

Order matters: `make_numbers_tex.py` rewrites `numbers.tex`, and the two
`analysis/*.py` scripts append their blocks afterwards. The last command
compiles the manuscript, Supplement 1 and the cover letter in a clean
directory and writes the upload-ready files to `part2/submission/`
(see `part2/paper/SUBMISSION_GUIDE.md`). `manuscript_lengthcheck.tex` is the
same body in Optica's two-column journal class, for the page count JOSA A
bills against.

## 2. Rerun the simulations (hours to days)

Each tier is a manifest in `experiments/manifest.py`; the runner resumes by
skipping any job whose result file exists, so it can be stopped and
restarted.

| Tier | What | Jobs |
| --- | --- | --- |
| M1A / M1B | main grid (coarse 9 K / fill-in) | 504 / 441 |
| M1C | the K = 3.93 cell of M1B (all budgets) | 63 |
| M2R | compute-matched arm, budget 2x | 27 |
| S1 | one-at-a-time mechanism ablation | 90 |
| S1X | saturation-mechanism factorial and controls | 216 |
| S2R | re-optimized parameter sensitivity, one K | 78 |
| S4, S5 | target families, detector noise | 24, 6 |
| S8 | slant dependence | 84 |
| NZ0 | readout slice-count convergence | 18 |

S3, S6 and S7 (design once, evaluate many) run through
`experiments/run_s3_mismatch.py`, `run_s6_joint_mismatch.py` and
`run_s7_depth_absorption.py`.

GPU (four workers sharing one GPU):

```bash
QUEUE=M1A,M1B,S8,S1,S4,S5 scripts/launch_campaign.sh 0 1 2 3
```

CPU (used for S1X, M1C, M2R and S2R; a MIL job at n_x = 1024 takes about
6 min on 4 threads):

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

Kogelnik versus RCWA check (CPU, needs `torcwa`):

```bash
python experiments/rcwa_crosscheck.py e7
```

Finite-difference gradient check (CPU, seconds):

```bash
python experiments/gradient_check.py
```

Single-seed illustrations (Fig. 2 and Fig. S1):

```bash
python -m experiments.make_r1_reconstructions
```

```bash
python -m experiments.make_r1_profiles
```

## Notes

- Result files record `git_commit`, `device` and `dtype`. The geometry is
  frozen (`n_z = 128`, `slant_deg = 0`) and both keys are written into every
  job config, so results from an older geometry cannot collide by hash.
- CPU and GPU runs are both float32 and differ slightly in rounding; the S1X
  full-model cell reproduces the GPU S1 value (macro `\SXDeviceMaxDiff`).
- Only M1 cells in which every method finished every seed enter the headline
  statistics (`analysis/aggregate.py::split_complete_m1`).
- Long unattended runs on a laptop: disable OS sleep and keep the machine on
  power. The runner's per-process sleep inhibition does not stop a closed
  lid or a forced update reboot.
