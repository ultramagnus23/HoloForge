# Reproducing HoloForge Part 2 results and figures

This is the exact command sequence that regenerates every table and
figure in `part2/paper/oe_main.tex` / `part2/paper/oe_supplement.tex` from
the archived result JSONs in `part2/code/results/`. All commands are run
from `part2/code/`.

## 0. Environment

```bash
cd part2/code
pip install -r requirements.txt
python tests/test_smoke.py                # ~2 min, verifies the core pipeline
```

## 1. Run the science manifests (M1/M2/S1/S2; GPU required)

Two ways to run these, same underlying engine (`experiments/run_manifest.py`)
either way -- resume, atomic writes, stall detection, and per-job seeding
behave identically:

**Option A -- local NVIDIA GPU** (e.g. a laptop RTX with 4GB+ VRAM; every
result field in this pipeline is 1D, so VRAM is not the bottleneck, wall
clock time is):

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
cd part2/code
python -m experiments.run_local            # runs M1, M2, S1, S2 in order, resumable
# or double-click part2/code/run_local.bat
```

**Option B -- Colab T4 GPU** (see `notebooks/colab_runner.ipynb`, or run the
same commands manually):

```bash
python -m experiments.run_manifest --manifest all --probe   # budget check first
python -m experiments.run_manifest --manifest M1 --max-minutes 170
python -m experiments.run_manifest --manifest M2 --max-minutes 170
python -m experiments.run_manifest --manifest S1 --max-minutes 170
python -m experiments.run_manifest --manifest S2 --max-minutes 170
```

S3 (twin-miscalibration robustness) does NOT run through
`run_manifest.py`: a job there is "one design, many evaluations", which is
not `run_job()`'s one-config-one-result shape. It has its own runner, and
is cheap -- the only optimization is the design stage (2 methods x 3 K x 3
seeds); everything else is a single forward pass:

```bash
python -m experiments.run_s3_mismatch
```

Resumable the same way: designed exposures are checkpointed under
`results/S3/_designs/` and each evaluation writes its own content-hashed
JSON, so re-running the identical command skips whatever is already done.

V3 needs no GPU, runs on CPU (torcwa):

```bash
python experiments/rcwa_crosscheck.py e7
```

V1/V2 are not yet execution-ready through this manifest system (see
`experiments/manifest.py`'s `VALIDATION_BUILDERS` docstring) -- V1 overlaps
with `experiments/fit_literature_curves.py`, V2 needs a genuine 3-way
Kogelnik/BPM/RCWA comparison that doesn't exist yet.

Each `--manifest` invocation (and `run_local.py`, which wraps the same
function) is resumable -- re-running the identical command after an
interruption skips every already-completed job and picks up where it left
off (see `experiments/run_manifest.py`'s docstring).

As of the current manuscript, M1 (765 jobs, including the 135 SAT
saturation-only-surrogate jobs), M2 (54 jobs), S1 (90 jobs), S2 (312
jobs) and S3 (504 evaluations from 18 designs) have all completed for
real on GPU -- this is the data behind `oe_main.tex`'s Results section.

### A note on seeds

The analysis uses **3 seeds** per iterative method (`manifest.PAPER_SEEDS`),
and `analysis/aggregate.py` enforces that with an explicit
`ANALYSIS_SEEDS = {0, 1, 2}` filter that reports how many files it drops.

`experiments/run_m1_seedbump.py` exists to raise M1 from 3 seeds to 8. It
was started and then stopped at roughly 16% (146 of 900 jobs, 8 of 45
configurations). Its output is real and uncorrupted and is kept on disk
and in version control -- but it is NOT used, because a partial bump makes
M1 unbalanced: a minority of configurations would carry 8 seeds and the
rest 3, so per-point confidence intervals would not be comparable across
the grid and no single seed count would be a true statement about the
study. Mixing them in also visibly contaminated the diffraction-efficiency
numbers (`DENPairs` became 157 rather than 45 configs x 3 seeds = 135).

To use 8 seeds, finish the bump for all 45 configurations first (budget
~52 GPU-hours on a laptop 3050, ~44 of them media-in-the-loop), then set
`ANALYSIS_SEEDS = None` and re-run steps 2-4.

## 2. Aggregate statistics

```bash
python -m analysis.aggregate
# writes results/summary/paper_numbers.json
```

## 3. Generate the LaTeX number macros

`numbers.tex` is assembled by FOUR scripts, and the order matters:
`make_numbers_tex.py` rewrites the file from scratch, while the other
three append their own marked blocks to it. Running only the first
silently drops the DE-confirmation, wasted-media and 2D macros, and the
manuscript then fails to build with `Undefined control sequence` on
`\DEMeanGainTwoX` and friends. Run all four, in this order:

```bash
python scripts/make_numbers_tex.py   # rewrites ../paper/numbers.tex
python -m analysis.de_confirmation   # appends DE* macros
python -m analysis.wasted_media      # appends WastedMedia* macros
python -m analysis.aggregate_2d      # appends TwoD* macros
```

Any macro that cannot be computed from the data present is emitted as a
red `[PENDING]` flag rather than omitted or guessed, so a missing input
shows up in the compiled PDF instead of failing silently.

## 4. Build figures

```bash
python -m figures.make_all
# writes figures/paper/F1_*.pdf ... F9c_*.pdf
```

## 5. Twin validation (needs digitized literature CSVs -- see
   `data/literature/README.md` for the digitization protocol; this step
   is skipped if `data/literature/*.csv` doesn't exist)

```bash
python experiments/fit_literature_curves.py
# writes ../results_literature_fit.json
```

Deferred for now (Twin Validation section of the manuscript is not yet
written) -- pending literature-curve digitization.

## 6. Build the manuscript

```bash
cd ../paper
python ../code/scripts/check_refs.py         # must exit 0
python ../code/scripts/check_consistency.py  # must exit 0
pdflatex oe_main.tex
bibtex oe_main
pdflatex oe_main.tex && pdflatex oe_main.tex
pdflatex oe_supplement.tex && pdflatex oe_supplement.tex
```

`oe_main_lengthcheck.tex` carries the SAME body as `oe_main.tex` and
differs only in preamble: it uses Optica's dedicated length-check class
(`opticajnl`, 9pt/twocolumn/twoside), which is what gives an
authoritative composed page count against the 10-page fee threshold.
Editing `oe_main.tex` without re-deriving it makes the page-budget check
measure a stale manuscript, so re-derive rather than hand-editing both.

This builds against the real Optica universal template
(`optica-article.cls` / `opticajnl.bst`, both vendored in `part2/paper/`),
plus the `jabbrv` package (also vendored there) that the class requires.
`part2/paper/styles/opticajournal.sty` is a labeled stopgap standing in for
Optica's real per-journal style file -- swap it before final submission.

## Archival status

- **License**: Apache 2.0 (OSI-approved), declared at repo root (`LICENSE`).
- **Zenodo DOI**: not yet cut. Per the project's own ground rules, a
  versioned release is tagged at "results freeze". Once you're ready to
  freeze results: tag a release (`git tag vX.Y -m "..."`, `git push --tag`)
  and deposit it via Zenodo's GitHub integration
  (https://zenodo.org/account/settings/github/), which mints a DOI
  automatically from the tag. Update `part2/paper/oe_main.tex`'s Code and
  Data Availability section with the resulting DOI.
- **Raw results**: every result JSON under `part2/code/results/`
  (plus `results_prelim*.json`, `results_confirm.json`,
  `results_ablation_gradients.json`, `results_rcwa*.json`) is
  append-only and committed to git -- the Zenodo deposit is a snapshot of
  the same tree, not a separate export.
