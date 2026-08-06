# Reproducing HoloForge Part 2 results and figures

This is the exact command sequence that regenerates every table and
figure in `paper/oe_main.tex` / `paper/oe_supplement.tex` from the
archived result JSONs in `holography_media/results/`. All commands are
run from `holography_media/`.

## 0. Environment

```bash
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
cd holography_media
python -m experiments.run_local            # runs M1, M2, S1, S2 in order, resumable
# or double-click holography_media/run_local.bat
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

## 2. Aggregate statistics

```bash
python -m analysis.aggregate
# writes results/summary/paper_numbers.json
```

## 3. Generate the LaTeX number macros

```bash
python scripts/make_numbers_tex.py
# writes ../paper/numbers.tex
```

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

## 6. Build the manuscript

```bash
cd ../paper
python ../holography_media/scripts/check_refs.py         # must exit 0
python ../holography_media/scripts/check_consistency.py  # must exit 0
pdflatex oe_main.tex && pdflatex oe_main.tex        # twice, for \bibliography
bibtex oe_main && pdflatex oe_main.tex && pdflatex oe_main.tex
pdflatex oe_supplement.tex && pdflatex oe_supplement.tex
```

## Archival status

- **License**: Apache 2.0 (OSI-approved), declared at repo root (`LICENSE`).
- **Zenodo DOI**: not yet cut. Per the project's own ground rules, a
  versioned release is tagged at "results freeze" -- which has not
  happened, since Phase 3 (the GPU science runs) has not run yet as of
  this writing. Tagging a release now, before there are any real
  paper-scale results to freeze, would be premature. Once Phase 3
  completes and you're ready to freeze results: tag a release
  (`git tag vX.Y -m "..."`, `git push --tag`) and deposit it via
  Zenodo's GitHub integration (https://zenodo.org/account/settings/github/),
  which mints a DOI automatically from the tag. Update
  `paper/oe_main.tex`'s Code and Data Availability section with the
  resulting DOI (currently marked `[TODO: Zenodo DOI -- pending results
  freeze]`).
- **Raw results**: every result JSON under `holography_media/results/`
  (plus `results_prelim*.json`, `results_confirm.json`,
  `results_ablation_gradients.json`, `results_rcwa*.json`) is
  append-only and committed to git -- the Zenodo deposit is a snapshot of
  the same tree, not a separate export.
