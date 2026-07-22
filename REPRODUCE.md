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

## 1. Run the science manifests (Colab T4 GPU; see `notebooks/colab_runner.ipynb`)

```bash
python -m experiments.run_manifest --manifest all --probe   # budget check first
python -m experiments.run_manifest --manifest E1 --max-minutes 170
python -m experiments.run_manifest --manifest E2 --max-minutes 170
python -m experiments.run_manifest --manifest E3 --max-minutes 170
python -m experiments.run_manifest --manifest E4 --max-minutes 170
python -m experiments.run_manifest --manifest E5 --max-minutes 170
# E6 optional, only if Gate 1 budget allows:
python -m experiments.run_manifest --manifest E6 --max-minutes 170
# E7 needs no GPU, runs on CPU (torcwa):
python experiments/rcwa_crosscheck.py e7
```

Each `--manifest` invocation is resumable -- re-running the identical
command after an interruption skips every already-completed job and
picks up where it left off (see `experiments/run_manifest.py`'s
docstring).

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
# writes figures/paper/F1_*.pdf ... F8e_*.pdf
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
