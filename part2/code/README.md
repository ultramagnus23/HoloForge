# holomedia: media-in-the-loop holography

Code, experiment manifests and results for the paper *Media-in-the-loop
holography: designing photopolymer exposures through a differentiable
recording model* (submitted to JOSA A).

Camera-in-the-loop correction needs a rewritable display; photopolymer holograms
are write-once, so the reconstruction error can be measured only after it is
permanent. This code places a differentiable model of the recording itself
(non-local polymerization-driven diffusion, NPDD, with split-step
beam-propagation readout) inside the exposure-design loop and compares it, in
simulation, with the same optimizer under a linear, media-blind recording
assumption.

## Layout

```
holomedia/            core library
  npdd.py             differentiable NPDD recording model, saturation-only surrogate
  diffraction.py      split-step BPM readout, Kogelnik closed form
  optimize.py         media-in-the-loop optimizer and every baseline
experiments/          manifests (manifest.py), runner (run_manifest.py),
                      S3/S6/S7 runners, twin fits, RCWA and gradient checks
analysis/             aggregation -> results/summary/paper_numbers.json
figures/              make_all.py renders every paper and supplement figure
scripts/              numbers.tex, consistency check, length check,
                      submission package, campaign launchers
configs/media/        medium parameter files
data/literature/      digitized literature curves and their source figures
results/              per-job result files, one JSON per tier/config/method/seed
tests/                pytest suite
docs/                 parameter provenance, precision policy
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m pytest -q tests
```

`REPRODUCE.md` lists every command that regenerates the paper's numbers,
figures and submission package from the committed results, and how to rerun
the simulations.
