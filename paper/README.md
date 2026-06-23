# HoloForge — Part 1 Paper

`part1_main.tex` — *Systematic Degradation Analysis in Phase-Only Computational
Holography: A Simulation Framework.* IEEE conference format (`IEEEtran`).

## Building

The paper has **no external `.bib`** (references are embedded in a
`thebibliography` block), so a plain run is enough:

```bash
pdflatex part1_main.tex
pdflatex part1_main.tex   # second pass resolves \ref / \cite cross-references
```

Or drop `part1_main.tex` into [Overleaf](https://overleaf.com) — it compiles
as-is with the default `IEEEtran` class.

## Figures

All `\includegraphics` paths point at `../holography_sandbox/results/`, which is
populated by:

```bash
cd ../holography_sandbox && python run_experiments.py
```

Run that first (it takes ~35 s on CPU) so the seven referenced figures exist:
`gs_convergence.png`, `sweep_resolution.png`, `sweep_phase_bits.png`,
`sweep_viewing_angle.png`, `sweep_speckle.png`, `metrics_summary_plot.png`,
`multi_scene_summary.png`.

## What still needs filling in by the authors

Every **numerical** value is already populated verbatim from the result CSVs.
What remains is author-supplied metadata, not data:

- **Author block** — three `[Author … ]` / `[Institution]` / `[email]`
  placeholders in the `\author{}` macro.
- Optional: expand any Related Work paragraph with venue-specific citations.

The quantitative claims trace to:
`metrics_summary.csv`, `multi_scene_summary.csv`, `seed_sensitivity_summary.csv`
(single-scene numbers use the `gaussian_spots` reference scene; "across-scene"
numbers are the mean ± std over the four-scene suite).

## Scope note

Part 1 is deliberately limited to **single-wavelength (532 nm)** holography and
**objective metrics only**. Multi-wavelength (RGB) colour and the human-observer
perceptual-threshold study are deferred to **Part 2**.
