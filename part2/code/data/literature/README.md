# Literature validation curves — status

**Honest status: not digitized.** True WebPlotDigitizer-style extraction needs
pixel coordinates picked off the actual published figure image. The papers
below are real, cited, and (per search) contain the right kind of curve, but
their full-text/figures sit behind an Optica/Elsevier/RSC paywall that this
pass could not fetch — abstracts and secondary text do not carry enough
precision to reconstruct a curve responsibly. Rather than approximate or
invent points, this directory is left empty of data and this file documents
exactly what to digitize and how, so it's a 20-minute task once you have PDF
access (via institutional login or by pasting the figure image to me).

## What to digitize

1. **DE (or Δn) vs. exposure-time/dose growth curves at several spatial
   frequencies**, for the NPDD signature the twin is validated against in
   `experiments/f1_validate_twin.py` panel (a): growth → saturation → high-K
   rolloff.
   - Sheridan/Kelly/Gleeson NPDD series, e.g. Kelly & Sheridan,
     "Monomer diffusion rates in photopolymer material. Part I,"
     *J. Opt. Soc. Am. B* 28(4):658 (2011), and the companion Part II/replies.
   - Look for the figure plotting diffraction efficiency (or recorded Δn)
     against exposure time/dose for at least 3 spatial frequencies on one
     axes set — this is the panel-(a) analogue.

2. **Recorded contrast vs. spatial frequency** (rolloff / MTF-shaped curve),
   for panel (b) — compared against `NPDDRecorder.small_signal_mtf` (Eq. 5,
   `holomedia/npdd.py`).
   - Same NPDD series; also Gleeson, Liu, Guo & Sheridan on chain-transfer
     agents and spatial-frequency response, *J. Opt. Soc. Am. B* 25(3):396
     (2008) — this paper's whole point is the σ-vs-spatial-frequency-response
     tradeoff, so its main figure is close to a direct match.

3. **Kogelnik angular selectivity curve** (DE vs. angular detuning) for
   panel (c) — any PVA/AA or Bayfol paper reporting a Bragg-selectivity scan;
   Fomenko & Berneth, *Polymers* 9(10):472 (2017) (open access, PMC6418958)
   is a candidate and IS fetchable in full text, but the version fetched in
   this pass returned prose/quantitative call-outs, not the figure's raw
   data points — the actual angular-selectivity plot in that paper (Fig. ~7-9
   region per the text's Table 3 discussion) still needs manual digitizing.

## CSV schema (Phase 6)

Every digitized CSV MUST have this exact header row (required columns, in
any order, but `x` and `y` must be present or the loader skips the file
with a warning rather than guessing):

```
x,y,source_doi,figure_id,digitized_by,date
```

| Column | Meaning |
|---|---|
| `x` | horizontal axis value, in the paper's own units (exposure time/dose, or angular detuning in degrees for panel (c) -- state which in the filename, see below) |
| `y` | vertical axis value (DE, or Δn, or normalized DE for angular selectivity) |
| `source_doi` | DOI (or arXiv ID if no DOI) of the source paper, same value repeated on every row of a given curve |
| `figure_id` | e.g. `Fig3a`, `Fig7` -- which figure/panel this curve was digitized from |
| `digitized_by` | who ran WebPlotDigitizer (name/initials) |
| `date` | ISO date (YYYY-MM-DD) the digitization was done |

One CSV per curve/series (not per figure -- a figure with 3 spatial-
frequency series becomes 3 CSVs). `experiments/f1_validate_twin.py`'s
`load_literature_curves()` reads `x,y` for plotting/overlay;
`experiments/fit_literature_curves.py` (Phase 6 fitting script) reads the
full schema including provenance.

## Digitization protocol (WebPlotDigitizer)

1. Open the figure image in https://apps.automeris.io/wpd/ (or equivalent).
2. Calibrate axes using two known tick values on each axis.
3. Pick points along each curve/series.
4. Export CSV with the header above; save as
   `data/literature/<short_name>_<panel>.csv`, e.g.
   `sheridan2011_growth_K6.csv`, `gleeson2008_mtf.csv`,
   `fomenko2017_angular_selectivity.csv`.
5. Fill in `source_doi`/`figure_id`/`digitized_by`/`date` on every row --
   this is the provenance the paper's Phase-6 validation section cites
   directly, not a separate sources.json.

## What IS real in this pass

Table 1 in `paper/part2_media_draft_v1.md` and `configs/media/*.yaml` were
updated with real cited numeric values (diffusion coefficients, Δn ranges,
shrinkage percentages) pulled from full text where fetchable (open-access
PMC copies) and from search-result quotes where not — see the citations
inline in Table 1. Those are point values from text, not digitized curves,
which is a materially different (and easier, already-done) task from what
this README describes.

## How `f1_validate_twin.py` uses this directory

If CSVs matching `*.csv` appear here, `experiments/f1_validate_twin.py` will
load and overlay them (see `load_literature_curves()`); if the directory is
empty, it runs the twin-only validation and prints a note that literature
overlay is unavailable, rather than silently pretending agreement.
