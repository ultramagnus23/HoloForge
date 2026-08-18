# Literature validation curves — status

**Updated 2026-08-18: one real source is now digitized.** Bruder, Fäcke &
Rölle 2017 (Bayfol HX, MDPI *Polymers*, DOI 10.3390/polym9100472) is CC-BY
open access. Its full-resolution Figure 3 (Δn₁ vs. average exposure dose
E_ave, Λ=700 nm i.e. K≈8.98 rad/µm, simulated + experimental series) was
fetched directly from the publisher CDN (not paywalled) and read visually
point-by-point — see `source_figures/bruder2017_fig3.png` for the source
image and `bruder2017_growth_K8.98_{sim,exp}.csv` for the extracted points.

**This is a visual read, not a WebPlotDigitizer pixel-calibrated
extraction** — no interactive pixel-picking tool was used, so treat the
x-values (log-scale axis) as accurate to roughly ±10% and y-values to
roughly ±0.0005 in Δn₁. Good enough to fit against and report an honest
NRMSE, not good enough to claim WebPlotDigitizer-grade precision in the
manuscript text. Re-digitizing with WebPlotDigitizer against the same
source image would tighten this if needed before submission.

**Known integration gap (not yet resolved):** `experiments/fit_literature_curves.py`'s
`growth` curve type currently fits diffraction efficiency (post-Kogelnik
conversion) against the twin's internal exposure-*time* units. This new
CSV reports raw Δn₁ against physical dose (mJ/cm²) — a different
quantity on a different axis. `MediumParams.kappa` already "folds in
[the] intensity scale" (see `holomedia/npdd.py`'s docstring), so a
dose-axis fit is physically sound in principle, but `fit_curve()` needs a
new `growth_dn` branch that reuses `NPDDRecorder`'s `dn1` output directly
(skipping the Kogelnik DE conversion `simulate_growth_de` currently
applies) before this CSV can actually be fit. Not done yet — flagged here
so it isn't silently skipped.

**Still genuinely blocked by paywall:** the two Optica-published NPDD
papers below (Kelly & Sheridan 2011, Gleeson et al. 2008) sit behind
`opg.optica.org`'s subscription wall — confirmed again this pass, not
just carried over from before. Same options as always: institutional
access (check whether Ashoka's library has an Optica agreement), or paste
the figure image directly and it can be digitized the same way Figure 3
above was.

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
