# Digitized literature curves

Data behind the twin validation (manuscript Section 5, Fig. 4; Supplement 1,
Section S9 and Fig. S5).

| File | Source | Curve |
| --- | --- | --- |
| `bruder2017_growth_dn_K8.98_sim.csv` | Bruder, Fäcke & Rölle, *Polymers* 9, 472 (2017), Fig. 3, DOI 10.3390/polym9100472 (CC BY) | Bayfol HX Δn₁ vs. average exposure dose (mJ/cm²), Λ = 700 nm (K = 8.98 rad/µm); the source's own kinetic-model curve at 16.7 mW/cm² |
| `bruder2017_growth_dn_K8.98_exp.csv` | same figure | the measured series at 19.2 mW/cm² |
| `hsieh2022_growth_dn_K24.94.csv` | Hsieh, Cheng & Chung, *ACS Omega* 7, 11770 (2022), Fig. 2b, DOI 10.1021/acsomega.1c06887 (CC BY-NC-ND) | PQ/PMMA Δn₁ vs. exposure time (s), Λ = 251.96 nm (K = 24.94 rad/µm); the source's own model output, outside the paper's tested K range |

`source_figures/` holds the published figure images the points were read from.

## How the points were obtained

The points were read visually from the figure images by an AI model (Claude,
Anthropic; column `digitized_by = claude-visual-read`), not with a calibrated
digitizer. Estimated accuracy: about ±10% in x (worse on the Bayfol log axis)
and ±3–5% of each curve's range in y. The manuscript states this and discloses
the AI use in its Acknowledgment. Re-digitizing the images in `source_figures/`
with WebPlotDigitizer (https://apps.automeris.io/wpd/) and replacing the CSVs is
a cheap improvement; every downstream number regenerates from them.

## CSV schema

```
x,y,source_doi,figure_id,digitized_by,date
```

`x` and `y` are in the source figure's own units; the other columns repeat the
provenance on every row.

## Consumers

- `experiments/fit_literature_curves.py`: fits with κ and Δn_max free and
  k_bleach held (`results_literature_fit.json`, Fig. S5).
- `experiments/fit_twin_holdout.py`: fits with k_bleach also free and the
  held-out cross-series prediction between the two Bayfol series
  (`results_twin_holdout.json`, Fig. 4).

Each curve is fitted against its own medium configuration in
`configs/media/` (`bayfol_hx_405nm.yaml`, `pq_pmma_405nm.yaml`).

## Not available

Multi-K growth-curve families for NPDD media (for example the Sheridan, Kelly
and Gleeson papers in *J. Opt. Soc. Am. B*) would identify D₀ and σ, but were
not accessible for this study.
