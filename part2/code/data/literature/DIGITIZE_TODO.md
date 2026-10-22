# Datasets to digitize for the Applied Optics multi-dataset validation

Status: **blocked on you.** Ground rule 2 of this revision is that literature
data comes only from CSVs in this directory. Nothing below is in the repository
yet, so the Phase 1a cross-dataset held-out validation cannot run until you
digitize at least one of the multi-spatial-frequency sources.

Everything in the "Verified" table was read from the published paper itself
(open access, figure captions and surrounding text quoted in the notes). Where a
figure number could not be checked against the full text, it says so.

Spatial frequency conversion used throughout: `K = 2*pi*SF` with `SF` in
lines/um, so 1000 lines/mm gives `K = 6.28 rad/um`. The manuscript's design grid
covers `K = 1.96` to `15.71 rad/um`, so a source is most useful when its
spatial frequencies land inside that window.

## Priority 1: Gleeson et al. 2011, Optics Express (open access)

Gleeson, Sheridan, Bruder, Roelle, Berneth, Weiser and Faecke, "Comparison of a
new self developing photopolymer with AA/PVA based photopolymer utilizing the
NPDD model," Opt. Express **19**(27), 26325 to 26342 (2011),
DOI `10.1364/OE.19.026325`. Open access, free PDF from Optica.

This is the best single source for this revision: it is the same NPDD model
family, two material families, several spatial frequencies, and some of the data
is in tables rather than figures, so it needs transcription rather than
digitizing.

| What | Where | x axis | y axis | Condition |
| --- | --- | --- | --- | --- |
| AA/PVA growth curves | **Fig. 3** | exposure time (s) | refractive index modulation n1 | SF = 1428 lines/mm (`K = 8.97`), three intensities: 1, 4, 8 mW/cm2 (red triangles, green squares, blue circles) |
| BMS (Bayfol-HX-type acrylate) growth curves | **Fig. 4** | exposure time (s) | n1 | SF = 1428 lines/mm (`K = 8.97`), same three intensities |
| Spatial frequency response, both media | **Fig. 5** | spatial frequency (lines/mm) | saturated n1 | SF = 500 to 3000 lines/mm (`K = 3.14` to `18.85`), I0 = 8 mW/cm2; AA/PVA red squares, BMS blue dots |
| Saturated n1 per SF, AA/PVA | **Table 6**, column 1 (values in square brackets) | SF | saturated n1 | transcribe, do not digitize |
| Saturated n1 per SF, BMS | **Table 8**, column 1 | SF | saturated n1 | transcribe, do not digitize |

Tables 6 and 8 also publish the authors' own NPDD fits for monomer diffusion
`Dm0` and the non-local response length (mean 61.3 nm for AA/PVA, 9.2 nm for
BMS). Those are independent published estimates of the two parameters this
study could never identify from a single-K curve, so they are a cross-check on
our fitted `D0` and `sigma`, not training data. Read Tables 6 and 8 with care:
the text of Section 4.2.2 refers to "Table 6" when discussing the BMS non-local
value that appears in Table 8, which looks like a slip in the published paper.

What this unlocks: fit on Fig. 3 or Fig. 4 at `K = 8.97`, then predict the
Fig. 5 roll-off at the other spatial frequencies without refitting. That is a
genuine held-out prediction across `K` in the same medium, which is exactly the
gap the current manuscript admits in Section 5 ("`D0` and `sigma` are
identifiable only from multi-K growth-curve families, which we did not have").

Note the geometry caveat before using Fig. 5: the paper's Section 4 records
unslanted transmission gratings, but the sentence introducing Fig. 5 says
"reflection geometries are being examined in both media." Check which geometry
each Fig. 5 series belongs to in the PDF before treating it as a transmission
dataset, and record the answer in the metadata YAML.

## Priority 2: Zanutta et al. 2016, Optical Materials Express (open access)

Zanutta, Orselli, Faecke and Bianco, "Photopolymeric films with highly tunable
refractive index modulation for high precision diffractive optics," Opt. Mater.
Express **6**(1), 252 to 263 (2016), DOI `10.1364/OME.6.000252`. Open access.

| What | Where | x axis | y axis | Condition |
| --- | --- | --- | --- | --- |
| Dn versus writing power density | **Fig. 4** | mean intensity of one writing arm, 0.143 to 21.5 mW/cm2 | refractive index modulation Dn | three line densities: 600, 1000 and 1400 lines/mm (`K = 3.77`, `6.28`, `8.80`), RGB Bayfol HX, 16 um, **total dose held constant at 42 mJ/cm2** |
| Dn versus beam ratio | **Fig. 5** | ratio of the two writing beam powers P1/P2 | Dn | 1000 lines/mm, dose 42.8 mJ/cm2 |

All three of Fig. 4's spatial frequencies sit inside the manuscript's design
grid, in the same material family as the data already in the repository.

Fig. 4 is the sharpest available test of a model assumption we hold fixed. Our
twin uses `gamma = 1`, which makes it exactly dose reciprocal: at constant dose
it predicts Dn independent of intensity, a flat line. Fig. 4 reports a clear
decreasing trend. Digitizing it therefore either calibrates `gamma` away from 1
or, if we keep `gamma = 1` as ground rule 6 requires, becomes a stated and
quantified limitation of the recording model. Either outcome is an honest
Applied Optics result; please digitize it even though it may not flatter the
model.

## Priority 3: Bruder et al. 2017, Polymers (open access, already partly used)

Bruder, Faecke and Roelle, "The Chemistry and Physics of Bayfol HX Film
Holographic Photopolymer," Polymers **9**(10), 472 (2017),
DOI `10.3390/polym9100472`. Open access, CC BY. Fig. 3 of this paper is already
in the repository as `bruder2017_growth_dn_K8.98_{sim,exp}.csv`.

| What | Where | x axis | y axis | Condition |
| --- | --- | --- | --- | --- |
| Dn1 versus grating spacing, Bayfol HX and an acrylamide photopolymer | **Fig. 5** | grating spacing Lambda | Dn1 | experimental symbols for two material families; extracted sigma^2 of 85 nm2 (Bayfol HX) and 3840 nm2 (acrylamide) |
| Two experimental Dn1 points on a calculated curve | **Fig. 4** | Lambda = 1/SF | Dn1 | the two filled symbols only, at SF = 1429 lines/mm (`K = 8.98`) and SF = 4525 lines/mm (`K = 28.43`), Pave = 8 mW/cm2 |
| Mastered power response curve | **Fig. 2** | recording power after calibration | Dn1 | experimental open symbols and simulated filled symbols; **this is a rescaled master curve**, not raw data at one SF, so it needs the shift factors of the paper's Eq. (3) before it means anything. Lowest priority. |

Fig. 5 of Bruder 2017 and Fig. 5 of Gleeson 2011 are two independent published
views of the same comparison, so agreeing with both is a stronger claim than
agreeing with either.

## Also re-digitize what is already here

`bruder2017_growth_dn_K8.98_sim.csv`, `bruder2017_growth_dn_K8.98_exp.csv` and
`hsieh2022_growth_dn_K24.94.csv` were read visually by an AI model, not with a
calibrated digitizer (see `README.md` in this directory, and manuscript
Section 5, which discloses about +/-10% in dose and +/-3 to 5% in Dn1). Both
source images are in `source_figures/`. Re-reading all three with
WebPlotDigitizer and overwriting the CSVs removes the single weakest sentence in
the paper, which an Applied Optics reviewer will go straight to. Keep the old
CSVs under `superseded/` so the change in every downstream number can be shown.

The Hsieh 2022 curve is the source's own model output, not a measurement, and
sits at `K = 24.94 rad/um`, outside the design grid. It stays a secondary check
either way.

## Not usable without a subscription

Gleeson, Liu, McLeod and Sheridan, "Nonlocal photopolymerization kinetics
including multiple termination mechanisms and dark reactions. Part II.
Experimental validation," J. Opt. Soc. Am. B **26**(9), 1746 to 1754 (2009),
DOI `10.1364/JOSAB.26.001746`. Paywalled; Optica shows it as not accessible. The
abstract states it fits diffraction efficiency growth curves, during and after
exposure, for an AA/PVA photopolymer and an epoxy-matrix photopolymer, and the
article has 4 figures and 7 tables. **Which figure holds which growth curve was
not verified**, because the full text could not be read. If you have library
access, open it and tell me the figure numbers and axes; the dark-reaction
(post-exposure) part would be new physics for the twin and is out of scope for
this revision under ground rule 6.

## Metadata YAML required next to each CSV

The Phase 1a harness will load any number of datasets from this directory. Each
`<name>.csv` needs a sibling `<name>.yaml`:

```yaml
source_doi: 10.1364/OE.19.026325
source_citation: "Gleeson et al., Opt. Express 19(27), 26325 (2011)"
figure: "Fig. 3"
series: "I0 = 4 mW/cm2"          # which curve in the figure, if several
medium: AA/PVA                    # must match a file in configs/media/
medium_config: pva_aa_405nm.yaml
observable: dn1                   # dn1 | de
geometry: transmission_unslanted  # transmission_unslanted | reflection | unknown
spatial_frequency_lines_per_mm: 1428
K_rad_per_um: 8.97
exposure_intensity_mW_per_cm2: 4.0
x_quantity: exposure_time         # exposure_time | dose | intensity | beam_ratio
x_units: s                        # s | mJ/cm^2 | mW/cm^2 | dimensionless
y_quantity: index_modulation
y_units: dimensionless
thickness_um: 97                  # layer thickness the source reports
digitized_by: wpd-4.x             # or your name
digitized_date: 2026-10-21
notes: >
  Anything that qualifies the points: log axis, read off a model curve rather
  than a measurement, uncertain geometry.
```

The CSV keeps its current schema (`x,y,source_doi,figure_id,digitized_by,date`)
so the existing fitters keep working. `x` and `y` stay in the source figure's
own units; the YAML is what tells the harness how to interpret them.

## How to digitize

1. Open <https://apps.automeris.io/wpd/> and load the figure image.
2. Calibrate both axes on named gridline values, and set the axis to log where
   the published axis is log (the Bruder dose axis is).
3. Export the points as CSV, then add the four provenance columns.
4. Save as `<firstauthor><year>_<observable>_K<value>[_<series>].csv` so the
   existing filename convention in `fit_literature_curves.py` still infers the
   curve type and `K`.
5. Write the sibling YAML.
6. Tell me which files landed, and I will run the multi-dataset fit, the
   held-out cross-K prediction and the bootstrap.
