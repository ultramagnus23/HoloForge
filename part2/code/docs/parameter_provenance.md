# NPDD parameter provenance audit

Phase A.1 of the Gate-A work: every parameter in every `configs/media/*.yaml`
file, its cited value, its actual source, and the exact condition that
source describes. Triggered by finding one mis-scoped citation
(`bayfol_hx_405nm.yaml`'s `dn_max`); this document is the "audit everything
else once" follow-through, not a spot-fix.

## Method

For each parameter: re-fetch the cited source, quote the specific sentence/
table/figure it comes from, and check whether that sentence/table/figure
describes the same experimental condition the parameter is actually used
for in this codebase (same medium variant, same recording geometry, same
spatial frequency regime). A citation can be a real paper, correctly
quoted, and still be wrong for our purposes if it was scoped to a
different condition than the one we're fitting against -- that is exactly
what happened with Bayfol's `dn_max`.

## bayfol_hx_405nm.yaml (Bruder, Fäcke & Rölle, *Polymers* 9(10):472, 2017,
DOI 10.3390/polym9100472)

| Param | Value | Source | Condition it describes | Status |
|---|---|---|---|---|
| `D0` | 0.025 | "D_ex = 2.5e-10 cm²/s" | Not condition-scoped beyond "Bruder et al. 2017" in the existing comment | **Not re-audited to a specific figure/table this pass** -- diagnostic fits (see below) held D0 fixed at this value and got reasonable curve-shape agreement once `dn_max` was corrected, which is indirect support but not a direct citation check |
| `sigma` | 0.0092 | "sigma² = 85 nm²" | Same as D0 -- not condition-scoped | Same caveat |
| `dn_max` | 6.0e-3 (was; see below) | Table 3, "maximum refractive index modulation as obtained in holography recording" | **Table 3 is a separate dye/borate-screening experiment (paper Section 4.4)**, confirmed verbatim from the paper's own text: *"Table 3's modulation values come from separate dye-screening holographic experiments... not from the Figure 3 dosage-response curves."* Figure 3 (the curve actually fit in Sec. 6, Λ=700nm, SF=1429 l/mm, reflection-type) is a **different condition** with no independently-stated dn_max in the text -- its own digitized peak (~0.016) is the only available estimate. | **RESOLVED as mis-scoped.** `dn_max` is now a free fit parameter for growth_dn curves (`fit_literature_curves.py`'s `SECOND_PARAM_BY_CURVE_TYPE`) rather than fixed at the Table-3 value, which does not describe the condition being validated against. |
| `shrinkage` | 0.025 | "below 3%, depending on composition" | General statement, not figure-specific | Low precision but not mis-scoped -- the cited range genuinely is a general material statement |
| `thickness` | 50.0 µm | "typical reflection-recording grade" | General statement | Same as shrinkage -- reasonable, not condition-pinned |

## pq_pmma_405nm.yaml (Hsieh, Cheng & Chung, *ACS Omega* 7(14):11770-11776,
2022, DOI 10.1021/acsomega.1c06887)

| Param | Value | Source | Condition | Status |
|---|---|---|---|---|
| `D0` | 1.24e-6 | "D_PQ = 1.24e-18 m²/s" | Not condition-scoped to Figure 2b specifically | Not re-audited this pass |
| `dn_max` | 3.0e-4 | "1.5e-4 (two-step thermal) to 3.6e-4 (solvent-cast)" | Bulk material characterization, not Figure 2b's specific simulated growth curve | **Flagged, unresolved, for a different reason than Bayfol's case.** A 2-parameter {kappa, dn_max} diagnostic fit against Figure 2b's digitized curve (K=24.94 rad/µm, out of this paper's tested range -- see below) converged to `dn_max≈0.018`, roughly 50-120x the cited 1.5e-4-3.6e-4 range. This is **not** read as evidence the cited range is wrong the way Bayfol's was: Figure 2b's curve is **still rising, not yet saturated**, at t=2000s (the full extent of the paper's own plotted data). A still-rising curve does not constrain `dn_max` at all -- the optimizer can place it anywhere "sufficiently large" and compensate with kappa/D0 to match the visible rising slope. `dn_max=0.018` from this fit should **not** be reported as a measurement of PQ/PMMA's saturation ceiling; it is an artifact of extrapolating past what the data shows. |
| `sigma` | 0.08 | dataclass default, explicitly flagged as unconfirmed in the file's own comment | n/a | Already honestly flagged before this audit -- no change |
| `shrinkage` | 0.01 | Already flagged unconfirmed | n/a | No change |
| `thickness` | 120 µm | "120 µm solvent-cast film" | Matches the general PQ/PMMA characterization, not necessarily Figure 2b's specific sample | Not re-audited to Figure 2b specifically |

## pva_aa_405nm.yaml (composite, multiple sources)

Two citation errors found and fixed this pass, both cases where the
underlying paper had already been corrected in `paper/refs.bib` (during
earlier bibliography verification) but this config file's comments were
never updated to match -- the same class of drift as the Bayfol/Fomenko
and PQ-PMMA/Jeong fixes, just missed in this file specifically because
nothing had triggered a full re-read of it until this audit:

| Param | Was cited as | Actually is | Fixed to |
|---|---|---|---|
| `D0` | "Kelly & Sheridan reply, JOSA B 28(4):658 (2011)" | That volume/issue/page is Close, Gleeson & Sheridan's **original** paper, not a reply, and no author named Kelly appears on either paper in this series | Sheridan, Gleeson & Close, JOSA B 29(2) (2012) -- `refs.bib` key `sheridan2012reply` (the actual reply) |
| `sigma` | "Gleeson/Guo, JOSA B 25(3):396, 2008" | Author list incomplete/wrong (paper has 6 authors: Gleeson, Sabol, Liu, Close, Kelly, Sheridan; no "Guo") | Gleeson, Sabol, Liu, Close, Kelly & Sheridan, JOSA B 25(3):396 (2008) -- `refs.bib` key `gleeson2008chain` |
| `shrinkage` | "Gallego et al., Appl. Phys. A, PubMed 21747495" | That PubMed ID resolves to Moothanchery, Naydenova & Toal, *Optics Express* 19(14):13395-13404 (2011) -- different authors, journal, and year. No matching Gallego/Applied Physics A/2008 paper was found (this is the same dead citation removed from `refs.bib` as `gallego2008shrinkage` earlier) | **Source removed, not replaced.** The 1-1.9% value is now marked unsourced in the config comment -- a placeholder consistent with typical AA/PVA magnitudes, not a citable number, until re-derived. |

`kappa` and `dn_max`/`thickness` in this file were already honestly marked
as calibration targets / order-of-magnitude illustrative points (not
literature-exact) before this audit -- no change needed.

## What this means for Section 6 and the K=24.94 rolloff conclusion

Per the Gate-A ground rule (A.5): the PQ/PMMA fit's poor `dn_max`
identifiability, combined with K=24.94 rad/µm sitting 1.6x above this
paper's own tested grid (1.96-15.71 rad/µm), means **neither the fit
quality nor the fitted parameters from the Hsieh source should be used as
evidence about model structure inside the tested K range.** It stays in
Section 6 as an out-of-range check with its own caveats, not as a second
independent confirmation of the Bayfol finding.

## Observable/condition audit (Gate A.3) -- one open item

`simulate_growth_dn` hardcodes fringe visibility at 0.9 (`exposure = 1.0 +
0.9*cos(Kx)`) for every growth_dn fit, regardless of source. Neither
Bruder et al. 2017 nor Hsieh et al. 2022 was checked for a stated
two-beam intensity ratio / fringe visibility for the specific recording
condition being fit -- this is a real, unaudited assumption, not
verified against either source. It was not identified as the cause of
the original poor fits (the `dn_max` citation-scoping error fully
explains the improvement seen), but it hasn't been ruled out as a
contributor to the residual NRMSE (0.17-0.29) or the 2.7x `dn_max`
disagreement between the two Bayfol series. Everything else on the A.3
checklist was checked: observable identity (Δn₁ on both sides, confirmed
against the source figure's own axis label), x-axis units (mJ/cm² used
as digitized, no mJ/J conversion error), and the readout path (growth_dn
correctly bypasses Kogelnik entirely, since the fitted quantity is Δn₁
itself, not diffraction efficiency -- so the "check the Kogelnik sin²
path" hypothesis for the original poor fits does not apply to this data,
verified by confirming the y-axis is literally labeled Δn₁ in the source
figure).

## Multi-start fit results

See `results_literature_fit.json` (`n_starts`, `nrmse_min`, `nrmse_max`,
`nrmse_std` per source) for the actual multi-start (n=10) fit spread under
the corrected {kappa, dn_max} parameterization -- generated by
`experiments/fit_literature_curves.py`, not hand-computed.
