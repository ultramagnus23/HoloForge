# Prism form fields

Every value to paste into the Applied Optics submission form, in roughly the
order Prism asks for them. Copy from here rather than retyping.

---

## Journal and article type

- **Journal:** Applied Optics
- **Article type:** Research Article

## Title

```
Media-in-the-loop holography: designing photopolymer exposures through a differentiable recording model
```

## Abstract (plain text)

Paste from `Abstract (plain text).txt` in the built package, which is generated
from the manuscript source with the numeric macros expanded, so it cannot drift
from the PDF. About 100 words.

## Keywords

```
holographic recording materials; photopolymers; volume holography; computer-generated holography; inverse design; differentiable simulation; diffraction gratings
```

## Topic codes: none to enter

**Do not supply OCIS codes.** Optica has retired the scheme. Its author page
states that OCIS codes "should no longer be included on manuscript submissions
to our journals", and that "all articles in the Optica Publishing Group website
are automatically indexed to the Optics and Photonics Topics" instead. Indexing
is automatic, so there is nothing for you to choose or paste here.

If Prism does present a topic or classification field anyway, take whatever the
interface offers and ignore this section; the interface is the authority. An
earlier draft of this file listed five OCIS codes to enter, which was wrong.

## Author

- **Name:** Chaitanya Tripathi (sole author)
- **Affiliation:** Ashoka University, Rajiv Gandhi Education City, Sonipat,
  Haryana 131029, India
- **Corresponding author email:** chaitanya.tripathi_ug2025@ashoka.edu.in
- **ORCID iD:** 0009-0002-4094-0323 (https://orcid.org/0009-0002-4094-0323)

The ORCID is already in the repository, in `CITATION.cff`, on the manuscript
title page and in the cover letter. No placeholder is needed.

## Funding

```
None. This research received no external funding.
```

Optica generates the published Funding section from what you enter in Prism, so
this must match the manuscript's Funding statement exactly.

## Disclosures

```
The author declares no conflicts of interest.
```

## Data availability

```
Code, experiment manifests, raw per-job result files and the scripts that generate every number and figure in this paper are available on GitHub (https://github.com/ultramagnus23/HoloForge) and archived on Zenodo under the concept DOI 10.5281/zenodo.22202762, which resolves to the most recent release.
```

## Previously submitted to an Optica journal?

**YES.**

```
This manuscript was previously submitted to JOSA A as manuscript ID 620231. It was declined on 15 October 2026 by the Editor-in-Chief at the editorial stage, without peer review. The reasons given were that the research area lies outside JOSA A's direct scope, and that JOSA A restricts papers relying on computer-simulated results. No reviews were returned and no scientific objection to the work was raised. The decision letter is available on request. The manuscript has since been substantially revised for Applied Optics: it adds a parameter-sensitivity analysis, a rigorous coupled-wave sweep that identifies the optical mechanism behind the method's failure at nonzero fringe slant, and a new section giving a calibration protocol and design guidelines; the main text is now self-contained without the supplement.
```

Have the decision email saved as a PDF before you start, in case Prism asks to
attach it.

## Preprint

This manuscript has not been posted as a preprint. A companion paper, Part 1,
is on Optica Open at https://doi.org/10.1364/opticaopen.32874356 and is cited
in the manuscript. If you post this manuscript before or during review, declare
it.

## Open access

**Decline.** Optional at $2,300 per article, or $2,550 with a CC BY licence.
Applied Optics has no mandatory publication charge, so declining costs nothing.

## Colour

**Colour online only (free).** Do not select colour in print, which costs $650
for the first figure and $325 for each additional one. All five figures and the
six supplementary figures are drawn with a colourblind-safe palette in which
every series also differs in line style or marker, so they remain readable in
greyscale if a reader prints them.

## Voluntary page charges

**Not paid.** Suggested at $125 per page; explicitly not required for
publication.

## Length

The two-column journal layout is what Applied Optics bills overlength against.
`scripts/pre_submit_check.py` reports the current count; at the time of writing
it is 9 pages, under the 10-page limit, so no overlength charge applies ($300
per page beyond 10). The universal submission template runs longer for the same
body and is not the billed count.

## Suggested reviewers

In `SUGGESTED_REVIEWERS.md`, with each contact verified against an
institutional page. Three names, plus the list of people not to suggest and
why.

## Files to upload

| File | Prism designation |
| --- | --- |
| `Media-in-the-Loop Holography - ... .pdf` | Manuscript |
| `... - Supplement 1.pdf` | Supplemental document, labelled "Supplement 1" |
| `Cover Letter.pdf` | Cover letter |
| `LaTeX source.zip` | Source files, if Prism asks at submission rather than at revision |
