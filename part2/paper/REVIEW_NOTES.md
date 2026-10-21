# Applied Optics revision: review notes

Working notes for the retarget from JOSA A to Applied Optics. Decisions that are
yours to make, facts verified against primary sources, and anything that could
not be verified.

Branch: `applied-optics`, cut from `orcid-link` (which carries the ORCID links
and `CITATION.cff`, and is still open as PR #13 against `main`). If you would
rather this work sat on `main`, say so and I will rebase.

## Verified external facts

| Fact | Value | Source |
| --- | --- | --- |
| Applied Optics Editor-in-Chief | Peter E. Andersen, Technical University of Denmark | Journal editorial board page, `opg.optica.org/ao/journal/ao/about.cfm`, and the Optica announcement of 29 January 2026 naming him successor to Gisele Bennett |
| Page limit before overlength charges | 10 published pages | Optica publication charges page, `opg.optica.org/submit/review/pub_charge.cfm` |
| Overlength charge | $300 per page beyond 10 | same |
| Voluntary page charge | $125 per page, not required | same |
| Colour | free online, $650 first figure plus $325 each additional in print | same |
| Optional open access | $2,300 per article, $2,550 with CC BY | same |

Applied Optics' limit is therefore identical to the JOSA A limit the manuscript
was already built against, so `scripts/make_lengthcheck.py` and the 10-page gate
in `scripts/make_submission_package.py` keep working unchanged; only their
wording needs updating.

Wikipedia still lists Gisele Bennett as editor of Applied Optics and is out of
date. The cover letter uses the Optica sources.

## Title: not changed, two options for you

Current title, which I recommend keeping:

> Media-in-the-loop holography: designing photopolymer exposures through a
> differentiable recording model

It already names the application and the method and it is the title of the
archived Zenodo release, so changing it costs continuity. Neither alternative is
clearly better, so I have not touched it. If you want a more
application-forward title, these two are the candidates:

1. **Design guidelines for photopolymer exposure in volume holographic
   recording from a differentiable recording model.** Leads with "design
   guidelines," which is literally one of Applied Optics' stated acceptance
   criteria. Loses the "media-in-the-loop" phrase that ties Part 2 to the
   camera-in-the-loop literature it is answering.
2. **Media-in-the-loop exposure design for volume photopolymer holograms: gains,
   parameter sensitivity and operating limits.** Keeps the hook and promises the
   boundary-of-validity content that the new Phase 1 work adds. Longer, and the
   colon-plus-three-nouns shape reads like a review.

## Citations

Not yet audited for this revision. `refs.bib` was verified for the JOSA A
submission (see the correction history in `refs.bib` comments and
`code/docs/parameter_provenance.md`, which record two citations that were wrong
on every field and were fixed). Any reference added during this revision gets
checked here before it reaches the manuscript.

Verified while searching for Phase 1a datasets, and available to cite:

- Gleeson, Sheridan, Bruder, Roelle, Berneth, Weiser, Faecke, "Comparison of a
  new self developing photopolymer with AA/PVA based photopolymer utilizing the
  NPDD model," Opt. Express **19**(27), 26325 to 26342 (2011),
  `10.1364/OE.19.026325`. Open access; full text read.
- Zanutta, Orselli, Faecke, Bianco, "Photopolymeric films with highly tunable
  refractive index modulation for high precision diffractive optics," Opt.
  Mater. Express **6**(1), 252 to 263 (2016), `10.1364/OME.6.000252`. Open
  access; full text read.
- Gleeson, Liu, McLeod, Sheridan, "Nonlocal photopolymerization kinetics
  including multiple termination mechanisms and dark reactions. Part II.
  Experimental validation," J. Opt. Soc. Am. B **26**(9), 1746 to 1754 (2009),
  `10.1364/JOSAB.26.001746`. Metadata verified on the Optica abstract page;
  **full text paywalled, so its figure numbers are unverified.**

## Found while preparing the submission

- **The JOSA A guide's first suggested reviewer had died.** Prof. John T.
  Sheridan, who originated the NPDD model this paper builds on, died on
  1 November 2022 (obituaries from Optica and SPIE; University College Dublin
  runs a memorial bursary in his name). He was listed first in the old
  `SUBMISSION_GUIDE.md`. The entry is removed and the replacement list in
  `SUGGESTED_REVIEWERS.md` has every contact checked against an institutional
  page. Please do not reinstate him.

- **Only 6 of 28 bibliography entries carry a DOI or URL field.** This is not a
  submission blocker: the Optica bibliography style does not print DOI fields
  and Optica resolves them at production, and the entries were verified by
  author, title, venue and year for the earlier submission. It is still worth
  closing, and `scripts/pre_submit_check.py` reports the list every run so it
  stays visible. Adding them means verifying 22 DOIs one at a time; I have not
  done it, and will not add a DOI I have not resolved.

- **The split-step beam propagation inside the design loop has still never been
  compared with RCWA directly.** The new slant sweep bounds the closed-form
  Kogelnik reference only, which is what the committed cross-check also does.
  A direct comparison would need first-order efficiencies extracted from the
  beam-propagation field at matched incidence, which is a self-contained piece
  of work and the strongest remaining answer to a reviewer who doubts the
  readout.

## Open items that are yours, not mine

1. **Digitize at least one multi-spatial-frequency dataset.** Exact
   specifications, with figure numbers and axes, are in
   `code/data/literature/DIGITIZE_TODO.md`. Phase 1a's cross-dataset held-out
   validation, which is the direct answer to the "simulation-only" objection, is
   blocked until this lands.
2. **Save the JOSA A rejection email as a PDF.** The cover letter discloses the
   earlier submission; Prism may ask for the decision letter.
3. **ORCID** is already in the repository (`0009-0002-4094-0323`, in
   `CITATION.cff`, the manuscript title page and the cover letter), so no
   placeholder is needed in the Prism form fields.

## Risks that could still draw a desk rejection

Recorded here as they are found, and summarised in the final report.

- The paper remains simulation-only. Applied Optics does not carry JOSA A's
  explicit restriction on simulation-based papers, but "incremental or
  incomplete work is rejected" can be read the same way if the validation
  against measured data stays as thin as it is today: one commercial medium, one
  spatial frequency, points read by eye rather than with a digitizer, and one of
  the two fitted series is itself a model output. Item 1 of the open list above
  is the main mitigation.
