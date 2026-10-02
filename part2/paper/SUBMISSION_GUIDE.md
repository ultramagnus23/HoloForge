# JOSA A submission guide

Journal: **Journal of the Optical Society of America A** (Optica Publishing Group),
submitted through Prism, <https://prism.optica.org>.
Article type: **Research Article**.

Rebuild everything with `python scripts/make_submission_package.py` (from `part2/code`).
It compiles the manuscript, Supplement 1 and the cover letter from source, fails on any
LaTeX error, undefined reference, `[PENDING]` macro or a manuscript over 10 pages, and
writes the files below to `part2/submission/`.

## 1. Files to upload

| File | Prism file designation |
| --- | --- |
| `Media-in-the-Loop Holography - Designing Photopolymer Exposures Through a Differentiable Recording Model.pdf` | Manuscript |
| `... - Supplement 1.pdf` | Supplemental document (label it "Supplement 1") |
| `Cover Letter.pdf` | Cover letter |
| `LaTeX source.zip` | Only if Prism asks for source files (otherwise needed at revision or acceptance) |

## 2. Prism form fields

- **Title:** Media-in-the-loop holography: designing photopolymer exposures through a
  differentiable recording model
- **Abstract:** paste `Abstract (plain text).txt` (about 100 words, as JOSA A asks).
- **Author:** Chaitanya Tripathi, Ashoka University, Rajiv Gandhi Education City, Sonipat,
  Haryana 131029, India. Corresponding author email:
  chaitanya.tripathi_ug2025@ashoka.edu.in. Add your ORCID iD if you have one.
- **Funding:** none (the manuscript's Funding section says "This research received no
  external funding"; Optica generates the published Funding section from what you enter
  in Prism, so the two must agree).
- **Previously submitted to an Optica journal?** No, unless this manuscript was submitted
  before. If it was, say so and attach the earlier decision and a response.
- **Preprint:** a companion paper (Part 1) is on Optica Open,
  <https://doi.org/10.1364/opticaopen.32874356>, and is cited. This manuscript itself has
  not been posted as a preprint. If you post it before or during review, declare it.
- **Colour figures:** choose colour online only (free). Colour in print costs $650 for
  the first figure and $325 for each additional one.
- **Open access:** optional ($2,300). JOSA A has no mandatory publication charge.
- **Length:** 10 pages in the universal template and 7 pages in the two-column journal
  layout (`paper/manuscript_lengthcheck.tex`), under the 10-page limit, so no overlength
  charge ($300 per page beyond 10).
- **Conflicts of interest:** none. **Data availability:** stated in the manuscript.

## 3. Suggested reviewers

Prism asks for three, with email addresses. Take the current address from each person's
institutional page. None of them is a collaborator of the author.

| Name | Affiliation | Why |
| --- | --- | --- |
| Prof. John T. Sheridan | University College Dublin, Ireland | Originator of the NPDD recording model the paper builds on |
| Prof. Izabela Naydenova | Technological University Dublin, Ireland | Holographic photopolymer materials and recording |
| Prof. Robert R. McLeod | University of Colorado Boulder, USA | Holographic photopolymer design and modelling |
| Prof. Yifan (Evan) Peng | The University of Hong Kong | Camera-in-the-loop neural holography |

Do not suggest Wechsler, Rizzo or Moser (EPFL): their concurrent work is discussed in the
paper, so they may have a competing interest.

## 4. Before you click Submit (author-only steps)

1. **Read the manuscript and supplement once, end to end.** You are the sole author and
   take responsibility for every statement.
2. **Check the AI-use statement** in the Acknowledgment. Optica requires the name, model and
   purpose of any generative AI tool. The statement lists Claude (Anthropic, through
   Claude Code: Sonnet 5, Opus 4.8, Opus 5 and Opus 5.5) for code, analysis scripts,
   reading the digitized literature points, and drafting and editing text. That list
   comes from the repository's commit history. Add any other tool you used.
3. **Archive the submitted code on Zenodo.** The Data availability reference uses the
   concept DOI `10.5281/zenodo.22202762`, which always resolves to the newest release.
   Create a GitHub release (for example `v2.0.0-josaa-submission`) from the merged `main`
   so Zenodo archives the exact code behind the submitted paper.
4. **Confirm the affiliation and address** on the title page.

## 5. Questions reviewers are likely to raise

Each of these is already disclosed in the paper:

- Simulation only, with no physical recording (Section 6, Scope).
- The twin is validated against one commercial medium at one spatial frequency, so
  transfer across K and across media is untested (Section 5).
- Three seeds measure initialization sensitivity only. Model error is probed by the
  miscalibration studies (Section 4.7).
- The scalar readout is restricted to unslanted gratings (Sections 4.6 and 6).
- The literature curves were read visually, not with a calibrated digitizer (Section 5).
  Re-digitizing `code/data/literature/source_figures/` with WebPlotDigitizer would be a
  cheap improvement during revision.
