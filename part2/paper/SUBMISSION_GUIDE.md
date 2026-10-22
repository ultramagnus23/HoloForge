# Applied Optics submission guide

Journal: **Applied Optics** (Optica Publishing Group), submitted through Prism,
<https://prism.optica.org>.
Article type: **Research Article**.
Editor-in-Chief: **Peter E. Andersen**, Technical University of Denmark.

Rebuild everything with `python scripts/make_submission_package.py` (from `part2/code`).
It compiles the manuscript, Supplement 1 and the cover letter from source, fails on any
LaTeX error, undefined reference, `[PENDING]` macro or a two-column journal layout over
10 pages, and writes the files below to `part2/submission/`.

## 1. Files to upload

| File | Prism file designation |
| --- | --- |
| `Media-in-the-Loop Holography - Designing Photopolymer Exposures Through a Differentiable Recording Model.pdf` | Manuscript |
| `... - Supplement 1.pdf` | Supplemental document (label it "Supplement 1") |
| `Cover Letter.pdf` | Cover letter |
| `LaTeX source.zip` | Only if Prism asks for source files (otherwise needed at revision or acceptance) |

Suggested reviewers are in `SUGGESTED_REVIEWERS.md`, which is deliberately not part of
the uploaded package: Prism takes them as form fields, not as a file.

## 2. Prism form fields

Every value you will paste is also collected in `FORM_FIELDS.md` in the built package.

- **Title:** Media-in-the-loop holography: designing photopolymer exposures through a
  differentiable recording model
- **Abstract:** paste `Abstract (plain text).txt` (about 100 words).
- **Author:** Chaitanya Tripathi, Ashoka University, Rajiv Gandhi Education City, Sonipat,
  Haryana 131029, India. Corresponding author email:
  chaitanya.tripathi_ug2025@ashoka.edu.in. ORCID iD: 0009-0002-4094-0323 (https://orcid.org/0009-0002-4094-0323).
- **Funding:** none (the manuscript's Funding section says "This research received no
  external funding"; Optica generates the published Funding section from what you enter
  in Prism, so the two must agree).
- **Previously submitted to an Optica journal?** **YES.** Manuscript ID **620231**,
  submitted to JOSA A, declined 15 October 2026 by the Editor-in-Chief at the editorial
  stage, without peer review, on scope grounds and that journal's limit on papers relying
  on computer-simulated results. No scientific objections were raised and no reviews were
  returned. The cover letter discloses this in full.
- **Preprint:** a companion paper (Part 1) is on Optica Open,
  <https://doi.org/10.1364/opticaopen.32874356>, and is cited. This manuscript itself has
  not been posted as a preprint. If you post it before or during review, declare it.
- **Colour figures:** choose **colour online only (free)**. Colour in print costs $650 for
  the first figure and $325 for each additional one; do not select it.
- **Open access:** **decline.** Optional at $2,300 per article ($2,550 with CC BY).
  Applied Optics has no mandatory publication charge.
- **Voluntary page charges:** **not paid.** These are suggested at $125 per page and are
  explicitly not required for publication.
- **Length:** the two-column journal layout (`paper/manuscript_lengthcheck.tex`) is the
  count Applied Optics bills against. It is currently **9 pages**, under the 10-page
  limit, so there is no overlength charge ($300 per page beyond 10). The universal
  template runs longer for the same body and is not what is billed.
- **Conflicts of interest:** none. **Data availability:** stated in the manuscript.

## 3. Before you click Submit (author-only steps)

1. **Read the manuscript and supplement once, end to end.** You are the sole author and
   take responsibility for every statement.
2. **Save the JOSA A decision email as a PDF.** Prism may ask for the previous decision
   letter alongside the "previously submitted" answer.
3. **Check the AI-use statement** in the Acknowledgment. Optica requires the name, model
   and purpose of any generative AI tool. The statement lists Claude (Anthropic, through
   Claude Code: Sonnet 5, Opus 4.8, Opus 5 and Opus 5.5) for code, analysis scripts,
   reading the digitized literature points, and drafting and editing text. That list
   comes from the repository's commit history. Add any other tool you used.
4. **Archive the submitted code on Zenodo.** The Data availability reference uses the
   concept DOI `10.5281/zenodo.22202762`, which always resolves to the newest release.
   Create the `v2.1.0-applied-optics` release from the merged `main` so Zenodo archives
   the exact code behind the submitted paper. Draft release notes are in
   `RELEASE_NOTES_v2.1.0.md`.
5. **Confirm the affiliation and address** on the title page.

## 4. Why this fits Applied Optics

The journal asks for applications-centered work that moves technology toward the
practical, shows the performance of a technology when theory is applied, and either gives
design guidelines or explains the limits of implementing the theory. The manuscript does
the last two explicitly: Section 6 is a step-by-step calibration protocol with the
minimum measurement set and a ranking of which parameters have to be right, and
Section 4.6 states the geometric boundary of the method and identifies the optical reason
for it. The main text is self-contained; Supplement 1 carries only full derivations and
extra sweeps.

## 5. Questions reviewers are likely to raise

Each of these is already disclosed in the paper:

- Simulation only, with no physical recording (Section 7, first paragraph).
- Why the gain rises at small fringe slant before collapsing. Section 4.6 answers this
  with a closed form, arcsin(K*lambda/(4*pi*n0)), the tilt that Bragg-matches a normally
  illuminated grating, predicted with nothing fitted and confirmed against RCWA.
- The twin is validated against one commercial medium at one spatial frequency, so
  transfer across K and across media is untested (Sections 5 and 7). The calibration
  protocol asks a lab to make exactly the multi-K measurement that would close this.
- The literature curves were read visually, not with a calibrated digitizer (Section 5).
  Re-digitizing `code/data/literature/source_figures/` with WebPlotDigitizer is the
  cheapest available improvement; see `code/data/literature/DIGITIZE_TODO.md`.
- Three seeds measure initialization sensitivity only. Model error is probed by the
  miscalibration and sensitivity studies (Sections 4.7 and 4.8).
- The readout is no longer a soft spot. The split-step beam propagation used inside the
  design loop is now compared with RCWA directly (Section 4.6, Supplement 1 Section S11)
  and agrees to at worst 0.037 in first-order fraction across the whole slant range. What
  restricts the headline results to unslanted gratings is the recording model and the
  benchmark, not the propagator, and the manuscript says so.
- The first-order attribution in Section 4.8 uses 30 draws and is stated as such: it
  ranks the parameters, it is not a Sobol decomposition.
