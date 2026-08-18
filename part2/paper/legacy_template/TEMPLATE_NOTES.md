# Optics Express / Optica template — status

Fetched via direct download (not AI-paraphrased) from the public CTAN mirror:
`opex3.sty`, `OpEx_temp.tex`, `OpEx_style.tex`. These are the **legacy**
(2003-era, OSA-branded) Optics Express template files -- real, usable
LaTeX, not fabricated.

## Confirmed from Optica's current (live, 2026) author guidelines

- **Abstract limit: ~100 words**, stand-alone (states problem, methods,
  results, conclusions -- not framed as an introduction). This matches
  what `oe_main.tex`'s abstract TODO already assumed; no change needed
  there.
- A **data availability statement** is required in the submission (Prism
  form), separate from the in-text "Code and Data Availability" section
  already present in `oe_main.tex`.

## Open question, not resolved here

Optica has since introduced a **"universal manuscript template"**
(current, hosted on Overleaf, submitted directly to Prism) that
supersedes per-journal legacy files for several Optica journals. The
journal lists I found naming which journals it covers did **not**
consistently include Optics Express by name (one listed AO/JOCN/JOSA A/
JOSA B/OL/Optica; Optics Express was absent from both lists I found).
Whether Optics Express:
  (a) now uses the universal template, or
  (b) still uses `opex3.sty`/the legacy format,
was not confirmed with certainty from what's publicly fetchable here.

**Recommendation**: before final formatting, open the universal template
on Overleaf (https://www.overleaf.com/latex/templates/universal-manuscript-template-for-optica-publishing-group-journals/ybkgndgdxpzy)
and check its journal-selection dropdown/instructions for Optics Express
explicitly, or start a submission in Prism (https://prism.optica.org) and
see which template it hands you -- that is authoritative in a way a
web search summary is not. Don't guess further from here.

## Files in this directory

- `opex3.sty`, `OpEx_temp.tex`, `OpEx_style.tex` -- legacy template,
  fetched 2026-08-11, kept as a working fallback and for reference on
  section/reference macros (`\ocis`, `\OEtitle`, journal abbreviations
  like `\josab`) even if the final submission ends up using the newer
  universal template instead.
