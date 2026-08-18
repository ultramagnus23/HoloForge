# Part 2 paper — draft history

The current, live manuscript is [`oe_main.tex`](oe_main.tex) (Optica
universal-template format, real M1/M2/S1/S2 GPU data, single author). This
file summarizes the two earlier draft stages that preceded it, which have
been retired to keep one authoritative paper source instead of three.

## v1 — `part2_media_draft_v1.md` (retired)

Earliest full draft, written before any experiments had been run at scale.
Structured as a Markdown outline with explicit `[RESULT: ...]` /
`[RESULT-FINAL: ...]` placeholder brackets marking every number that still
needed to come from real runs, and `[EXPAND]` markers for sections needing
author prose once results existed. Proposed two-author byline (Chaitanya
Tripathi, Ashoka University + a BITS Pilani co-author) — the final paper
ended up single-author. Introduced the *compensation cliff* hypothesis
(a spatial-frequency threshold beyond which exposure pre-compensation stops
helping) as the paper's headline claim, predicted analytically at
K_c ≈ 4.2–7.0 rad/µm.

## v2 — `part2_media_main.tex` (retired)

First LaTeX draft (IEEE `conference` class, matching Part 1's build
conventions, targeting arXiv). Carried forward the compensation-cliff
framing with CPU-scale preliminary numbers (n_x=256, 100 IMEX steps, 150
Adam iterations, "2 seeds"). Mid-draft, discovered and documented a real bug:
the exposure variable was initialized to exact zeros, so `torch.manual_seed`
had nothing random left to act on — the "2 seeds" were bit-identical runs,
not an independent confirmation. Fixed in `holomedia/optimize.py`; the
single deterministic run itself was not invalidated, but the "matches across
seeds" framing was retracted pending a genuine multi-seed rerun.

## What changed going into `oe_main.tex`

Once the real paper-scale, multi-seed M1/M2/S1/S2 GPU sweep (1086 result
files) actually ran, the data did **not** reproduce the compensation cliff:
gain from media-aware optimization stays positive at every tested K and
every dose budget, with no crossing/collapse point. `oe_main.tex` reports
that finding directly rather than carrying the cliff hypothesis forward as
if confirmed — the abstract, Results section, and figures (F4–F8) were
rewritten from scratch against the real data rather than patched from v2.
The switch from IEEE `conference` class to the Optica universal template
(`optica-article.cls`) happened after this rewrite, once submission target
was confirmed as Optics Express.

See `git log --follow -- paper/part2_media_main.tex paper/part2_media_draft_v1.md`
for the full line-by-line history if a specific earlier claim's exact wording
is needed.
