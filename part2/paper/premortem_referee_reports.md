# Pre-mortem: anticipated referee reports

WP11 (Applied Optics revision). Two referee personas, each grounded in a
real, already-disclosed limitation of the manuscript (Discussion and
Limitations, Section 6/Twin Validation, and the per-section "referee
residual" notes accumulated across WP1-WP9) — not invented objections.
Each major/minor point below names the exact section a response would
point to, and whether the current draft already answers it or needs a
targeted addition before submission.

---

## Referee 1: skeptical of simulation-only scope and twin fidelity

**R1.1 (major).** "The entire study is in simulation. Without a single
physical recording-and-readout experiment, how do we know any of this
transfers to a real photopolymer?"

*Where the draft already answers this:* Section 6 (Twin Validation)
fits the twin against two independent, in-regime digitized literature
curves (Bayfol HX), reaching NRMSE 0.17–0.29 after correcting a real
citation-scoping error found during that fit. Section 6's own closing
paragraph states plainly that this is mechanism-level validation, not
quantitative agreement, and that the paper's claims are paired
comparisons (media-in-the-loop minus media-blind SGD, same twin for
both arms) specifically *because* that structure cancels a systematic
twin miscalibration — the response should lead with that paired-design
argument, not just the fit quality.

*Residual gap, not fully closed:* only one medium family (Bayfol HX) is
confirmed in-regime; the two primary NPDD growth-curve sources remain
paywalled (stated in Section 6's last paragraph). A response should
state this as a scope limitation accepted going into the revision, not
argue it away — the manuscript already does this correctly and should
not be softened under review pressure.

**R1.2 (major).** "The recording model treats the medium as uniform
through depth. Real photopolymer films absorb the recording beam — this
seems like it could invalidate the whole comparison."

*Where the draft already answers this:* this is no longer a purely
parametric concession. Section 6 (Discussion) now reports a real,
depth-resolved Beer–Lambert extension (WP6) evaluated at the same
exposures used throughout, at optical densities 0, 0.1, and 0.3 (the
paper's own stated "good approximation" and "not a small perturbation"
boundaries): paired gain narrows by roughly 13% across that range but
never closes. This is the single strongest available answer to R1.2 and
should be foregrounded, not buried in a limitations paragraph.

**R1.3 (minor).** "Kogelnik theory and RCWA disagree badly for the
slanted geometry this paper actually uses (Figure with the RCWA
validity envelope) — doesn't that undermine confidence in the readout
model?"

*Where the draft already answers this:* the Discussion section states
directly that this divergence is *why* BPM, not Kogelnik, is the model
that sits inside every optimization loop — Kogelnik is only ever used as
closed-form validation context. A response should re-emphasize that
BPM's own validity is bounded by the Klein–Cook regime discussion in
Section 3.4 (readout model), which is a separate, already-addressed
argument from the Kogelnik/RCWA comparison.

**R1.4 (minor).** "The physics-component ablation removes one NPDD
mechanism at a time. Doesn't this miss interactions between mechanisms?"

*Where the draft already answers this:* stated explicitly as a
limitation in the Discussion ("one-at-a-time and does not capture
interactions"). No further defense is available or warranted here —
this should be accepted as-is; claiming otherwise would overstate the
ablation's scope.

---

## Referee 2: skeptical of statistical rigor and effect-size reporting

**R2.1 (major).** "Three seeds is a very small sample for confidence
intervals. How much of the headline result is seed noise?"

*Where the draft already answers this:* the manuscript reports
t-distribution 95% CIs at n=3 throughout (correctly wide, not
overstated), and the WP4 revision added nonparametric bootstrap CIs
alongside them for the target-ensemble and joint-miscalibration checks
specifically so a referee could see the two methods agree in order of
magnitude despite neither being individually trustworthy at this sample
size. A response should point to both explicitly and note the seed
count was a deliberate compute-budget reduction from an earlier 5-seed
design (documented in the commit history), not an oversight.

**R2.2 (major).** "Some reported gains are close to zero (e.g., the
minimum gain across the whole M1 grid is essentially 0 dB at one
budget). Is the headline result actually robust, or does it wash out at
the margin?"

*Where the draft already answers this:* the failure-rate framing
(Section 5, "What write-once recording actually pays for") exists
specifically because mean-PSNR framing invites exactly this objection —
it reports the *fraction of exposures that fail outright* rather than
an average that a few near-zero points can pull down without changing
the practical picture. A response should lead with the failure-rate
table (now covering all seven baselines, not just BSGD/MIL) rather than
re-litigating the mean-gain number.

**R2.3 (major).** "The joint-miscalibration Monte Carlo only draws 30
samples. Is that enough to claim the sign never flips?"

*Where the draft already answers this:* Section 5 (Joint
miscalibration) reports the exact count (30 draws), the worst single
draw's gain (still positive), and a bootstrap CI over draws — the claim
is stated as "0 of 30 draws negative," not "provably never negative."
A response should not overstate this further; if pressed, the honest
answer is that 30 is a bounded, disclosed sample, not an exhaustive
proof, and the manuscript already frames it that way.

**R2.4 (minor).** "The regularized-SGD and gamma-precompensation
baselines (RSGD, GPC) show gains indistinguishable from zero at two of
three budgets, and GPC is sometimes numerically identical to a simpler
baseline (LPC). Why include them?"

*Where the draft already answers this:* Section 5.2 (Baseline
completeness) reports both findings as real, negative results — RSGD's
tuned regularization weight is exactly zero at two budgets (so it
*is* BSGD there, not a failed attempt to differ), and GPC's
near-coincidence with LPC is explained mechanistically (both hit the
same contrast-clipping ceiling on binary targets) rather than hidden. A
response should state plainly that these are two straw-man objections
tested and closed, and that a null result reported honestly is not a
weaker finding than a fabricated positive one would have been.

**R2.5 (minor).** "The 2D study only uses three synthetic target images.
Is that a systematic enough test?"

*Where the draft already answers this:* Section 5 (2D generalization)
states directly that this is "a bounded check, not a repetition of the
1D study's statistical design." No further defense is warranted; this
should be accepted as a disclosed scope limit, same as R1.4.

---

## What this pre-mortem changes about the submission, if anything

Every major point above already has a real answer somewhere in the
current draft — this pass did not surface an objection the manuscript
cannot currently address. The action items are about **emphasis**, not
new content:

1. The cover letter (see `cover_letter.md`) should lead with the
   paired-comparison argument (R1.1) and the failure-rate framing
   (R2.2), since those are the two arguments doing the most defensive
   work across the largest number of anticipated objections.
2. If a real reviewer raises R1.2 (depth-resolved absorption) or R2.3
   (joint miscalibration sample size), the response letter should cite
   the specific section and macro-backed numbers directly rather than
   re-deriving them — both already exist.
3. No revision to the manuscript text itself is indicated by this
   pass. Where a limitation has no full defense (R1.1's residual gap,
   R1.4, R2.5), the correct response is to accept it plainly, not
   argue further — which is also what the current draft already does.
