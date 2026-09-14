Dear Editor,

We submit "Media-in-the-Loop Holography: Computer-Generated Holography
Through a Differentiable Model of Volume Photopolymer Recording" for
consideration as a Research Article in Applied Optics.

**What the paper does.** Camera-in-the-loop correction — the standard
fix for hologram-recording error in dynamic or rewritable media — is
structurally unavailable for write-once volume photopolymer recording:
the reconstruction error becomes measurable only once it is permanent.
We replace it with a differentiable digital twin of the underlying
non-local polymerization-driven diffusion (NPDD) recording chemistry,
placed inside the hologram-design optimization loop itself, so the
optimizer sees the medium's actual nonlinear, saturating response
during design rather than only at an uncorrectable readout.

**Why we report the result the way we do.** A mean PSNR gain is the
wrong unit for a write-once medium: what it actually charges for is not
average quality but ruined material. We therefore lead with a
failure-rate framing — the fraction of exposures that fall unacceptably
short of an achievability ceiling — because that is the quantity a
practitioner is billed in, and report it across every baseline method we
test, not only our own.

**What we found.** Against media-blind optimization, the twin-aware
approach cuts this failure rate substantially, and the advantage
survives every robustness check we ran: two-dimensional, non-separable
targets; joint (not just single-parameter) miscalibration of the
recording twin; a disclosed detector-noise model at the readout stage;
and a genuinely depth-resolved (Beer–Lambert) recording extension in
place of the uniform-through-depth assumption used elsewhere in the
literature. We also test, rather than assume, whether a much cheaper
saturation-only surrogate model can substitute for the full recording
physics: it recovers a substantial but incomplete fraction of the
advantage, concentrated at low spatial frequency, which we believe is
itself a useful practical finding for anyone deciding whether the full
model's cost is justified for their own application.

**On scope.** This is a simulation study, validated against independent
published photopolymer response data (with a citation-scoping error we
found and corrected during that process reported directly, not
smoothed over) but not against a new physical recording experiment;
every claim in the paper is stated at the level of evidence it actually
has, including where that evidence is bounded — a single confirmed
in-regime medium, a one-parameter-at-a-time physics ablation, a bounded
rather than exhaustive two-dimensional study. We believe this level of
honesty about scope, backed throughout by a fully reproducible,
script-generated numbers pipeline (no hand-transcribed figures anywhere
in the manuscript), is itself part of what we are offering the
reviewers to evaluate.

We believe this work will interest Applied Optics's readership working
on holographic data storage, computer-generated holography, and
differentiable modeling of nonlinear recording media, and we welcome
the reviewers' scrutiny of where its scope should be extended further.

All source code, experiment manifests, and raw result data are publicly
available (see the manuscript's Data availability statement) under the
Apache 2.0 license, with a versioned snapshot archived at Zenodo.

Thank you for considering our manuscript.

Sincerely,
Chaitanya Tripathi
Ashoka University, Sonipat, India
