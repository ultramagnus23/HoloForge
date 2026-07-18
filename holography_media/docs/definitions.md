# Pinned definitions (Phase 0.5)

Extracted directly from code (`holomedia/optimize.py`, `npdd.py`,
`diffraction.py`), not from the paper's prose, so the paper's prose can be
checked against these rather than the other way around.

## (a) In-support diffraction efficiency

```python
def diffraction_efficiency(recon, target_mask):
    return (recon * target_mask).sum() / (recon.sum() + 1e-12)
```
(`holomedia/optimize.py`)

**Definition for the paper**: given the reconstructed intensity profile
$I_\text{recon}(x)$ at the observation plane and a binary support mask
$M(x) \in \{0,1\}$ marking the target's nonzero region (in every current
experiment, $M(x) = \mathbb{1}[I_\text{target}(x) > 0.05]$, i.e. thresholded
at 5% of the target's own peak, not an independent aperture), in-support
diffraction efficiency is
$$
\eta_\text{support} = \frac{\sum_x I_\text{recon}(x)\, M(x)}{\sum_x I_\text{recon}(x)}
$$
— the fraction of total reconstructed optical power that lands inside the
target's support region. It is **not** absolute diffraction efficiency
(power out / power in) and **not** a resolution or sharpness metric; it
purely measures how much reconstructed power is being wasted outside where
the target wants it. A method could in principle reach $\eta_\text{support}=1$
with a blurry, wrong-shaped reconstruction as long as all its energy stays
inside the (possibly loose) mask — PSNR is the metric doing the shape-fidelity
work; $\eta_\text{support}$ is a complementary, coarser "is the light going
roughly to the right place" check. The 5% threshold that defines $M(x)$ is a
fixed implementation constant, not swept or justified elsewhere in the code;
worth stating explicitly in the paper rather than leaving as an implicit
default.

## (b) What the current "oracle" is: constrained, not unconstrained

```python
def oracle_ideal(target, recorder, bpm, n_iters=400, lr=5e-2,
                 dose_budget=1.0, seed=0):
    """Upper bound: linear medium optimized AND evaluated as linear."""
    ...
    theta = _seeded_init_theta(recorder.n_x, device, torch.float64, seed)
    ...
    for _ in range(n_iters):
        E = dose_project(softplus(theta) + 1e-6, dose_budget)   # E >= 0, projected
        recon = bpm(c_lin * (E - E.mean()), shrinkage=0.0)      # linear medium, no shrinkage
        ...
```
(`holomedia/optimize.py`)

**The oracle currently in the codebase is the CONSTRAINED oracle** (what the
master prompt's Phase 2 calls M5a): it enforces exactly the same two
constraints as `media_in_the_loop` (M4) — nonnegativity via `softplus(theta)`
and the dose budget via `dose_project` (mean(E) = B) — and uses the same
optimizer (Adam, same `n_iters`/`lr` defaults). The only thing that differs
from M4 is the *medium response*: instead of the full nonlinear NPDD twin
(`recorder(E)`), it uses an idealized linear map $\Delta n = c_\text{lin}
(E - \bar E)$ with $c_\text{lin} = \Delta n_\text{max}$ (zero-mean, no
saturation, no non-local blur, no diffusion transport limit, no shrinkage).

**There is currently no unconstrained oracle (M5b) anywhere in the
codebase.** The paper's existing "oracle" language ("ideal-medium oracle,"
"upper bound") does not currently distinguish these two cases because only
one exists. Phase 2 needs to *add* an M5b (free real-valued exposure/index
optimization with no nonnegativity or dose constraint) if the
constraint-vs-physics decomposition the master prompt asks for is to be
possible — right now the "oracle gap" reported in the paper (3.1–13.0 dB,
per `docs/provenance_report.md`) is entirely attributable to *some
combination* of exposure-domain constraints and medium physics, with no way
to split the two from the current data.

## (c) Recording geometry: 1D-in-x kinetics; z enters ONLY at readout as extrusion, with NO depth-resolved absorption implemented

Checked directly against code, not the paper's prose, which claims more than
the code does (see discrepancy below).

**What the code actually does:**
- `NPDDRecorder` (`npdd.py`) is **strictly 1D in the transverse coordinate
  x**. Its state variables `u`, `N`, `d` are all `(n_x,)` tensors; there is
  no z (depth) index anywhere in the recording kinetics. The IMEX time
  integration produces a single 1D index-modulation profile $\Delta n(x)$ —
  there is no per-depth recording chemistry.
- `SlabBPM.forward` (`diffraction.py`) takes that single 1D `dn_profile(x)`
  and **extrudes it uniformly through the slab thickness**: at each of
  `n_z` depth slices, it applies the *same* $\Delta n(x)$ (optionally
  laterally shifted by the shrinkage/slant model — a lateral shear, not an
  amplitude change) as a phase kick, then propagates. The index modulation's
  *magnitude* does not vary with depth anywhere in the code; only its
  lateral position can, and only if `shrinkage > 0`.
- **Depth (z) enters the model exclusively at the readout stage**, as a
  propagation/extrusion parameter, never inside the recording kinetics
  itself.

**Discrepancy with the paper's prose**: `part2_media_main.tex` (Sec. 3.1 /
Table 1 area) states "Dye depletion couples exposure history to sensitivity
and, extended over depth via Beer–Lambert absorption, produces the
depth-dependence that invalidates thin-element treatment." **No
Beer-Lambert term, no depth-dependent absorption, and no depth-resolved dye
concentration exist anywhere in `holomedia/`** — a repo-wide search for
"Beer," "Lambert," and "absorption" returns zero hits outside this doc.
Dye depletion ($d(x,t)$ in Eq. 3) is real and implemented, but it is a
function of $x$ and $t$ only, exactly like $u$ and $N$ — there is no z in
it. The paper's sentence describes a mechanism that is not implemented.

**Precise statement for the paper** (replacing the current, overclaiming
sentence): *"Recording kinetics (Eqs. 1–3) are solved on the transverse
coordinate x only; the recorded index modulation $\Delta n(x)$ is treated as
uniform through the slab depth at the recording stage. Depth (z) enters
solely at readout, via extrusion of this single profile through the
split-step BPM propagator, with an optional depth-dependent lateral shift
(not an amplitude change) from the shrinkage/slant model (Sec. 3.3). The
model does not currently implement depth-resolved absorption (e.g.
Beer–Lambert attenuation of exposure intensity or dye concentration with
depth); such an extension would require generalizing `NPDDRecorder` to a 2D
(x, z) or (x, y, z) grid, distinct from the existing `npdd3d.py`/
`diffraction3d.py` modules, which generalize the *transverse* dimension
(x → x, y) while keeping the same z-enters-only-at-readout structure."*

This is a real scope correction, not a cosmetic one: it affects how strongly
the paper can claim to model "depth-dependence" as a source of thin-element
invalidity, since that specific depth-dependence mechanism (dye depletion
varying with depth via absorption) is asserted in prose but absent from the
twin.
