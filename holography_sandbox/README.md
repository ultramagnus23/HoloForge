# Computational Perceptual Holography Sandbox
### Phase 0 / Project 2 — WaveOptics Degradation Experiments

---

## What this is

A pure-Python simulation framework for exploring how different degradation
parameters affect holographic reconstruction **quality** — both physically
(PSNR / SSIM) and perceptually.

The central question: **"Does this actually look much worse?"**

---

## Structure

```
holography_sandbox/
├── core/
│   ├── waveoptics.py    Angular Spectrum propagation + Gerchberg-Saxton
│   ├── degradation.py   All degradation knobs
│   ├── metrics.py       PSNR, SSIM, LPIPS-proxy
│   └── scenes.py        Synthetic test scenes
├── run_experiments.py   Full sweep runner → results/
├── explore.ipynb        Interactive Jupyter notebook
└── README.md
```

---

## Quick start

```bash
# Install dependencies
pip install numpy scipy matplotlib Pillow scikit-image opencv-python

# Run all sweeps (takes ~2–5 min at SIZE=256)
python run_experiments.py

# Or open the interactive notebook
jupyter notebook explore.ipynb
```

---

## Degradation knobs

| Knob | Function | Values swept |
|------|----------|-------------|
| Resolution | `degrade_resolution(phase, res)` | 256 → 128 → 64 → 32 → 16 |
| Phase bits | `quantise_phase(phase, bits)` | 8 → 4 → 2 → 1 |
| Color | `degrade_color(img, mode)` | RGB → RG → mono |
| Viewing angle | `limit_viewing_angle(phase, frac)` | 100% → 75% → 50% → 25% → 10% |
| Depth planes | (see sweep_depth_planes) | 4 → 3 → 2 → 1 |
| Speckle noise | `add_speckle(recon, sigma)` | 0 → 0.05 → 0.1 → 0.2 → 0.4 |

---

## Physics used

- **Angular Spectrum Method (ASM)** — exact diffraction propagation for
  near-field scenarios. Valid when pixel pitch is comparable to wavelength.
- **Gerchberg-Saxton** — iterative phase retrieval. Alternates between
  SLM plane (phase-only constraint) and image plane (amplitude constraint).
- **Fresnel** — paraxial approximation, faster for large z, less accurate.

Parameters used:
```
λ  = 532 nm   (green laser)
dx = 8 µm     (SLM pixel pitch)
z  = 15 cm    (reconstruction distance)
```

---

## Key insight to look for

Run cell 7 in `explore.ipynb` to see cases where:
- PSNR says "this is bad"
- SSIM says "this is actually fine"

Those divergence points are the core argument of the paper:

> *"Towards Perception-Driven Computational Holography:
>  A Simulation Framework for Exploring Trade-Offs Between
>  Optical Fidelity and Visual Experience"*

---

## Next steps (Phase 0 continuation)

- **Project 3:** Formalise the metrics comparison into a table
- **Project 4:** Head-tracked display (MediaPipe + Three.js)
- **Project 5:** Projection mapping demo (Blender + projector)
- **Phase 1:** Small pilot user study (n=5–10) on degradation threshold detection
