# HoloForge — Perception-Driven Computational Holography

A simulation framework for a single question: **when you degrade a hologram, does it actually look worse — or just measure worse?**

HoloForge models the full pipeline of a phase-only holographic display (Angular-Spectrum propagation + Gerchberg-Saxton phase retrieval), then systematically degrades it along the axes a real SLM-based display is constrained by — spatial resolution, phase-quantization bits, colour channels, viewing angle, depth planes, and speckle — and scores each reconstruction with both physical (PSNR, SSIM) and perceptual-proxy (LPIPS-proxy) metrics. The recurring finding: physical and perceptual quality diverge, and that gap is where display engineering budgets should actually be spent.

> **Part 1 preprint published:** *"Systematic Degradation Analysis in Phase-Only Computational Holography: A Simulation Framework"* — [Optica Open](https://preprints.opticaopen.org/articles/preprint/Systematic_Degradation_Analysis_in_Phase-Only_Computational_Holography_A_Simulation_Framework/32874356?file=66233162).
>
> Part 2 (RGB + human perceptual study): in progress, targeting arXiv (cs.GR / eess.IV).

---

## What's built

- **Wave-optics core** (`holography_sandbox/core/waveoptics.py`) — Angular-Spectrum Method propagation and an iterative **Gerchberg-Saxton** phase-retrieval solver.
- **Degradation suite** (`degradation.py`) — resolution downsampling, phase quantization (8→1 bit), colour reduction, viewing-angle bandwidth limiting, depth-plane reduction, speckle injection.
- **Metrics** (`metrics.py`) — PSNR, SSIM, and an LPIPS-style perceptual proxy.
- **Experiment runner** (`run_experiments.py`) — sweeps every knob, emits side-by-side comparison figures and a consolidated `results/metrics_summary.csv`.
- **Interactive notebook** (`explore.ipynb`) for inspecting individual divergence cases.

Optical parameters: λ = 532 nm (green), 8 µm SLM pixel pitch, 15 cm reconstruction distance, 30 GS iterations.

## In progress / planned (honest status)

- **Gradient-descent phase retrieval (PyTorch autograd)** as a comparison baseline against Gerchberg-Saxton — *not yet implemented; this is the next core experiment.*
- Formalised metrics comparison table and statistical write-up for the preprint.
- Head-tracked display prototype (MediaPipe + Three.js) and a small (n≈5–10) perceptual threshold pilot study.

---

## Selected results (`results/metrics_summary.csv`)

| Degradation | Setting | PSNR (dB) | SSIM | Takeaway |
|---|---|---|---|---|
| Phase bits | 2-bit | 42.1 | 0.93 | Holds up far better than expected |
| Phase bits | 1-bit | 0.05 | 0.003 | Catastrophic — the real cliff edge |
| Viewing angle | 25% bandwidth | 29.5 | 0.97 | Near-lossless down to ¼ aperture |
| Speckle | σ = 0.2 | 41.3 | 0.998 | Perceptually negligible |
| Resolution | 64×64 | 18.8 | 0.019 | Degrades fast |
| Depth planes | 1 plane | 16.8 | 0.009 | Multi-plane depth matters most |

The headline: phase precision and viewing-angle bandwidth are far more forgiving than naive PSNR suggests, while resolution and true multi-plane depth dominate perceived quality.

---

## Quick start

```bash
cd holography_sandbox
pip install numpy scipy matplotlib Pillow scikit-image opencv-python
python run_experiments.py        # full sweep → results/  (~2–5 min at SIZE=256)
# or:
jupyter notebook explore.ipynb
```

## Structure

```
holography_sandbox/
├── core/
│   ├── waveoptics.py    # Angular-Spectrum propagation + Gerchberg-Saxton
│   ├── degradation.py   # all degradation knobs
│   ├── metrics.py       # PSNR, SSIM, LPIPS-proxy
│   └── scenes.py        # synthetic test scenes
├── run_experiments.py   # full sweep runner → results/
├── explore.ipynb        # interactive exploration
└── results/             # generated figures + metrics_summary.csv
```

## Stack

Python · NumPy · SciPy · scikit-image · Matplotlib · OpenCV · Jupyter

## License

Apache License 2.0 — see [LICENSE](LICENSE).
