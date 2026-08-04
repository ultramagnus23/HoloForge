# Legacy result-set audit (pre-restructure, execution spec Sec. "before restructuring")

Every script that has ever produced or is meant to produce a results file,
audited before the V1-V3/M1-M3/S1-S2 restructure. **Nothing has been
deleted or modified.** Commits found via `git log --follow` on each result
file, not asserted from memory.

Legend for **Provenance (new schema)**: none of these embed a git commit
hash / manifest name / job ID *inside* the JSON itself (that schema field
didn't exist until `run_manifest.py`, built after all of these). "External"
means resolvable via `git log --follow` (as done here); "None" means the
result was never committed at all.

| # | Script | Result file | Commit (latest) | dtype | Device | Seeds | Provenance | Regenerable under new pipeline |
|---|---|---|---|---|---|---|---|---|
| 1 | `run_prelim.py` | `results_prelim.json` | `1619df6` (2026-07-13) | float64 (hardcoded) | CPU | `[0,1]` **pre seed-fix (61a1563) — bugged: both "seeds" are bit-identical** | External only | Yes — maps to M1-M3 (cliff sweep), superseded by manifest pipeline |
| 2 | `run_prelim2.py` | `results_prelim2.json` | `1619df6` (2026-07-13) | float64 (hardcoded) | CPU | `[0,1]` **pre seed-fix — bugged, same as above** | External only | Yes — maps to M1-M3 (sigma/shrinkage sweeps) |
| 3 | `run_confirm.py` | `results_confirm.json` | `5c4c141` (2026-07-17) | float64 (hardcoded) | CPU | `[0,1,2]` **pre seed-fix — bugged** | External only | Yes — maps to M1-M3, confirmation-scale |
| 4 | `f1_validate_twin.py` | `results_f1.pt` | **N/A — never committed** (`.gitignore` excludes `*.pt`) | float64 | CPU | n/a (deterministic growth curves) | **None** | Yes — this is exactly V1 (NPDD vs. published data), but literature overlay was never digitized either (see `data/literature/README.md`) |
| 5 | `f2_panel.py` | none (prints only; feeds `figures/figD_panel.png`, which IS tracked) | `f56bfe8` (fig only) | float64 | CPU | seed=0 default, single run | **None** (numbers only ever existed in stdout/my transcript) | Superseded — was a qualitative demo, not claimed as a result |
| 6 | `f2_f3_recovery.py` | `results_f2f3.json` | **N/A — never run to completion / never committed** | float64 (hardcoded) | CPU | `[0,1,2]` (comment: "bump to 5 for the paper") | **None** | Yes — this IS the M1-M3 CPU-scale precursor |
| 7 | `gpu_f2_f3_recovery.py` | `results/gpu_reruns/f2_f3_recovery/` | **N/A — never run, no GPU available when written** | float64 (hardcoded, batched) | CUDA (target) | `[0..4]` | **None** | Superseded directly by the new M1-M3 manifest (this script predates the manifest system entirely) |
| 8 | `gpu_npdd_mesh_convergence_sweep.py` | `results/gpu_reruns/npdd_mesh_sweep/results.json` | `8d63c5e` (2026-07-18) | float64 (hardcoded) | **CUDA, real Colab T4 run** | `seed=0` only, single-seed by design | External only | Not directly (mesh-convergence isn't in the new tier structure) — becomes S-tier supplementary evidence, or dropped |
| 9 | `gpu_bpm_wavelength_sweep.py` | `results/gpu_reruns/bpm_wavelength_sweep/results.json` | `8d63c5e` (2026-07-18) | float64 (hardcoded) | **CUDA, real Colab T4 run** | `seed=0` only, single-seed by design | External only | Not directly — supplementary; not part of V/M/S as scoped by you |
| 10 | `ablation_gradients.py` | `results_ablation_gradients.json` | `f56bfe8` (2026-07-16) | float64 (hardcoded) | CPU | n/a — compares gradient *pathways* on one fixed trajectory by design, not a multi-seed claim; seed-fix doesn't apply here | External only | Maps to S1 (component ablation) — but note S1 in your spec is "ablation of media-model components" (physics terms), while this is "ablation of *gradient computation methods*" (engineering) — **different thing, same name risk, flagging now** |
| 11a | `rcwa_crosscheck.py` (3-case) | `results_rcwa.json` | `f56bfe8` (2026-07-16) | **float64, hardcoded unconditionally** (independent of the new float32 policy — see `docs/precision_policy.md`) | CPU (torcwa) | n/a — deterministic | External only | Maps to V3 (RCWA cross-check) directly |
| 11b | `rcwa_crosscheck.py e7` | `results_rcwa_e7.json` | `e98f248` (2026-07-19, **post seed-fix**, though irrelevant here) | float64, hardcoded | CPU (torcwa) | n/a — deterministic | External only | Maps to V2/V3 (regime map + cross-check) directly — this is real, already-run data ready to feed the new structure as-is |
| 12 | `showcase_3d.py` | `results_3d_showcase.json` | `f56bfe8` (2026-07-16) | float64 (hardcoded) | CPU | `seed=0` default, single run. **Has its OWN unfixed copy of the zero-init seed bug** (`media_in_the_loop_3d`'s local `torch.manual_seed`/zero-init, never patched when `holomedia/optimize.py` was fixed) — dormant only because it's single-seed and nothing multi-seed is claimed from it | External only | Out of scope for V/M/S as you've defined them (2D x,z is the paper's main geometry) — leave as supplement, flag the dormant bug if anyone ever seeds-sweeps it |
| 13 | `fit_literature_curves.py` | `results_literature_fit.json` | **N/A — never run for real** (no digitized CSVs exist; running it against the empty `data/literature/` prints "nothing to fit" and writes nothing) | float64 (hardcoded) | CPU | n/a | **None** | Maps to V1, blocked on you digitizing literature curves (unchanged blocker, not new) |

## What must be re-run to meet the provenance requirement (Sec. 0.2)

**Scientific-validity re-runs (not just formatting) — highest priority:**
- **#1, #2, #3** (`results_prelim.json`, `results_prelim2.json`,
  `results_confirm.json`): produced under the seed-init bug. Every
  multi-seed number in the current `part2_media_main.tex` traces to these
  three files. They are not just missing the new provenance schema — their
  "N seeds" claim is **false** (bit-identical trajectories, see
  `docs/provenance_report.md`). These must be superseded by real M1-M3
  runs under the new manifest+seed-fix, not merely re-tagged.

**Formatting-only gaps (real data, just missing embedded provenance) — lower priority, re-run only if you want the new schema's convenience:**
- #8, #9 (GPU mesh/wavelength sweeps): real single-seed Colab T4 data,
  scientifically fine as single-seed-labeled supplement, just predates
  `run_manifest.py`'s schema.
- #10, #11a, #11b (ablation, RCWA 3-case, RCWA E7): real, deterministic or
  single-trajectory-by-design data, same story.

**Never existed — not a re-run, a first run:**
- #4, #6, #7, #13 (`f1_validate_twin` output, `f2_f3_recovery`,
  `gpu_f2_f3_recovery`, `fit_literature_curves`): no committed result at
  all. #4 and #13 are also blocked on you digitizing literature curves
  (pre-existing blocker, this audit doesn't change it).

## One naming collision worth flagging now, before the restructure

Your S1 ("ablation of media-model components") and the existing
`ablation_gradients.py`/`results_ablation_gradients.json` sound like the
same thing but aren't: the existing ablation compares **gradient
computation pathways** (unrolled autodiff vs. checkpointed adjoint vs.
neural surrogate) — an engineering/implementation question. Your S1 spec
is a **physics** ablation (which NPDD model components — non-locality,
diffusion, dye depletion, saturation — carry the media-aware effect). These
are genuinely different experiments. I'll build S1 as the new physics
ablation and keep `ablation_gradients.py`'s output as supplementary
material under a distinct label, not conflate them.
