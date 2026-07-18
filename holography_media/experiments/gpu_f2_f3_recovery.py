"""
GPU-scale, BATCHED F2/F3 method-comparison sweep (paper Figures 2 and 3).

Same sweep as experiments/f2_f3_recovery.py (5 axes x their value lists =
26 settings, x 4 targets x 5 seeds), but each setting's 4 targets x 5 seeds
= 20 (target, seed) combinations run as ONE batched Adam loop per method
instead of 20 sequential Python-loop calls -- see holomedia/optimize.py's
"batched methods" section. This exists because the un-batched GPU sweep
(gpu_npdd_mesh_convergence_sweep.py) measured wall-clock FLAT across a 4x
mesh-resolution range (~450s regardless of n_x=512/1024/2048): that run is
overhead-bound (per-iteration Python/kernel-launch cost dominates the
actual FFT math at this problem size), not compute-bound, so the naive
un-batched full sweep would cost an estimated ~195 GPU-hours. Batching the
20 (target, seed) combinations into one call amortizes that per-iteration
overhead across the batch; batched vs. looped-unbatched correctness is
verified bit-identical in tests/test_smoke.py::test_batched_matches_unbatched
and directly for all 4 methods during development (max diff ~1e-14).

IMPORTANT -- this script's speedup has NOT been measured on an actual GPU
(none was available while writing it). Run with mode="probe" FIRST: it
times exactly one setting (20 combos x 4 methods) and prints a projected
total time for all 26 settings before you commit to the full run. Do not
skip straight to mode="full" on trust.

Paper-scale params (n_x=1024, n_iters=800, 5 seeds) per README.md's
"GPU rerun" recipe -- dx is left at the ORIGINAL script's 0.1 (window
grows to 102.4um at n_x=1024, not held fixed at 51.2um like the mesh-sweep
script); this matches what f2_f3_recovery.py's own paper-scale comment
says to change (N_X/N_ITERS/SEEDS only), not what gpu_npdd_mesh_
convergence_sweep.py used for its window-fixed mesh sweep -- a real
methodological difference between those two GPU scripts, noted here so
it isn't mistaken for an inconsistency.

Output: results/gpu_reruns/f2_f3_recovery/results.json, checkpointed
after every one of the 26 settings (safe to interrupt/resume). Does NOT
touch results_f2f3.json (the un-batched CPU-scale script's own output).

Colab usage:
    !python experiments/gpu_f2_f3_recovery.py probe   # ~1 setting, prints ETA
    !python experiments/gpu_f2_f3_recovery.py full     # all 26, checkpointed
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
                       media_in_the_loop_batched, media_blind_sgd_batched,
                       media_blind_gs_batched, oracle_ideal_batched,
                       psnr_batch, diffraction_efficiency_batch)
from _gpu_common import get_device, load_checkpoint, save_checkpoint, RunTimer

torch.set_default_dtype(torch.float64)

N_X, DX, LAM = 1024, 0.1, 0.405     # paper scale per README's f2_f3_recovery.py recipe
SEEDS = [0, 1, 2, 3, 4]
N_ITERS = 800
N_Z = 24                             # matches f2_f3_recovery.py's n_z=24

SWEEPS = {
    "sigma": [0.02, 0.05, 0.08, 0.12, 0.20, 0.30],
    "D0": [0.01, 0.03, 0.1, 0.3, 1.0],
    "dn_max": [1e-3, 2e-3, 3.5e-3, 5e-3, 6e-3],
    "shrinkage": [0.0, 0.003, 0.01, 0.02, 0.03],
    "thickness": [10.0, 20.0, 30.0, 50.0, 80.0],
}

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpu_reruns",
                   "f2_f3_recovery", "results.json")


def make_targets(n_x, device):
    x = torch.arange(n_x, device=device)
    targets = {}
    for period in [16, 32, 64]:
        targets[f"bars{period}"] = ((x // (period // 2)) % 2).double()
    g = torch.zeros(n_x, device=device)
    torch.manual_seed(7)  # fixed spots-target geometry, independent of sweep seeds
    for _ in range(6):
        c = torch.randint(n_x // 8, 7 * n_x // 8, (1,)).item()
        w = torch.randint(8, 40, (1,)).item()
        g[max(0, c - w):min(n_x, c + w)] = torch.rand(1).item() + 0.3
    targets["spots"] = g
    return targets


def run_setting(params: MediumParams, device):
    """Run all 4 methods, batched over (target x seed) = 4*5 = 20 rows."""
    rec = NPDDRecorder(N_X, DX, t_total=10, n_steps=250, params=params).to(device)
    bpm = SlabBPM(N_X, DX, LAM, params.thickness, n_z=N_Z, n0=params.n0).to(device)

    tgt_dict = make_targets(N_X, device)
    names = list(tgt_dict.keys())
    B = len(names) * len(SEEDS)
    targets_batch = torch.stack([tgt_dict[n] for n in names]).repeat_interleave(len(SEEDS), dim=0)
    seeds_batch = SEEDS * len(names)
    row_meta = [(names[i // len(SEEDS)], seeds_batch[i]) for i in range(B)]
    masks = (targets_batch > 0.05).double()

    _, r_ours, _ = media_in_the_loop_batched(targets_batch, rec, bpm, seeds_batch,
                                             n_iters=N_ITERS, verbose=False)
    _, r_blind = media_blind_sgd_batched(targets_batch, rec, bpm, seeds_batch, n_iters=N_ITERS)
    _, r_gs = media_blind_gs_batched(targets_batch, rec, bpm, seeds_batch)
    _, r_orc = oracle_ideal_batched(targets_batch, rec, bpm, seeds_batch, n_iters=N_ITERS)

    p_ours, p_blind = psnr_batch(r_ours, targets_batch), psnr_batch(r_blind, targets_batch)
    p_gs, p_orc = psnr_batch(r_gs, targets_batch), psnr_batch(r_orc, targets_batch)
    de_ours = diffraction_efficiency_batch(r_ours, masks)
    de_blind = diffraction_efficiency_batch(r_blind, masks)

    rows = []
    for i, (name, seed) in enumerate(row_meta):
        rows.append(dict(target=name, seed=seed,
                         psnr_ours=float(p_ours[i]), psnr_blind=float(p_blind[i]),
                         psnr_gs=float(p_gs[i]), psnr_oracle=float(p_orc[i]),
                         de_ours=float(de_ours[i]), de_blind=float(de_blind[i])))
    return rows


def probe(device):
    print("[probe] running ONE setting (sigma=0.08 default) to measure wall-clock...")
    p = MediumParams()
    with RunTimer(device) as t:
        rows = run_setting(p, device)
    n_settings = sum(len(v) for v in SWEEPS.values())
    est_total_s = t.wall_s * n_settings
    print(f"[probe] one setting: {t.wall_s:.1f}s, peak_mem={t.peak_mem_mb}MB, "
          f"{len(rows)} rows (target x seed combos)")
    print(f"[probe] {n_settings} settings total -> projected full-sweep time: "
          f"{est_total_s:.0f}s ({est_total_s/3600:.2f}h)")
    print("[probe] compare against the un-batched estimate (~195h) before "
          "deciding whether to run mode='full'.")


def full(device):
    R = load_checkpoint(OUT)
    R.setdefault("meta", dict(n_x=N_X, dx=DX, lam=LAM, seeds=SEEDS,
                              n_iters=N_ITERS, n_z=N_Z, device=str(device),
                              batched=True))
    R.setdefault("settings", {})

    for axis, values in SWEEPS.items():
        for v in values:
            key = f"{axis}={v}"
            if key in R["settings"]:
                print(f"[full] {key} already done, skipping")
                continue
            p = MediumParams()
            setattr(p, axis, v)
            print(f"[full] running {key} ...", flush=True)
            with RunTimer(device) as t:
                rows = run_setting(p, device)
            R["settings"][key] = dict(rows=rows, **t.as_dict())
            save_checkpoint(OUT, R)
            print(f"[full] {key} done: {t.wall_s:.1f}s", flush=True)

    print(f"DONE -- wrote {OUT}")


if __name__ == "__main__":
    device = get_device(require_gpu=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        probe(device)
    elif mode == "full":
        full(device)
    else:
        raise SystemExit(f"unknown mode {mode!r}, use 'probe' or 'full'")
