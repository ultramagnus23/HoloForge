"""
GPU-scale NPDD solver sweep: mesh density + convergence threshold.

Two sweeps, 3 runs each (6 total), run on CUDA (Colab GPU runtime):

  1. Mesh density: n_x in {512, 1024, 2048} at a FIXED physical window
     (51.2 um) -- the same window convention already used by
     f1_validate_twin.py (n_x=1024, dx=0.05) and run_confirm.py/
     f2_f3_recovery.py (n_x=512, dx=0.1), so dx is derived, not picked
     arbitrarily: dx = 51.2 / n_x. n_iters is fixed at paper scale (800,
     no early stop) so the only thing varying between these three runs is
     spatial resolution.

  2. Convergence threshold: at the paper-scale mesh (n_x=1024, dx=0.05),
     sweep the NEW `converge_tol` early-stopping option added to
     `media_in_the_loop` (holomedia/optimize.py) -- this parameter did not
     exist before this pass; there was no convergence-threshold concept in
     the optimizer to sweep. tol in {1e-3, 1e-4, 1e-5}, patience=3,
     n_iters capped at 1500 so early stop has room to actually trigger
     before the cap for the looser tolerances.

Medium parameters: MediumParams() defaults -- the same reference medium
used throughout run_confirm.py/f2_f3_recovery.py, so these numbers are
comparable against the existing CPU-scale results, not a fresh unvalidated
config.

Output: results/gpu_reruns/npdd_mesh_sweep/results.json (checkpointed after
every run -- safe to interrupt and re-run; completed cells are skipped).
CPU-scale results elsewhere in the repo are untouched.

Colab usage:
    !git clone <repo_url> && cd HoloForge/holography_media
    !pip install -q torch  # Colab GPU runtime already has a CUDA build
    !python experiments/gpu_npdd_mesh_convergence_sweep.py
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import NPDDRecorder, MediumParams, SlabBPM, media_in_the_loop, psnr
from _gpu_common import get_device, load_checkpoint, save_checkpoint, RunTimer

torch.set_default_dtype(torch.float64)

WINDOW_UM = 51.2          # fixed physical window, matches existing scripts' convention
LAM_UM = 0.405
T_TOTAL, N_STEPS_TIME = 10.0, 300   # fixed temporal resolution, matches npdd.py default
N_Z = 32
PERIOD_UM = 1.6            # target grating period (physical, mesh-independent)

MESH_NX = [512, 1024, 2048]
PAPER_NX, PAPER_DX = 1024, WINDOW_UM / 1024
CONVERGE_TOLS = [1e-3, 1e-4, 1e-5]
PAPER_N_ITERS = 800         # fixed budget for the mesh sweep (no early stop)
CONVERGE_MAX_ITERS = 1500   # cap for the convergence-threshold sweep
PATIENCE = 3
LOG_EVERY = 10
SEED = 0

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpu_reruns",
                   "npdd_mesh_sweep", "results.json")


def bars_target(n_x: int, dx: float, period_um: float, device) -> torch.Tensor:
    period_px = max(2, round(period_um / dx))
    x = torch.arange(n_x, device=device)
    return ((x // (period_px // 2)) % 2).double()


def run_one(n_x: int, dx: float, n_iters: int, device, converge_tol=None):
    params = MediumParams()
    rec = NPDDRecorder(n_x, dx, t_total=T_TOTAL, n_steps=N_STEPS_TIME,
                       params=params).to(device)
    bpm = SlabBPM(n_x, dx, LAM_UM, params.thickness, n_z=N_Z,
                 n0=params.n0).to(device)
    tgt = bars_target(n_x, dx, PERIOD_UM, device)

    with RunTimer(device) as t:
        E, recon, history = media_in_the_loop(
            tgt, rec, bpm, n_iters=n_iters, seed=SEED, log_every=LOG_EVERY,
            verbose=False, converge_tol=converge_tol, patience=PATIENCE)

    result = dict(n_x=n_x, dx=dx, n_iters_budget=n_iters,
                  iters_run=history[-1][0] if history else 0,
                  converge_tol=converge_tol,
                  final_loss=history[-1][1] if history else None,
                  loss_history=history,
                  psnr=psnr(recon, tgt))
    result.update(t.as_dict())
    return result


def main():
    device = get_device(require_gpu=True)
    R = load_checkpoint(OUT)
    R.setdefault("meta", dict(window_um=WINDOW_UM, lam_um=LAM_UM,
                              n_steps_time=N_STEPS_TIME, n_z=N_Z,
                              period_um=PERIOD_UM, seed=SEED,
                              device=str(device)))
    R.setdefault("mesh_density", {})
    R.setdefault("convergence_threshold", {})

    # --- sweep 1: mesh density (fixed physical window, fixed iters) ---
    for n_x in MESH_NX:
        key = f"n_x={n_x}"
        if key in R["mesh_density"]:
            print(f"[mesh] {key} already done, skipping")
            continue
        dx = WINDOW_UM / n_x
        print(f"[mesh] running n_x={n_x} dx={dx:.4f}um ...", flush=True)
        R["mesh_density"][key] = run_one(n_x, dx, PAPER_N_ITERS, device)
        save_checkpoint(OUT, R)
        print(f"[mesh] {key} done: {R['mesh_density'][key]['wall_s']:.1f}s, "
              f"psnr={R['mesh_density'][key]['psnr']:.2f}dB", flush=True)

    # --- sweep 2: convergence threshold (fixed paper-scale mesh) ---
    for tol in CONVERGE_TOLS:
        key = f"tol={tol}"
        if key in R["convergence_threshold"]:
            print(f"[conv] {key} already done, skipping")
            continue
        print(f"[conv] running tol={tol} at n_x={PAPER_NX} ...", flush=True)
        R["convergence_threshold"][key] = run_one(
            PAPER_NX, PAPER_DX, CONVERGE_MAX_ITERS, device, converge_tol=tol)
        save_checkpoint(OUT, R)
        r = R["convergence_threshold"][key]
        print(f"[conv] {key} done: stopped at iter {r['iters_run']}/"
              f"{CONVERGE_MAX_ITERS}, {r['wall_s']:.1f}s, psnr={r['psnr']:.2f}dB",
              flush=True)

    print(f"DONE -- wrote {OUT}")


if __name__ == "__main__":
    main()
