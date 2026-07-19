"""
GPU-scale BPM diffraction wavelength sweep.

Records ONE hologram (media-in-the-loop optimization at the design
wavelength, 405 nm -- the wavelength the configs/media/*.yaml files are
labeled for) at paper scale, then reads the SAME recorded index profile
out through SlabBPM at each of 5 wavelengths spanning 400-450 nm. This
brackets the paper's stated 405/450 nm operating band (part2_media_draft_v1.md
Sec. 1 point 4, and the F4 design-chart description) -- it is not an
arbitrary range. Recording is done once; only the diffraction (readout)
stage is re-run per wavelength, since "BPM diffraction wavelength sweep" is
about the readout module's off-design behavior, not re-recording at each
wavelength (that would conflate two different physical questions).

Output: results/gpu_reruns/bpm_wavelength_sweep/results.json (checkpointed
after every wavelength; safe to interrupt/resume). CPU-scale results
elsewhere in the repo are untouched.

Colab usage:
    !git clone <repo_url> && cd HoloForge/holography_media
    !python experiments/gpu_bpm_wavelength_sweep.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
                       media_in_the_loop, psnr, diffraction_efficiency)
from _gpu_common import get_device, load_checkpoint, save_checkpoint, RunTimer

torch.set_default_dtype(torch.float64)

N_X, DX = 1024, 51.2 / 1024   # paper-scale mesh, same window convention as
                               # gpu_npdd_mesh_convergence_sweep.py / f1_validate_twin.py
DESIGN_LAM_UM = 0.405
SWEEP_LAM_UM = [0.400, 0.405, 0.420, 0.435, 0.450]  # brackets paper's 405/450nm band
T_TOTAL, N_STEPS_TIME, N_Z = 10.0, 300, 32
N_ITERS, SEED = 800, 0
PERIOD_UM = 1.6

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "gpu_reruns",
                   "bpm_wavelength_sweep", "results.json")


def bars_target(n_x, dx, period_um, device):
    period_px = max(2, round(period_um / dx))
    x = torch.arange(n_x, device=device)
    return ((x // (period_px // 2)) % 2).double()


def main():
    device = get_device(require_gpu=True)
    R = load_checkpoint(OUT)
    R.setdefault("meta", dict(n_x=N_X, dx=DX, design_lam_um=DESIGN_LAM_UM,
                              sweep_lam_um=SWEEP_LAM_UM, n_iters=N_ITERS,
                              seed=SEED, device=str(device)))
    R.setdefault("by_wavelength", {})

    params = MediumParams()
    rec = NPDDRecorder(N_X, DX, t_total=T_TOTAL, n_steps=N_STEPS_TIME,
                       params=params).to(device)
    tgt = bars_target(N_X, DX, PERIOD_UM, device)

    if "recording" not in R:
        design_bpm = SlabBPM(N_X, DX, DESIGN_LAM_UM, params.thickness,
                             n_z=N_Z, n0=params.n0).to(device)
        print(f"[record] optimizing exposure at design wavelength "
              f"{DESIGN_LAM_UM}um, n_x={N_X} ...", flush=True)
        with RunTimer(device) as t:
            E, recon_design, history = media_in_the_loop(
                tgt, rec, design_bpm, n_iters=N_ITERS, seed=SEED,
                log_every=25, verbose=False)
            dn_profile = rec(E)
        R["recording"] = dict(
            final_loss=history[-1][1] if history else None,
            psnr_at_design_wavelength=psnr(recon_design, tgt), **t.as_dict())
        R["_dn_profile"] = dn_profile.detach().cpu().tolist()
        save_checkpoint(OUT, R)
        print(f"[record] done: {R['recording']['wall_s']:.1f}s, "
              f"psnr_at_design={R['recording']['psnr_at_design_wavelength']:.2f}dB",
              flush=True)
    else:
        print("[record] recorded profile already checkpointed, reusing", flush=True)
        dn_profile = torch.tensor(R["_dn_profile"], device=device,
                                  dtype=torch.float64)

    mask = (tgt > 0.05).double()
    for lam in SWEEP_LAM_UM:
        key = f"{lam}"
        if key in R["by_wavelength"]:
            print(f"[readout] lam={lam}um already done, skipping")
            continue
        print(f"[readout] lam={lam}um ...", flush=True)
        bpm = SlabBPM(N_X, DX, lam, params.thickness, n_z=N_Z,
                     n0=params.n0).to(device)
        with RunTimer(device) as t:
            recon = bpm(dn_profile, shrinkage=params.shrinkage)
        R["by_wavelength"][key] = dict(
            lam_um=lam, detuning_from_design_um=lam - DESIGN_LAM_UM,
            psnr=psnr(recon, tgt),
            diffraction_efficiency=diffraction_efficiency(recon, mask),
            **t.as_dict())
        save_checkpoint(OUT, R)
        r = R["by_wavelength"][key]
        print(f"[readout] lam={lam}um done: psnr={r['psnr']:.2f}dB "
              f"de={r['diffraction_efficiency']:.4f}", flush=True)

    print(f"DONE -- wrote {OUT}")


if __name__ == "__main__":
    main()
