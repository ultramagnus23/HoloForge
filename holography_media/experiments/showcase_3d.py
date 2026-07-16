"""
F5 (part) -- One 3D (x, y, z) showcase case (paper Sec. 6 claim: "a 3D case
demonstrates that no qualitative behavior is lost").

Target: a small 2D binary pattern (a ring), recorded via NPDDRecorder3D and
read out via SlabBPM3D. Compares media-in-the-loop optimization against
media-blind (naive linear exposure = target) at the same dose budget, on
the full 3D volume -- the qualitative analogue of the 1D/2D(x,z) main
results (Sec. 5, F1-F3), run once in the full 3D geometry.

Kept intentionally small (n_x=n_y=48) since NPDDRecorder3D's per-step cost
scales as O(n_x^2 n_y^2 log) via 2D FFT and there is no GPU in this
environment; this is a qualitative demonstration, not a paper-scale 3D sweep.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import MediumParams, psnr
from holomedia.npdd3d import NPDDRecorder3D
from holomedia.diffraction3d import SlabBPM3D
from holomedia.optimize import dose_project

torch.set_default_dtype(torch.float64)
N, DX, LAM = 48, 0.15, 0.405
N_STEPS, N_Z, N_ITERS = 120, 12, 200


def ring_target(n):
    x = torch.linspace(-1, 1, n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2)
    return ((r > 0.35) & (r < 0.55)).double()


def media_in_the_loop_3d(target, rec, bpm, n_iters=200, lr=5e-2, dose_budget=1.0, seed=0):
    torch.manual_seed(seed)
    theta = torch.zeros(rec.n_x, rec.n_y, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    t_norm = target / (target.sum() + 1e-12)
    for it in range(n_iters):
        opt.zero_grad()
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = dose_project(E, dose_budget)
        dn = rec(E)
        recon = bpm(dn)
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * (rec.n_x * rec.n_y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        E = dose_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon = bpm(rec(E), )
    return E.detach(), recon.detach()


def media_blind_3d(target, rec, bpm, dose_budget=1.0):
    """Naive linear exposure = target, dose-normalized -- current practice,
    evaluated on the real 3D twin (no compensation for NPDD blur/saturation)."""
    E = dose_project(target + 1e-3, dose_budget)
    with torch.no_grad():
        recon = bpm(rec(E))
    return E, recon


def main():
    p = MediumParams()
    rec = NPDDRecorder3D(N, N, DX, DX, t_total=8, n_steps=N_STEPS, params=p)
    bpm = SlabBPM3D(N, N, DX, DX, LAM, p.thickness, n_z=N_Z, n0=p.n0)
    target = ring_target(N)

    t0 = time.time()
    E_blind, recon_blind = media_blind_3d(target, rec, bpm)
    t1 = time.time()
    print(f"media-blind (3D)        : {t1-t0:.1f}s")

    E_ours, recon_ours = media_in_the_loop_3d(target, rec, bpm, n_iters=N_ITERS)
    t2 = time.time()
    print(f"media-in-the-loop (3D)  : {t2-t1:.1f}s  ({N_ITERS} iters)")

    p_blind = psnr(recon_blind, target)
    p_ours = psnr(recon_ours, target)
    print(f"\nPSNR media-blind        : {p_blind:.2f} dB")
    print(f"PSNR media-in-the-loop  : {p_ours:.2f} dB")
    print(f"gain                    : {p_ours - p_blind:+.2f} dB")
    print("\nQualitative check: same direction of effect as the 2D(x,z) main "
          "results (media-in-the-loop beats naive linear exposure) -- "
          "consistent with Sec. 6's claim that no qualitative behavior is "
          "lost moving to full 3D." if p_ours > p_blind else
          "\nWARNING: 3D showcase did NOT reproduce the 2D qualitative "
          "direction (ours should beat blind) -- flag before citing Sec. 6's claim.")

    out = dict(n=N, n_steps=N_STEPS, n_z=N_Z, n_iters=N_ITERS,
              psnr_blind=p_blind, psnr_ours=p_ours, gain_db=p_ours - p_blind,
              wall_blind=t1 - t0, wall_ours=t2 - t1)
    with open(os.path.join(os.path.dirname(__file__), "..", "results_3d_showcase.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results_3d_showcase.json")


if __name__ == "__main__":
    main()
