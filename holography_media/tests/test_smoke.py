"""Fast end-to-end smoke test: twin runs, gradients flow, ours beats blind-GS."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM, kogelnik_de,
                       media_in_the_loop, media_blind_gs, psnr)

torch.set_default_dtype(torch.float64)


def test_kogelnik_peak():
    # DE should reach ~1 when nu = pi/2 -> dn*T = lam*cos(th)/2
    import math
    lam, th = 0.405, math.radians(10)
    dn = torch.tensor(lam * math.cos(th) / (2 * 30.0))
    de = kogelnik_de(dn, 30.0, lam)
    assert abs(float(de) - 1.0) < 1e-6, float(de)
    print("kogelnik peak OK:", float(de))


def test_gradients_flow():
    rec = NPDDRecorder(128, 0.1, t_total=3, n_steps=40)
    E = torch.rand(128, dtype=torch.float64, requires_grad=True)
    dn = rec(torch.nn.functional.softplus(E))
    dn.sum().backward()
    g = E.grad.abs().sum()
    assert torch.isfinite(g) and g > 0
    print("gradients flow OK, |grad| sum =", float(g))


def test_optimization_improves():
    n = 256
    rec = NPDDRecorder(n, 0.1, t_total=8, n_steps=120)
    bpm = SlabBPM(n, 0.1, 0.405, rec.p.thickness, n_z=12)
    x = torch.arange(n)
    target = ((x // 16) % 2).double()
    _, r_ours, hist = media_in_the_loop(target, rec, bpm, n_iters=120,
                                        verbose=False)
    _, r_gs = media_blind_gs(target, rec, bpm)
    p_ours, p_gs = psnr(r_ours, target), psnr(r_gs, target)
    print(f"loss start {hist[0][1]:.3e} -> end {hist[-1][1]:.3e}")
    print(f"PSNR ours {p_ours:.2f} dB vs blind-GS {p_gs:.2f} dB")
    assert hist[-1][1] < hist[0][1], "loss did not decrease"
    assert p_ours > p_gs, "media-in-the-loop did not beat blind GS"


def test_checkpointed_forward_matches():
    from holomedia.npdd import NPDDRecorder as _R
    rec = _R(64, 0.1, t_total=3, n_steps=60)
    torch.manual_seed(0)
    E = torch.rand(64, dtype=torch.float64) * 0.5 + 0.75
    dn1 = rec(E)
    dn2 = rec.forward_checkpointed(E, block=15)
    diff = (dn1 - dn2).abs().max()
    assert diff < 1e-10, f"checkpointed forward diverged from forward(): {diff}"
    print("checkpointed forward matches forward() OK, max diff =", float(diff))


def test_3d_twin_runs():
    from holomedia.npdd3d import NPDDRecorder3D
    from holomedia.diffraction3d import SlabBPM3D
    rec = NPDDRecorder3D(24, 24, 0.15, 0.15, t_total=4, n_steps=30)
    bpm = SlabBPM3D(24, 24, 0.15, 0.15, 0.405, rec.p.thickness, n_z=6)
    E = (torch.rand(24, 24, dtype=torch.float64) * 0.5 + 0.5).requires_grad_(True)
    dn = rec(E)
    recon = bpm(dn)
    assert recon.shape == (24, 24) and torch.isfinite(recon).all()
    recon.sum().backward()
    assert torch.isfinite(E.grad).all() and E.grad.abs().sum() > 0
    print("3D twin runs OK, recon sum =", float(recon.detach().sum()))


if __name__ == "__main__":
    test_kogelnik_peak()
    test_gradients_flow()
    test_optimization_improves()
    test_checkpointed_forward_matches()
    test_3d_twin_runs()
    print("ALL SMOKE TESTS PASSED")
