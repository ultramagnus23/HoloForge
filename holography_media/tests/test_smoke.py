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


if __name__ == "__main__":
    test_kogelnik_peak()
    test_gradients_flow()
    test_optimization_improves()
    print("ALL SMOKE TESTS PASSED")
