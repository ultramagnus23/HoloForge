"""
F5 (part) -- Gradient-pathway ablation: unrolled autodiff vs. checkpointed
discrete-adjoint vs. neural-surrogate gradients (paper Sec. 3.2 EXPAND note).

Two comparisons:
  1. Gradient fidelity: for a fixed (exposure, target) pair, compute dL/dE
     via each pathway and report wall-clock plus cosine similarity to the
     unrolled (ground-truth) gradient.
  2. Downstream optimization: run media-in-the-loop-style optimization with
     each gradient source for a fixed iteration budget, report final PSNR
     and total wall-clock (surrogate pays a one-time offline training cost,
     reported separately).
"""
import sys, os, time, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import NPDDRecorder, MediumParams, SlabBPM, psnr
from holomedia.optimize import dose_project
from holomedia.surrogate import train_surrogate

torch.set_default_dtype(torch.float64)
N_X, DX, LAM = 256, 0.1, 0.405


def make_target(n):
    x = torch.arange(n)
    return ((x // 16) % 2).double()


def _recon_loss(dn, bpm, target, shrinkage):
    """The actual optimization objective (reconstruction MSE), not a proxy --
    a bare dn.pow(2).sum() probe sits near a flat point for typical E_probe
    (gradient norm ~1e-6), which makes cosine-similarity comparisons
    numerically unstable (floating-point non-associativity in FFT
    recomputation is amplified when the signal itself is that small)."""
    recon = bpm(dn, shrinkage=shrinkage)
    t_norm = target / (target.sum() + 1e-12)
    r_norm = recon / (recon.sum() + 1e-12)
    return torch.mean((r_norm - t_norm) ** 2) * dn.shape[0]


def grad_unrolled(recorder, E, bpm, target):
    E = E.clone().requires_grad_(True)
    t0 = time.time()
    dn = recorder(E)
    loss = _recon_loss(dn, bpm, target, recorder.p.shrinkage)
    loss.backward()
    return E.grad.detach().clone(), time.time() - t0


def grad_checkpointed(recorder, E, bpm, target, block=25):
    E = E.clone().requires_grad_(True)
    t0 = time.time()
    dn = recorder.forward_checkpointed(E, block=block)
    loss = _recon_loss(dn, bpm, target, recorder.p.shrinkage)
    loss.backward()
    return E.grad.detach().clone(), time.time() - t0


def grad_surrogate(surrogate, E, bpm, target, shrinkage):
    E = E.clone().requires_grad_(True)
    t0 = time.time()
    dn = surrogate(E)
    loss = _recon_loss(dn, bpm, target, shrinkage)
    loss.backward()
    return E.grad.detach().clone(), time.time() - t0


def cos_sim(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def optimize_with_grad_fn(recorder, bpm, target, grad_fn, n_iters=150, lr=5e-2,
                          dose_budget=1.0, eval_recorder=None):
    """Generic optimizer: grad_fn(E) -> dL/dE (custom gradient source);
    reconstruction quality always evaluated on the TRUE twin."""
    eval_recorder = eval_recorder or recorder
    theta = torch.zeros(recorder.n_x, dtype=torch.float64, requires_grad=False)
    t_norm = target / (target.sum() + 1e-12)
    m = torch.zeros_like(theta); v = torch.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t0 = time.time()
    for it in range(1, n_iters + 1):
        theta.requires_grad_(True)
        E = torch.nn.functional.softplus(theta) + 1e-6
        E = dose_project(E, dose_budget)
        dn = grad_fn.recorder_forward(E)
        recon = bpm(dn, shrinkage=recorder.p.shrinkage)
        r_norm = recon / (recon.sum() + 1e-12)
        loss = torch.mean((r_norm - t_norm) ** 2) * recorder.n_x
        grad_theta, = torch.autograd.grad(loss, theta)
        with torch.no_grad():
            m = beta1 * m + (1 - beta1) * grad_theta
            v = beta2 * v + (1 - beta2) * grad_theta ** 2
            mhat = m / (1 - beta1 ** it); vhat = v / (1 - beta2 ** it)
            theta = (theta - lr * mhat / (vhat.sqrt() + eps)).detach()
    wall = time.time() - t0
    with torch.no_grad():
        E = dose_project(torch.nn.functional.softplus(theta) + 1e-6, dose_budget)
        recon_true = bpm(eval_recorder(E), shrinkage=recorder.p.shrinkage)
    return psnr(recon_true, target), wall


class _ForwardWrapper:
    def __init__(self, fn):
        self.recorder_forward = fn


def main():
    p = MediumParams()
    rec = NPDDRecorder(N_X, DX, t_total=8, n_steps=150, params=p)
    bpm = SlabBPM(N_X, DX, LAM, p.thickness, n_z=16, n0=p.n0)
    target = make_target(N_X)

    torch.manual_seed(0)
    E_probe = torch.rand(N_X, dtype=torch.float64) * 0.5 + 0.75

    print("== Gradient fidelity (fixed exposure/target probe, actual reconstruction loss) ==")
    g_unroll, t_unroll = grad_unrolled(rec, E_probe, bpm, target)
    g_ckpt, t_ckpt = grad_checkpointed(rec, E_probe, bpm, target, block=25)
    print(f"unrolled       : {t_unroll:.3f}s  (reference, |grad|={g_unroll.norm():.3e})")
    print(f"adjoint(ckpt25): {t_ckpt:.3f}s  cos-sim to unrolled = {cos_sim(g_unroll, g_ckpt):.6f}")

    t0 = time.time()
    surrogate, surrogate_mse = train_surrogate(rec, N_X, n_samples=300, epochs=200, verbose=False)
    surrogate_train_time = time.time() - t0
    g_sur, t_sur = grad_surrogate(surrogate, E_probe, bpm, target, rec.p.shrinkage)
    print(f"surrogate      : {t_sur:.3f}s (+ {surrogate_train_time:.1f}s one-time offline training, "
          f"fit MSE {surrogate_mse:.3e})  cos-sim to unrolled = {cos_sim(g_unroll, g_sur):.4f}")

    print("\n== Downstream optimization (150 iters, fixed budget) ==")
    N_ITERS = 150
    fw_true = _ForwardWrapper(lambda E: rec(E))
    fw_ckpt = _ForwardWrapper(lambda E: rec.forward_checkpointed(E, block=25))
    fw_sur = _ForwardWrapper(lambda E: surrogate(E))

    psnr_unroll, wall_unroll = optimize_with_grad_fn(rec, bpm, target, fw_true, n_iters=N_ITERS)
    print(f"unrolled       : PSNR {psnr_unroll:.2f} dB   wall {wall_unroll:.1f}s")

    psnr_ckpt, wall_ckpt = optimize_with_grad_fn(rec, bpm, target, fw_ckpt, n_iters=N_ITERS)
    print(f"adjoint(ckpt25): PSNR {psnr_ckpt:.2f} dB   wall {wall_ckpt:.1f}s")

    psnr_sur, wall_sur = optimize_with_grad_fn(rec, bpm, target, fw_sur, n_iters=N_ITERS,
                                               eval_recorder=rec)
    print(f"surrogate      : PSNR {psnr_sur:.2f} dB   wall {wall_sur:.1f}s "
          f"(+ {surrogate_train_time:.1f}s offline training)")

    out = dict(
        fidelity=dict(unrolled_time=t_unroll,
                      checkpoint_time=t_ckpt, checkpoint_cossim=cos_sim(g_unroll, g_ckpt),
                      surrogate_time=t_sur, surrogate_cossim=cos_sim(g_unroll, g_sur),
                      surrogate_train_time=surrogate_train_time, surrogate_fit_mse=surrogate_mse),
        optimization=dict(
            unrolled=dict(psnr=psnr_unroll, wall=wall_unroll),
            checkpoint=dict(psnr=psnr_ckpt, wall=wall_ckpt),
            surrogate=dict(psnr=psnr_sur, wall=wall_sur, offline_train_wall=surrogate_train_time),
        ),
    )
    with open(os.path.join(os.path.dirname(__file__), "..", "results_ablation_gradients.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results_ablation_gradients.json")


if __name__ == "__main__":
    main()
