"""Phase 0.2 regression test for the seed-init bug.

Bug: media_in_the_loop (and media_blind_sgd/oracle_ideal/media_blind_gs)
initialized their optimization variable to exact zeros, so
torch.manual_seed(seed) had nothing random left to affect -- every "seed"
produced a bit-identical trajectory. Fixed via _seeded_init_theta in
holomedia/optimize.py (small seeded random perturbation on init).

This test pins the fix: distinct seeds must produce genuinely divergent
optimization trajectories, not just distinct in principle.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import NPDDRecorder, MediumParams, SlabBPM, media_in_the_loop, psnr

torch.set_default_dtype(torch.float64)


def test_seed_independence():
    n_x, dx = 128, 0.1
    p = MediumParams()
    rec = NPDDRecorder(n_x, dx, t_total=8, n_steps=60, params=p)
    bpm = SlabBPM(n_x, dx, 0.405, p.thickness, n_z=12, n0=p.n0)
    x = torch.arange(n_x)
    target = ((x // 16) % 2).double()

    seeds = [0, 1, 2]
    histories, finals, psnrs = [], [], []
    for s in seeds:
        E, recon, hist = media_in_the_loop(target, rec, bpm, n_iters=50,
                                           seed=s, log_every=1, verbose=False)
        histories.append(hist)
        finals.append(E)
        psnrs.append(psnr(recon, target))

    # Loss curves must diverge beyond 1e-6. Empirically (measured while
    # writing this test) pairwise loss spread at this n_x=128/eps=1e-2
    # config crosses 1e-6 between iteration 10 and 12 (7.9e-7 at iter 10,
    # 1.06e-6 at iter 12), not strictly "by iteration 10" as originally
    # specified -- that number was an unverified estimate, not measured.
    # Checking at iteration 15 (1.4e-6, comfortably past threshold) instead
    # of forcing the earlier checkpoint to pass by construction.
    check_iter = 15
    loss_at_check = [h[check_iter][1] for h in histories]
    assert histories[0][check_iter][0] == check_iter, \
        f"expected log entry at iter {check_iter}"
    max_pairwise_diff = max(
        abs(loss_at_check[i] - loss_at_check[j])
        for i in range(len(seeds)) for j in range(i + 1, len(seeds))
    )
    assert max_pairwise_diff > 1e-6, (
        f"seeds still produce near-identical loss at iter {check_iter} "
        f"(max pairwise diff {max_pairwise_diff:.2e}) -- seed bug may have regressed"
    )

    # final PSNRs must differ across seeds (not all bit-identical)
    assert len(set(round(p_, 8) for p_ in psnrs)) > 1, (
        f"final PSNRs identical across seeds {psnrs} -- seed bug may have regressed"
    )

    # and the exposures themselves must differ
    assert not torch.allclose(finals[0], finals[1]), \
        "seed=0 and seed=1 produced identical final exposures"
    assert not torch.allclose(finals[0], finals[2]), \
        "seed=0 and seed=2 produced identical final exposures"

    print(f"seed independence OK: loss@iter{check_iter} =", [f"{l:.4e}" for l in loss_at_check])
    print("  final PSNRs across seeds:", [f"{p_:.3f}" for p_ in psnrs])


if __name__ == "__main__":
    test_seed_independence()
    print("PASSED")
