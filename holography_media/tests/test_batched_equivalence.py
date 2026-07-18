"""Phase 0.3: re-verify batched == unbatched, post seed-fix.

The *_batched functions (holomedia/optimize.py) were verified bit-identical
to the unbatched per-item loop when first written (~1e-14). The seed-init
fix (test_seed_independence.py) changed how theta is initialized in BOTH
paths via the same _seeded_init_theta/_seeded_init_theta_batch helpers, so
this re-verifies the equivalence still holds post-fix rather than assuming
it does because it held before a related change.

Backward scalar in the batched path is loss_per_row.sum() (not .mean()) so
each row's Adam trajectory is mathematically independent of the other rows
in the batch -- see optimize.py's "batched methods" section docstring for
why this makes bit-identical agreement possible at all.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from holomedia import (NPDDRecorder, MediumParams, SlabBPM,
    media_in_the_loop, media_blind_sgd, media_blind_gs, oracle_ideal,
    media_in_the_loop_batched, media_blind_sgd_batched,
    media_blind_gs_batched, oracle_ideal_batched)

torch.set_default_dtype(torch.float64)

TOL = 1e-10


def _setup():
    n_x, dx = 64, 0.1
    p = MediumParams()
    rec = NPDDRecorder(n_x, dx, t_total=8, n_steps=25, params=p)
    bpm = SlabBPM(n_x, dx, 0.405, p.thickness, n_z=8, n0=p.n0)
    x = torch.arange(n_x)
    targets = torch.stack([
        ((x // 8) % 2).double(),
        ((x // 16) % 2).double(),
        ((x // 4) % 2).double(),
    ])
    seeds = [0, 1, 2]
    return rec, bpm, targets, seeds


def test_media_in_the_loop_batched_matches_unbatched():
    rec, bpm, targets, seeds = _setup()
    n_iters = 20
    E_loop, losses_loop = [], []
    for i, s in enumerate(seeds):
        E, r, hist = media_in_the_loop(targets[i], rec, bpm, n_iters=n_iters,
                                       seed=s, log_every=n_iters - 1, verbose=False)
        E_loop.append(E)
        losses_loop.append(hist[-1][1])
    E_loop = torch.stack(E_loop)

    Eb, Rb, histb = media_in_the_loop_batched(targets, rec, bpm, seeds,
                                              n_iters=n_iters, verbose=False)

    max_E_diff = (E_loop - Eb).abs().max().item()
    assert max_E_diff < TOL, f"media_in_the_loop batched E diverged: {max_E_diff:.2e}"
    print("media_in_the_loop: max E diff =", max_E_diff)


def test_media_blind_sgd_batched_matches_unbatched():
    rec, bpm, targets, seeds = _setup()
    n_iters = 20
    E_loop = torch.stack([media_blind_sgd(targets[i], rec, bpm, n_iters=n_iters,
                                          seed=seeds[i])[0] for i in range(3)])
    Eb, Rb = media_blind_sgd_batched(targets, rec, bpm, seeds, n_iters=n_iters)
    max_diff = (E_loop - Eb).abs().max().item()
    assert max_diff < TOL, f"media_blind_sgd batched E diverged: {max_diff:.2e}"
    print("media_blind_sgd: max E diff =", max_diff)


def test_oracle_ideal_batched_matches_unbatched():
    rec, bpm, targets, seeds = _setup()
    n_iters = 20
    E_loop = torch.stack([oracle_ideal(targets[i], rec, bpm, n_iters=n_iters,
                                       seed=seeds[i])[0] for i in range(3)])
    Eb, Rb = oracle_ideal_batched(targets, rec, bpm, seeds, n_iters=n_iters)
    max_diff = (E_loop - Eb).abs().max().item()
    assert max_diff < TOL, f"oracle_ideal batched E diverged: {max_diff:.2e}"
    print("oracle_ideal: max E diff =", max_diff)


def test_media_blind_gs_batched_matches_unbatched():
    rec, bpm, targets, seeds = _setup()
    E_loop = torch.stack([media_blind_gs(targets[i], rec, bpm, seed=seeds[i])[0]
                          for i in range(3)])
    Eb, Rb = media_blind_gs_batched(targets, rec, bpm, seeds)
    max_diff = (E_loop - Eb).abs().max().item()
    assert max_diff < TOL, f"media_blind_gs batched E diverged: {max_diff:.2e}"
    print("media_blind_gs: max E diff =", max_diff)


if __name__ == "__main__":
    test_media_in_the_loop_batched_matches_unbatched()
    test_media_blind_sgd_batched_matches_unbatched()
    test_oracle_ideal_batched_matches_unbatched()
    test_media_blind_gs_batched_matches_unbatched()
    print("PASSED")
