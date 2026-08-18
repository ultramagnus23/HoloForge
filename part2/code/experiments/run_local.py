r"""
One-command local runner for M1/M2/S1/S2, for a local CUDA GPU (e.g. a
laptop RTX 3050, 4GB VRAM) instead of the Colab path in
notebooks/colab_runner.ipynb. No Drive mount, no repo clone (the code is
already local) -- everything else (resumable jobs, atomic writes, stall
detection, per-job seeding, the Gate-1 probe table) is identical, reusing
run_manifest.py's functions directly so there is exactly one execution
engine, not a second parallel implementation.

All field sizes in this pipeline are 1D with a small number of BPM/NPDD
steps -- per-job VRAM is on the order of tens of MB even with n_iters=800
unrolled, so 4GB is not a binding constraint; the Gate-1 table's
compute-hour estimate is about wall-clock time, not memory.

n_x defaults are PER TIER, not one global value: M1/M2 stay at the
paper-scale 1024 (that's the headline cliff-location result). S1/S2
default to 512 -- results/gpu_reruns/npdd_mesh_sweep/results.json shows
PSNR is mesh-independent within 0.04dB across n_x=512/1024/2048, so this
is a real, evidenced cut for the supporting tiers, not an unjustified
shortcut. Override per-invocation with --n-x if you want uniform scale.

--workers N spawns N subprocesses per manifest, each running a disjoint
--shard i/N slice (see run_manifest.py's apply_shard() docstring for why
this is safe with no coordination). Jobs here are 1D and small, so this
is the practical way to use more than one CPU core / GPU context instead
of the strictly-serial single-process default -- run_manifest.py has no
in-process parallelism.

Usage:
    python -m experiments.run_local                       # all 4 tiers, serial
    python -m experiments.run_local --manifests M1         # just M1 first --
                                                            # recommended: look
                                                            # at the aggregated
                                                            # M1 result before
                                                            # committing compute
                                                            # to M2/S1/S2
    python -m experiments.run_local --workers 8             # 8-way sharded
    python -m experiments.run_local --skip-probe --chunk-minutes 15

If a run is interrupted (Ctrl+C, crash, reboot), just run the same
command again -- already-finished jobs (results/{id}/{hash}/seed{n}.json)
are detected and skipped automatically.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import run_manifest as rm

HERE = os.path.dirname(__file__)
DEFAULT_ORDER = ["M1", "M2", "S1", "S2"]
TIER_N_X = {"M1": 1024, "M2": 1024, "S1": 512, "S2": 512}


def _run_sharded(name: str, n_x: int, n_iters: int, workers: int) -> None:
    """Spawn `workers` subprocesses, each running --shard i/workers of
    manifest `name` to completion, and wait for all of them. stdout/stderr
    are inherited (not captured) so every worker's heartbeat lines
    interleave live in this terminal -- the [heartbeat] job= prefix on
    each line is enough to tell shards apart."""
    script = os.path.join(HERE, "run_manifest.py")
    procs = []
    for i in range(workers):
        cmd = [sys.executable, script, "--manifest", name,
              "--shard", f"{i}/{workers}", "--n-x", str(n_x),
              "--n-iters", str(n_iters)]
        procs.append(subprocess.Popen(cmd))
    failed = [p.pid for p in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f"[run_local] {len(failed)}/{workers} shard "
                          f"process(es) for manifest {name!r} exited "
                          f"non-zero (pids {failed}) -- check their output "
                          f"above for the actual error.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=DEFAULT_ORDER,
                    choices=DEFAULT_ORDER,
                    help="which manifests to run, in order (default: all four; "
                         "recommended: run --manifests M1 alone first, look at "
                         "the aggregated result, THEN commit compute to the rest)")
    ap.add_argument("--n-x", type=int, default=None,
                    help="override the per-tier default (M1/M2=1024, S1/S2=512, "
                         "see module docstring) uniformly for all requested tiers")
    ap.add_argument("--n-iters", type=int, default=800)
    ap.add_argument("--chunk-minutes", type=float, default=30,
                    help="checkpoint interval for the single-worker path -- a "
                         "crash mid-chunk loses at most the one job that was "
                         "in flight, not the whole chunk (each job result is "
                         "written atomically on completion). Ignored when "
                         "--workers > 1 (each shard just runs to completion; "
                         "a crash there still loses at most one in-flight job, "
                         "same atomic-write guarantee, just no periodic "
                         "resume-check in between).")
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many sharded subprocesses per manifest "
                         "instead of one serial process (see module docstring)")
    ap.add_argument("--skip-probe", action="store_true",
                    help="skip the Gate-1 timing table (useful on repeat runs)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[run_local] No CUDA device visible to PyTorch. Before "
              "re-running:")
        print("  1. Run `nvidia-smi` in this same terminal and confirm it "
              "lists your GPU (if it errors, the NVIDIA driver service "
              "needs a restart or the machine needs a reboot).")
        print("  2. Confirm torch has a CUDA build: "
              "`python -c \"import torch; print(torch.__version__)\"` "
              "should print something like `2.x.x+cu124`, not `+cpu`.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"[run_local] GPU: {torch.cuda.get_device_name(device)}, "
         f"{props.total_memory / 1e9:.1f} GB VRAM")
    print(f"[run_local] results directory: {rm.RESULTS_ROOT}")

    n_cleaned = rm.clean_partial_files(rm.RESULTS_ROOT)
    if n_cleaned:
        print(f"[run_local] cleaned {n_cleaned} leftover .tmp file(s) from a "
             f"previously-interrupted write.")

    if not args.skip_probe:
        probe_n_x = args.n_x if args.n_x is not None else 1024
        rm.probe("all", n_x=probe_n_x, n_iters=args.n_iters)
        print(f"\n[run_local] probe used n_x={probe_n_x} for every tier "
              f"(one uniform scale, for a single comparable table); the "
              f"real run below uses n_x={TIER_N_X['M1']} for M1/M2 and "
              f"n_x={TIER_N_X['S1']} for S1/S2 by default, so the actual "
              f"S1/S2 wall-clock will be lower than this table implies. "
              f"On your own machine there's no compute cost to a longer "
              f"run, so this script proceeds regardless of the Gate-1 "
              f"verdict -- it's still worth reading, an RTX 3050 laptop "
              f"GPU is meaningfully slower than the T4 the {rm.GATE1_HOURS:.0f}h "
              f"threshold was calibrated against. Re-run with --skip-probe "
              f"once you've seen this and don't need it again.\n")

    for name in args.manifests:
        n_x = args.n_x if args.n_x is not None else TIER_N_X[name]
        print(f"\n{'=' * 70}\n[run_local] starting manifest {name!r} "
             f"(n_x={n_x}, workers={args.workers})\n{'=' * 70}")
        if args.workers > 1:
            _run_sharded(name, n_x, args.n_iters, args.workers)
            print(f"[run_local] manifest {name!r} done (sharded across "
                 f"{args.workers} workers).")
        else:
            status = rm.run_manifest_until_complete(
                name, chunk_minutes=args.chunk_minutes,
                n_x=n_x, n_iters=args.n_iters)
            print(f"[run_local] manifest {name!r} done: "
                 f"{status['n_done']}/{status['n_total']} jobs.")

    print("\n[run_local] all requested manifests complete. Next: "
          "python analysis/aggregate.py, then python -m figures.make_all, "
          "then python scripts/make_numbers_tex.py.")


if __name__ == "__main__":
    main()
