#!/usr/bin/env bash
# CPU-only launcher for the S1X, M1C, M2R and S2R queue (run from part2/code).
# Used when CUDA is unavailable: a MIL job at n_x=1024 measures ~0.65 s/iter on
# 8 CPU threads here, so the remaining tiers are CPU-feasible.
# Usage: scripts/launch_cpu_queue.sh 0 1 2 3
# Queue (priority order): S1X mechanism factorial -> M1C (K=3.927 main-grid cell) -> M2R reduced compute-matched
# arm -> S2R one-K parameter sensitivity.
set -euo pipefail
N_SHARDS="${N_SHARDS:-4}"
QUEUE="${QUEUE:-S1X,M1C,M2R,S2R}"
HOURS="${HOURS:-30}"
THREADS="${THREADS:-4}"
for i in "$@"; do
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" PYTHONUTF8=1 nohup python -u -m experiments.overnight_runner \
      --manifests "$QUEUE" --hours "$HOURS" --n-x 1024 --allow-cpu \
      --shard "$i/$N_SHARDS" --balanced-shard > "cpu_w$i.log" 2>&1 &
  echo "launched shard $i/$N_SHARDS (pid $!)"
done
