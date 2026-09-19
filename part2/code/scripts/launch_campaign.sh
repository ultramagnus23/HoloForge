#!/usr/bin/env bash
# Launch N_SHARDS parallel workers for the post-freeze campaign (run from part2/code).
# Usage: scripts/launch_campaign.sh <shard-ids...>   e.g.  scripts/launch_campaign.sh 0 1 2 3
# Every worker runs the same priority queue over its own balanced shard, so
# workers can be added later (the shard count N_SHARDS stays fixed).
# Queue (priority order): main grid coarse pass -> compute-matched arm -> robustness
# -> physics ablation -> slant study -> main grid fill-in -> S2 (reduced, then full).
set -euo pipefail
N_SHARDS=6
QUEUE="M1A,M2,S4,S5,S1,S8,M1B,S2R,S2"
HOURS="${HOURS:-70}"
for i in "$@"; do
  PYTHONUTF8=1 nohup python -u -m experiments.overnight_runner \
      --manifests "$QUEUE" --hours "$HOURS" --n-x 1024 \
      --shard "$i/$N_SHARDS" --balanced-shard > "campaign_w$i.log" 2>&1 &
  echo "launched shard $i/$N_SHARDS (pid $!)"
done
