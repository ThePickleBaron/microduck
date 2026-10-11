#!/usr/bin/env bash
# Two-weight response surface on C6v2, unattended (2026-10-11). About 3 hours:
#   18 short runs (GPU, ~9 min each), then evaluation and analysis (CPU, ~10 min),
#   then results copied to OneDrive.
#
#   cd ~/microduck-pretraining && git pull && bash scripts/reward_surface_batch.sh
#
# Leave the terminal open. Log: logs/reward_surface_batch_<time>.log.
# Safe to re-run: finished runs and cached simulations are reused.
set -uo pipefail
cd "$(dirname "$0")/.."
ONEDRIVE=/mnt/c/Users/jsmcl/OneDrive/Documents/Claude/Projects/microduck/results
mkdir -p logs
LOG="logs/reward_surface_batch_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $(date '+%H:%M:%S')  $* ==="; }
FAILED=()
run() { "$@" || { echo "!!! failed ($?): $*"; FAILED+=("$*"); return 1; }; }

step "start ($(git log --oneline -1))"
uv run python scripts/reward_surface.py design

step "1/2 training (18 runs; TensorBoard: --logdir logs/rsl_rl/reward_surface)"
run uv run python scripts/reward_surface.py train

step "2/2 evaluate and analyze"
run uv run python scripts/reward_surface.py evaluate && run uv run python scripts/reward_surface.py analyze

step "copy results to OneDrive"
if [ -d "$(dirname "$ONEDRIVE")" ]; then
  mkdir -p "$ONEDRIVE/reward_surface"
  cp -f results/reward_surface/* "$ONEDRIVE/reward_surface"/ 2>/dev/null
  cp -f "$LOG" "$ONEDRIVE"/ 2>/dev/null
  echo "copied to $ONEDRIVE/reward_surface"
else
  echo "OneDrive folder not found; results stay in $(pwd)/results/reward_surface"
fi

step "done"
if [ ${#FAILED[@]} -eq 0 ]; then echo "All steps succeeded."; else echo "Failed steps:"; printf '  %s\n' "${FAILED[@]}"; fi
echo "Log: $LOG"
