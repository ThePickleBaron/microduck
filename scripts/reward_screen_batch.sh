#!/usr/bin/env bash
# Reward-weight investigation, unattended (2026-10-11). About 3.5 hours:
#   0. Wait for scripts/c6v2_batch.sh to finish (safe to start now)
#   1. Budget curves for C2 and C4 (CPU, ~10 min; the rest is cached)
#   2. Reward terms over training, from TensorBoard logs (CPU, ~1 min)
#   3. Reward-weight screen: 20 short runs (GPU, ~9 min each, ~3 h)
#   4. Evaluate and analyze the screen (CPU, ~10 min)
#   Results are copied to OneDrive after steps 2 and 4.
#
#   cd ~/microduck-pretraining && git pull && bash scripts/reward_screen_batch.sh
#
# Leave the terminal open. Log: logs/reward_screen_batch_<time>.log.
# Safe to re-run: finished screening runs and cached simulations are reused.
set -uo pipefail
cd "$(dirname "$0")/.."
ONEDRIVE=/mnt/c/Users/jsmcl/OneDrive/Documents/Claude/Projects/microduck/results
mkdir -p logs
LOG="logs/reward_screen_batch_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $(date '+%H:%M:%S')  $* ==="; }
FAILED=()
run() { "$@" || { echo "!!! failed ($?): $*"; FAILED+=("$*"); return 1; }; }

copy_results() {
  step "copy results to OneDrive"
  if [ -d "$(dirname "$ONEDRIVE")" ]; then
    mkdir -p "$ONEDRIVE/stress" "$ONEDRIVE/reward_terms" "$ONEDRIVE/reward_screen"
    cp -f results/stress/*.md results/stress/*.csv results/stress/*.png "$ONEDRIVE/stress"/ 2>/dev/null
    cp -f results/reward_terms/* "$ONEDRIVE/reward_terms"/ 2>/dev/null
    cp -f results/reward_screen/* "$ONEDRIVE/reward_screen"/ 2>/dev/null
    cp -f "$LOG" "$ONEDRIVE"/ 2>/dev/null
    echo "copied to $ONEDRIVE"
  else
    echo "OneDrive folder not found; results stay in $(pwd)/results"
  fi
}

step "start ($(git log --oneline -1))"
if pgrep -f "c6v2_batch\.sh" >/dev/null; then
  step "0/4 waiting for the C6v2 batch to finish (checking every minute)"
  while pgrep -f "c6v2_batch\.sh" >/dev/null; do sleep 60; done
fi

step "1/4 budget curves, all conditions (only C2 and C4 are new)"
run uv run python scripts/stress_tests.py budget --conditions c2 c3 c4 c5 c3x c6 c6v2

step "2/4 reward terms over training"
run uv run python scripts/reward_terms.py --conditions c3 c2 c4 c3x c6 c6v2
copy_results

step "3/4 reward-weight screen: training (20 runs)"
uv run python scripts/reward_screen.py design
run uv run python scripts/reward_screen.py train

step "4/4 reward-weight screen: evaluate and analyze"
run uv run python scripts/reward_screen.py evaluate && run uv run python scripts/reward_screen.py analyze
copy_results

step "done"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All steps succeeded."
else
  echo "Failed steps:"; printf '  %s\n' "${FAILED[@]}"
fi
echo "Log: $LOG"
