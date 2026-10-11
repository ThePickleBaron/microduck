#!/usr/bin/env bash
# Unattended C6v2 batch (2026-10-11). About 3 hours on the desktop:
#   1. Which careful-site moves stall, for C3x, C6 and steady-cam v2   (~10 min, CPU)
#   2. Train C6v2 seeds 1-3                                             (~45 min each, GPU)
#   3. Export, evaluate, seed spread, stress tests, deployment check     (~30-45 min, CPU)
#   4. Copy results to OneDrive for Claude to read
#
#   cd ~/microduck-pretraining && git pull && bash scripts/c6v2_batch.sh
#
# Leave the terminal open. Everything is logged to logs/c6v2_batch_<time>.log.
# Safe to re-run: finished seeds are skipped, evaluations and cached stress
# runs are reused.
set -uo pipefail
cd "$(dirname "$0")/.."
ONEDRIVE=/mnt/c/Users/jsmcl/OneDrive/Documents/Claude/Projects/microduck/results
mkdir -p logs
LOG="logs/c6v2_batch_$(date +%Y%m%d_%H%M).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $(date '+%H:%M:%S')  $* ==="; }
FAILED=()
run() { "$@" || { echo "!!! failed ($?): $*"; FAILED+=("$*"); return 1; }; }

copy_results() {
  step "copy results to OneDrive"
  if [ -d "$(dirname "$ONEDRIVE")" ]; then
    mkdir -p "$ONEDRIVE/stress/careful"
    cp -f results/*.json results/*.md results/*.csv results/*.png "$ONEDRIVE"/ 2>/dev/null
    cp -f results/stress/*.md results/stress/*.csv results/stress/*.png "$ONEDRIVE/stress"/ 2>/dev/null
    cp -f results/stress/careful/*.json "$ONEDRIVE/stress/careful"/ 2>/dev/null
    cp -f "$LOG" "$ONEDRIVE"/ 2>/dev/null
    echo "copied to $ONEDRIVE"
  else
    echo "OneDrive folder not found; results stay in $(pwd)/results"
  fi
}

step "start ($(git log --oneline -1))"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

# 1. Per-move stall breakdown on the careful site (re-simulates once: older
#    cached runs have no per-move data). Copied right away so Claude can read
#    it while training runs.
step "1/4 careful-site breakdown: c3x, c6, steadycam_v2"
run uv run python scripts/stress_tests.py careful --conditions c3x c6 steadycam_v2
copy_results

# 2. Train C6v2, skipping seeds that already finished (final checkpoint saved).
for seed in 1 2 3; do
  if compgen -G "logs/rsl_rl/c6v2_careful/*_c6v2_seed${seed}/model_1500.pt" >/dev/null \
     || compgen -G "logs/rsl_rl/c6v2_careful/*_c6v2_seed${seed}/model_149[0-9].pt" >/dev/null; then
    step "2/4 C6v2 seed $seed already trained; skipping"
    continue
  fi
  step "2/4 train C6v2 seed $seed (TensorBoard: --logdir logs/rsl_rl/c6v2_careful)"
  run uv run python scripts/train_followup.py c6v2 --seeds "$seed"
done

# 3. Evaluate.
step "3/4 export"
run uv run python scripts/train_followup.py --export
step "3/4 evaluate (nominal + held-out; existing results are skipped)"
run uv run python scripts/evaluate_all.py
step "3/4 seed spread"
run uv run python scripts/seed_spread.py --md results/seed_spread.md
step "3/4 stress tests: careful, envelope, budget (c6 and c3x mostly cached)"
run uv run python scripts/stress_tests.py all --conditions c6v2 c6 c3x
step "3/4 deployment check"
run uv run python scripts/check_policies.py --manifest

# 4. Hand-off.
copy_results
step "done"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All steps succeeded."
else
  echo "Failed steps:"; printf '  %s\n' "${FAILED[@]}"
fi
echo "Log: $LOG"
