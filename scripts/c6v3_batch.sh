#!/usr/bin/env bash
# C6v3 confirmation run, unattended (2026-10-12). About 6 hours:
#   Part A (~3 h): C6v3 = C6v2 with turning tracking x2, seeds 1-3, then the full
#                  evaluation; results copied to OneDrive as soon as it is done.
#   Part B (~3 h): C6v3t3 = turning x3, seeds 1-3, same evaluation, copied again.
#
#   cd ~/microduck-pretraining && git pull && bash scripts/c6v3_batch.sh
#
# Leave the terminal open. Log: logs/c6v3_batch_<time>.log. Safe to re-run:
# finished seeds are skipped, evaluations and cached simulations are reused.
set -uo pipefail
cd "$(dirname "$0")/.."
ONEDRIVE=/mnt/c/Users/jsmcl/OneDrive/Documents/Claude/Projects/microduck/results
mkdir -p logs
LOG="logs/c6v3_batch_$(date +%Y%m%d_%H%M).log"
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

train_condition() {  # $1 = condition key
  local key=$1 seed
  for seed in 1 2 3; do
    if compgen -G "logs/rsl_rl/${key}_careful/*_${key}_seed${seed}/model_1500.pt" >/dev/null \
       || compgen -G "logs/rsl_rl/${key}_careful/*_${key}_seed${seed}/model_149[0-9].pt" >/dev/null; then
      step "${key} seed ${seed} already trained; skipping"
      continue
    fi
    step "train ${key} seed ${seed} (TensorBoard: --logdir logs/rsl_rl/${key}_careful)"
    run uv run python scripts/train_followup.py "$key" --seeds "$seed"
  done
}

evaluate_conditions() {  # $@ = condition keys for the stress tests
  step "export"
  run uv run python scripts/train_followup.py --export
  step "evaluate (nominal + held-out; existing results are skipped)"
  run uv run python scripts/evaluate_all.py
  step "seed spread"
  run uv run python scripts/seed_spread.py --md results/seed_spread.md
  step "stress tests: careful, envelope, budget"
  run uv run python scripts/stress_tests.py all --conditions "$@"
  step "deployment check (exit code 1 only means some policy failed; C4 always does)"
  uv run python scripts/check_policies.py --manifest || echo "(see results/policy_check.md)"
}

step "start ($(git log --oneline -1))"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

step "Part A: C6v3 (turning x2)"
train_condition c6v3
evaluate_conditions c6v3 c6v2 c3x
copy_results

step "Part B: C6v3t3 (turning x3)"
train_condition c6v3t3
evaluate_conditions c6v3t3 c6v3 c6v2 c3x
copy_results

step "done"
if [ ${#FAILED[@]} -eq 0 ]; then echo "All steps succeeded."; else echo "Failed steps:"; printf '  %s\n' "${FAILED[@]}"; fi
echo "Log: $LOG"
