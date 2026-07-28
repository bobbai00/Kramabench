#!/bin/bash
# Full 104-task run + 2 retry rounds per arm, all 4 mini arms.
# Sequential arms, KB_MAX_PARALLEL=4 (engine-contention safe). Watchdog 8min.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
export KB_MAX_PARALLEL=4
PROG=judgment_runs/mini_star/fullrun_progress.log
: > "$PROG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

ARMS=(DataflowSystemGPT5MiniDelta1kSchemaOnly \
      DataflowSystemGPT5MiniDelta5kSchemaOnly \
      DataflowSystemGPT5MiniDeltaStats1kD2 \
      DataflowSystemGPT5MiniLatest1kCodeInSnap)

overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

for ARM in "${ARMS[@]}"; do
  log "===== ARM $ARM ====="
  log "phase FULL104 start"
  ./kb.py run --sut "$ARM" --parallel --watchdog-min 8 >> judgment_runs/mini_star/full_${ARM}.log 2>&1
  log "phase FULL104 done  -> OVERALL $(overall "$ARM")"
  for R in 1 2; do
    log "phase RETRY$R start"
    ./kb.py rerun-failed --sut "$ARM" --all-failed --parallel --watchdog-min 8 >> judgment_runs/mini_star/full_${ARM}.log 2>&1
    log "phase RETRY$R done  -> OVERALL $(overall "$ARM")"
  done
  # rebuild bulk cache from freshest scratch + rescore (canonical)
  ./kb.py reeval --sut "$ARM" >> judgment_runs/mini_star/full_${ARM}.log 2>&1
  log "ARM $ARM FINAL -> OVERALL $(overall "$ARM")"
done
log "############ ALL DONE ############"
