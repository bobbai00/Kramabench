#!/bin/bash
# gpt-5.2 star: full 104 + 2 retries per arm, 4 arms (C1/C2/C3 mirror of mini).
# Sequential arms, KB_MAX_PARALLEL=4, watchdog 8min.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
export KB_MAX_PARALLEL=4
PROG=judgment_runs/mini_star/gpt52_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }

ARMS=(DataflowSystemGPT52Delta1kSchemaOnly \
      DataflowSystemGPT52Delta5kSchemaOnly \
      DataflowSystemGPT52DeltaStats1kD2 \
      DataflowSystemGPT52Latest1kCodeInSnap)

overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

for ARM in "${ARMS[@]}"; do
  log "===== ARM $ARM ====="
  log "phase FULL104 start"
  ./kb.py run --sut "$ARM" --parallel --watchdog-min 8 >> judgment_runs/mini_star/gpt52_${ARM}.log 2>&1
  log "phase FULL104 done  -> OVERALL $(overall "$ARM")"
  for R in 1 2; do
    log "phase RETRY$R start"
    ./kb.py rerun-failed --sut "$ARM" --all-failed --parallel --watchdog-min 8 >> judgment_runs/mini_star/gpt52_${ARM}.log 2>&1
    log "phase RETRY$R done  -> OVERALL $(overall "$ARM")"
  done
  ./kb.py reeval --sut "$ARM" >> judgment_runs/mini_star/gpt52_${ARM}.log 2>&1
  log "ARM $ARM FINAL -> OVERALL $(overall "$ARM")"
done
log "############ GPT52 ALL DONE ############"
