#!/bin/bash
# Variance replicates (gpt-5-mini only): 14 arms (anchor+C1-C6 x Rep1/Rep2).
# Single round0, NO retries (raw single-shot for randomness floor).
# Global 4-wide task pool across all 14 arms => engine saturated, max efficiency.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch5_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
PY=.venv/bin/python
mapfile -t ARMS < "$D/replicates.txt"

run_pool(){
  xargs -P 4 -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/poolrep_$0__$2.log" 2>&1
  '
}

log "REPLICATE run start — 14 arms x 104 = 1456 task-runs, global P4, NO retries"
$PY $D/failed_pairs.py --all "${ARMS[@]}" | run_pool
log "pool done, reeval + score per arm:"
for A in "${ARMS[@]}"; do
  $PY -c "import sys" # noop keepalive
  ./kb.py reeval --sut "$A" >> $D/rep_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCH5 ALL DONE ############"
