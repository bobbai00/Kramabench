#!/bin/bash
# Replicate0: clean single-shot re-run of the 7 base mini arms (anchor+C1-C6),
# to recover clean round0-type traces (base arms' round0 traces were overwritten
# by their recovery rounds). Global P4 pool, single round0, NO retries.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch6_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
PY=.venv/bin/python
mapfile -t ARMS < "$D/replicate0.txt"

log "REPLICATE0 run start — 7 arms x 104 = 728 task-runs, global P4, NO retries"
$PY $D/failed_pairs.py --all "${ARMS[@]}" | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/pool0_$0__$2.log" 2>&1
'
log "pool done, reeval + score per arm:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/rep0_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCH6 ALL DONE ############"
