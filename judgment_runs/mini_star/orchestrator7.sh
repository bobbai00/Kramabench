#!/bin/bash
# Rep3/Rep4 for anchor+C1-C6 (14 arms) + C7 Delta2kSchemaOnly Rep0-4 (5 arms).
# 19 arms x 104 = 1976 task-runs. Global P4 pool, single round0, NO retries.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch7_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
PY=.venv/bin/python
mapfile -t ARMS < "$D/replicate34.txt"

log "REP3/4 + C7 run start — 19 arms x 104 = 1976 task-runs, global P4, NO retries"
$PY $D/failed_pairs.py --all "${ARMS[@]}" | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/pool34_$0__$2.log" 2>&1
'
log "pool done, reeval + score per arm:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/rep34_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCH7 ALL DONE ############"
