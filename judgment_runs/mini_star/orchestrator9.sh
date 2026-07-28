#!/bin/bash
# C8 = Latest5k + code-in-snapshot, 5 single-shot reps (Replicate0-4).
# 5 arms x 104 = 520 tasks, global P4, NO retries, reeval+score each.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch9_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
mapfile -t ARMS < "$D/c8.txt"
log "C8 run start — 5 arms x 104 = 520 tasks, global P4, NO retries"
.venv/bin/python $D/failed_pairs.py --all "${ARMS[@]}" | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/pool8_$0__$2.log" 2>&1
'
log "pool done, reeval + score per arm:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/c8_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCH9 ALL DONE ############"
