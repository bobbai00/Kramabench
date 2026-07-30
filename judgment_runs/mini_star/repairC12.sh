#!/bin/bash
# A7/C12 REPAIR: 150 runs whose answer is the literal "(empty response)" —
# HTTP 429 rate-limiting from the LLM gateway during the tail of the P18 pool
# (14 x 429 in the logs; gateway + quota verified healthy afterwards).
# Concentrated in C12 reps 2-3 (39 + 103); A7 lost only 7 of 312.
#
# Keyed on the "(empty response)" MARKER, not on the log: a 429-killed run still
# writes "Total score is: 0.0", so resume-skip would treat it as complete — the
# same trap that hid the earlier service-restart damage.
#
# P6, deliberately well below the P18 that tripped the rate limit.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a $D/repairC12_progress.log; }
: > $D/repairC12_progress.log
log "A7/C12 repair start — $(wc -l < /tmp/c12_repair.txt) runs at P6 (429-throttled at P18)"
cat /tmp/c12_repair.txt | xargs -P 6 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolC12_$0__$2.log" 2>&1
'
log "repair pass done; re-evaluating"
for A in $(awk '{print $1}' /tmp/c12_repair.txt | sort -u); do
  ./kb.py reeval --sut "$A" >> $D/repairC12_${A}.log 2>&1
  log "  $A -> $(.venv/bin/python compute_scores.py --sut "$A" 2>/dev/null | awk '/OVERALL/{print $2}')"
done
log "############ REPAIRC12 ALL DONE ############"
