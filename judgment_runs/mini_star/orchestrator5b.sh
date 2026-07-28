#!/bin/bash
# Re-run the quota-corrupted replicate tasks (315 pairs) after credit refill.
# Global P4 pool, single-shot (no retries, same protocol), then reeval the 4
# affected arms so C5 gets its 3rd sample and C6 its 2nd/3rd.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch5b_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

AFFECTED=(DataflowSystemGPT5MiniDeltaStats2kD2Replicate1 \
          DataflowSystemGPT5MiniDeltaStats2kD2Replicate2 \
          DataflowSystemGPT5MiniLatestStats1kD2Replicate1 \
          DataflowSystemGPT5MiniLatestStats1kD2Replicate2)

log "RECOVERY re-run start — $(wc -l < $D/corrupt_pairs.txt) corrupt pairs, global P4"
cat $D/corrupt_pairs.txt | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolrep_$0__$2.log" 2>&1
'
log "pool done, reeval + score affected arms:"
for A in "${AFFECTED[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/rep_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCH5B ALL DONE ############"
