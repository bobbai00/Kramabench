#!/bin/bash
# CA-guided replicate study: mini guided code agent 1k+5k, 5 samples each
# (base=rep0 + Replicate1-4). 10 arms x 104 = 1040 tasks. Own P4 pool —
# code-agent never touches the Texera engine, safe concurrent with orch7.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch8_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
mapfile -t ARMS < "$D/ca_guided.txt"
log "CA-GUIDED run start — 10 arms x 104 = 1040 tasks, own P4 pool, NO retries"
.venv/bin/python $D/failed_pairs.py --all "${ARMS[@]}" | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolca_$0__$2.log" 2>&1
'
log "pool done, reeval + score per arm:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ca_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCH8 ALL DONE ############"
