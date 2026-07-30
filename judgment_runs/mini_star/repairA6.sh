#!/bin/bash
# A6 REPAIR: 58 runs killed mid-flight when :3002 was restarted to deploy
# b9fd6d4f1 while the A6 pool was still draining (133/160). Symptom in the log:
# "Connection to remote host was lost" + "Connection refused", and CRITICALLY the
# run still logs "Total score is: 0.0" — so the orchestrators' resume-skip
# (grep 'Total score') treats it as complete. Repair is therefore keyed on a
# MISSING response.txt, not on the log.
# P6 so it coexists with the C9 pool at P18 without exceeding the ~P26 that ran
# clean earlier tonight. DO NOT restart :3002 while this runs.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a $D/repairA6_progress.log; }
: > $D/repairA6_progress.log
log "A6 repair start — $(wc -l < /tmp/a6_repair.txt) runs at P6, :3002 @ b9fd6d4f1"
cat /tmp/a6_repair.txt | xargs -P 6 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolA6_$0__$2.log" 2>&1
'
log "repair pass done; re-evaluating"
for A in $(awk '{print $1}' /tmp/a6_repair.txt | sort -u); do
  ./kb.py reeval --sut "$A" >> $D/repairA6_${A}.log 2>&1
  log "  $A reevaled"
done
log "############ REPAIRA6 ALL DONE ############"
