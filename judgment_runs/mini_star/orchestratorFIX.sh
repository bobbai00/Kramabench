#!/bin/bash
# REPAIR pass: re-run every 5.6 run whose response.txt starts with "Error:" —
# infra-killed runs (engine recycles killing in-flight agent calls) that were
# scored 0.0 and skipped by resume-skip. See emit_pending comments in
# orchestratorTC5.sh and HANDOFF 4.7. Two passes at P12/P6, then serial reeval
# of every arm that was actually touched.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orchFIX_progress.log
touch "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
mapfile -t ARMS < "$D/ruleFIX.txt"
emit_pending(){
  for A in "${ARMS[@]}"; do
    for R in system_scratch/$A/*/response.txt; do
      [ -f "$R" ] || continue
      if head -c 6 "$R" 2>/dev/null | grep -q '^Error:'; then
        T=$(basename "$(dirname "$R")")
        echo "$A ${T%%-*} $T"
      fi
    done
  done
}
TOUCHED=$(emit_pending | awk '{print $1}' | sort -u)
log "FIX pass: $(emit_pending | wc -l) poisoned runs across $(echo "$TOUCHED" | wc -l) arms"
run_pass(){
  emit_pending | xargs -P "$1" -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/poolFIX_$0__$2.log" 2>&1
  '
}
run_pass 3
log "pass 1 done; remaining errors: $(emit_pending | wc -l)"
run_pass 3
log "pass 2 done; remaining errors: $(emit_pending | wc -l)"
log "reeval (serial) of touched arms:"
for A in $TOUCHED; do
  timeout 1800 ./kb.py reeval --sut "$A" >> $D/ruleFIX_${A}.log 2>&1
  log "  $A rc=$?"
done
log "############ ORCHFIX ALL DONE ############"
