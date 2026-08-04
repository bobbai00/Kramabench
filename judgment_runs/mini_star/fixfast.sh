#!/bin/bash
# Split-lane repair: memory-light tasks (LLM-bound wildfire/legal/etc) run WIDE at
# P14 while memory-heavy environment tasks run NARROW at P6, concurrently. One P8
# queue served both badly: env workers hit 5 GB each (so wide is unsafe), while the
# light tasks idle in LLM-wait (so narrow wastes slots).
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/fixfast_progress.log
touch "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
mapfile -t ARMS < "$D/ruleFIX.txt"
emit(){ # $1 = quick (everything except wildfire-hard-19) | w19 | all
  for A in "${ARMS[@]}"; do
    for R in system_scratch/$A/*/response.txt; do
      [ -f "$R" ] || continue
      head -c 6 "$R" 2>/dev/null | grep -q '^Error:' || continue
      T=$(basename "$(dirname "$R")")
      case "$1" in
        w19)   [ "$T" = "wildfire-hard-19" ] && echo "$A ${T%%-*} $T";;
        quick) [ "$T" != "wildfire-hard-19" ] && echo "$A ${T%%-*} $T";;
        all)   echo "$A ${T%%-*} $T";;
      esac
    done
  done
}
runp(){ emit "$1" | xargs -P "$2" -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolFIX_$0__$2.log" 2>&1'; }
log "fixfast v2: quick=$(emit quick|wc -l) @P3 first, then w19=$(emit w19|wc -l) @P4"
runp quick 3
log "quick lane done; remaining quick=$(emit quick|wc -l). QUICK-LANE-CLEAR marker for early reeval."
runp w19 4
log "w19 pass 1 done; remaining=$(emit all|wc -l)"
runp all 3
log "sweep done; remaining=$(emit all|wc -l)"
log "reeval (serial) of touched arms:"
for A in $(ls $D/poolFIX_* 2>/dev/null | sed 's/.*poolFIX_//;s/__.*//' | sort -u); do
  timeout 1800 ./kb.py reeval --sut "$A" >> $D/ruleFIX_${A}.log 2>&1
  log "  $A rc=$?"
done
log "############ FIXFAST ALL DONE ############"
