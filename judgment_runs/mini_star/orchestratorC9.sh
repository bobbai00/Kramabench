#!/bin/bash
# C9 / C10 / C11 — per-operator CHAR budget at the raw-data boundary, FULL 104.
#   C9  LATEST+code : sources 5k + stats, every downstream op 1k + no stats
#   C10 DELTA       : same split (char-budget leg binds on DELTA event renders)
#   C11 LATEST+code : 5k + stats for ALL ops (no per-op policy) — the reference
# 9 arms (3 SUTs x 3 reps) x 104 tasks = 936 runs. All on the CURRENT sha; the
# pre-existing LatestStats5kD2Code reps predate the stats-bound + provenance
# commits and are a different vintage, so C11 is re-run rather than reused.
#
# GATED by PID on the A6 pool (pattern-matching a process name has previously
# matched this script's own shell — always gate by PID).
# P12: the engine is exclusive tonight and P8 produced 0 instant-fails over 75
# runs, so the old "P6 causes instant-fails" note is stale. If instant-fails
# appear, kill by PID and relaunch at P8 — resume-skip makes that nearly free.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orchC9_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

GATE_PID="${1:-}"
if [[ -n "$GATE_PID" ]]; then
  log "gating on pid $GATE_PID ..."
  while kill -0 "$GATE_PID" 2>/dev/null; do sleep 60; done
  log "gate open."
fi

mapfile -t ARMS < "$D/ruleC9.txt"
mapfile -t TASKS < "$D/tasks_full104.txt"
log "C9/C10/C11 start — ${#ARMS[@]} arms x ${#TASKS[@]} tasks = $(( ${#ARMS[@]} * ${#TASKS[@]} )) runs, P12, :3002 @ b9fd6d4f1, no retries."

emit_pending(){
  for A in "${ARMS[@]}"; do
    for T in "${TASKS[@]}"; do
      L="judgment_runs/mini_star/poolC9_${A}__${T}.log"
      grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
    done
  done
}

emit_pending | xargs -P 18 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolC9_$0__$2.log" 2>&1
'
log "main pass done — repair pass"
emit_pending | xargs -P 12 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolC9_$0__$2.log" 2>&1
'
log "repair pass done. reeval + score (full 104):"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleC9_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHC9 ALL DONE ############"
