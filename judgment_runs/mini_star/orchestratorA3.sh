#!/bin/bash
# REP EXPANSION: A1RolePolicy vs A0Control at 8 reps/arm to resolve the +5.0pt
# question (3 reps can't split it from the +-4-5pt run-level floor).
# Reps 1-3 exist from the 14:25 run (sha 4af1e98da); this adds A1 reps 4-8 and
# A0 reps 5-8 on sha 9d60d01dc (render byte-identical for these configs — golden
# parity; A0ControlReplicate4 is the cross-sha sentinel).
# GATED: waits for the A2/A3 pool orchestrator (pid passed as $1) to exit before
# starting, so total engine concurrency never exceeds P5. Gate is BY PID —
# pattern-matching a process name has previously matched this script's own shell.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orchA3_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

GATE_PID="${1:-}"
if [[ -n "$GATE_PID" ]]; then
  log "gating on A2/A3 orchestrator pid $GATE_PID ..."
  while kill -0 "$GATE_PID" 2>/dev/null; do sleep 60; done
  log "gate open — A2/A3 pool drained."
fi

ARMS=(
  DataflowSystemGPT5MiniA1RolePolicyReplicate4
  DataflowSystemGPT5MiniA1RolePolicyReplicate5
  DataflowSystemGPT5MiniA1RolePolicyReplicate6
  DataflowSystemGPT5MiniA1RolePolicyReplicate7
  DataflowSystemGPT5MiniA1RolePolicyReplicate8
  DataflowSystemGPT5MiniA0ControlReplicate5
  DataflowSystemGPT5MiniA0ControlReplicate6
  DataflowSystemGPT5MiniA0ControlReplicate7
  DataflowSystemGPT5MiniA0ControlReplicate8
)
TASKS=$(cat $D/subset_hard.txt)
log "A1/A0 rep expansion start — 9 arms x 20 hard tasks = 180 runs, P5, :3002 @ 9d60d01dc, no retries."

for A in "${ARMS[@]}"; do
  for T in $TASKS; do
    L="judgment_runs/mini_star/poolA3_${A}__${T}.log"
    grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
  done
done | xargs -P 5 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolA3_$0__$2.log" 2>&1
'
log "main pass done — repair pass"
for A in "${ARMS[@]}"; do
  for T in $TASKS; do
    L="judgment_runs/mini_star/poolA3_${A}__${T}.log"
    grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
  done
done | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolA3_$0__$2.log" 2>&1
'
log "repair pass done. reeval:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleA3_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHA3 ALL DONE ############"
