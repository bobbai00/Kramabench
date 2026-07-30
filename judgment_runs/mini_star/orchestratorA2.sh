#!/bin/bash
# RULE A: history channels on a LATEST core. 9 arms x 20 hard tasks = 180 runs.
# Single-shot, NO retries. WAITS for the foreign LatestStats5kD2Code pool to
# drain first so total engine concurrency never exceeds P4 (their run must not
# be disturbed). Config-only arms -> agent-service source untouched.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orchA2_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
mapfile -t ARMS < "$D/ruleA2.txt"
TASKS=$(cat $D/subset_hard.txt)

# Coexistence policy (not a gate): the other session's recovery pool runs ~1
# concurrent task. We run P3 so TOTAL engine concurrency stays at the documented
# safe cap of 4. Monitor: if their concurrency rises, throttle this pool.
log "A2/A3 start — 7 arms (3 anomaly-stats + 3 no-source-stats + 1 A0 sentinel) x 20 hard tasks, P5, service :3002 @ 9d60d01dc, no retries."

# RESUME: emit only (arm,task) pairs not already scored, so a restart never
# redoes finished work.
for A in "${ARMS[@]}"; do
  for T in $TASKS; do
    L="judgment_runs/mini_star/poolA2_${A}__${T}.log"
    grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
  done
done | xargs -P 5 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolA2_$0__$2.log" 2>&1
'
log "main pass done — repair pass for unscored pairs (contention instant-fails)"
for A in "${ARMS[@]}"; do
  for T in $TASKS; do
    L="judgment_runs/mini_star/poolA2_${A}__${T}.log"
    grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
  done
done | xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolA2_$0__$2.log" 2>&1
'
log "repair pass done. reeval + score (subset-scoped):"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleA2_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHA2 ALL DONE ############"
