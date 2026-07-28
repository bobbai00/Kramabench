#!/bin/bash
# C4/C5/C6 stats-star, both models — MAX-EFFICIENCY variant.
# Global task-level pool (4-wide, engine-safe) across all 6 arms simultaneously,
# so all 4 engine slots stay saturated (no per-arm/batch idle). round0 + 2 retries.
# Starts as soon as gpt-5.2 C3 RUNS finish (orch3 "C3 FINAL"); orch3's judge
# (OpenAI-only) overlaps freely.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch4_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }
PY=.venv/bin/python

ARMS=(DataflowSystemGPT5MiniDeltaStats5kD2 \
      DataflowSystemGPT5MiniDeltaStats2kD2 \
      DataflowSystemGPT5MiniLatestStats1kD2 \
      DataflowSystemGPT52DeltaStats5kD2 \
      DataflowSystemGPT52DeltaStats2kD2 \
      DataflowSystemGPT52LatestStats1kD2)

# run one worklist ("ARM WL TASK" lines on stdin) through a 4-wide pool
run_pool(){
  xargs -P 4 -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/pool_$0__$2.log" 2>&1
  '
}

# 1. wait for gpt-5.2 C3 RUNS to finish (engine free). orch3 logs "C3 FINAL".
log "waiting for gpt-5.2 C3 runs to finish (orch3 'C3 FINAL')..."
while ! grep -q 'C3 FINAL' $D/orch3_progress.log 2>/dev/null; do
  if ! pgrep -f '[o]rchestrator3.sh' >/dev/null && ! grep -q 'C3 FINAL' $D/orch3_progress.log 2>/dev/null; then
    log "WARN orch3 gone before C3 FINAL; proceeding"; break
  fi
  sleep 30
done
log "C3 runs done, engine free. Starting global 4-wide pool for C4/C5/C6 (6 arms)."

# 2. round 0 — every (arm,task), one global pool
log "ROUND0 start (6 arms x 104 = 624 task-runs, P4 saturated)"
$PY $D/failed_pairs.py --all "${ARMS[@]}" | run_pool
for A in "${ARMS[@]}"; do ./kb.py reeval --sut "$A" >> $D/c456_${A}.log 2>&1; done
log "ROUND0 done:"
for A in "${ARMS[@]}"; do log "  $A -> $(overall "$A")"; done

# 3. retry rounds — pool only the still-failing (arm,task) pairs
for R in 1 2; do
  log "RETRY$R start"
  $PY $D/failed_pairs.py "${ARMS[@]}" | run_pool
  for A in "${ARMS[@]}"; do ./kb.py reeval --sut "$A" >> $D/c456_${A}.log 2>&1; done
  log "RETRY$R done:"
  for A in "${ARMS[@]}"; do log "  $A -> $(overall "$A")"; done
done

# 4. M3/M4 judge on all 6
log "M3/M4 judge start (6 arms)"
./kb.py judge --sut "${ARMS[@]}" --tasks-file $D/all104.txt --lens both --force > $D/judge_c456_full.log 2>&1
log "M3/M4 judge done"
log "############ ORCH4 ALL DONE ############"
