#!/bin/bash
# Q-SERIES — the missing cell: 5k + STATS + LATEST + fact on the CURRENT render.
#
# A7 (70.5) and N1 (70.1) already covered 5k+stats+latest+fact, but both finished
# before the layout commit 23a5325fc (21:03 Jul 29) and before 6f544c4c1 (20:52) made
# `fileIoFacts` an independent default-on flag — they got the fact via the original
# stats-coupling. So stats-on has never run against the current render, where
# `Files read:` sits above `Code:` with `Inputs:`.
#
#   Q0  5k + code + fact, NO stats   (co-run control, = P0)
#   Q1  5k + code + fact + stats/dl2 (the missing cell)
#
# Q0 is co-run rather than reusing P0's reps so the pair shares engine age exactly.
# Prior expectation is parity-or-worse: stats have never helped LATEST (D8 71.3 vs
# N1 70.1; C8 69.0 vs C8s 68.6) and cost more. This closes the cell.
#
# CONCURRENCY, MEASURED THE HARD WAY (2026-07-30). P16-P32 does NOT work here.
# The binding resource is not the Kramabench runners but the ENGINE's Python UDF
# workers: ~10 per run at ~750 MB each, so a run costs ~7.5 GB and ~4.5 cores. At
# P16 that is 3x CPU oversubscription (load 76 on 24 cores) and at P32 the box
# reached 60/62 GB with 66 workers holding 49.9 GB. Starved runs then blow the 900s
# timeout: environment scored only 59 of 118 attempts, where earlier pools at lower
# effective load completed 100-104 of 104. Recycling the engine returned memory
# 60/62 -> 3/62 GB, which is also how the leak was confirmed.
#   safe: P6 main / P4 repair on this 24-core / 62 GB box.
#   watch `ps -eo args= | grep -c 'dataflow-agent/.venv/bin/python'` (workers),
#   not the run count.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orchQ_progress.log
# NOTE: append, never truncate. A relaunch (e.g. by memwatch after an engine
# recycle) used to `: > "$PROG"` and erase the previous instance's audit trail,
# which is how a bogus "ALL DONE" came to sit alone in the log.
touch "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

mapfile -t ARMS < "$D/ruleQ.txt"
mapfile -t TASKS < "$D/tasks_full104.txt"
log "P-SERIES code budget — ${#ARMS[@]} arms x ${#TASKS[@]} tasks = $(( ${#ARMS[@]} * ${#TASKS[@]} )) runs, xargs -P 10, task-major interleave, :3004."

# Abort if the engine dies, so we never accumulate 26-step/(empty response)
# garbage the way C12 did. Counts java by /proc/PID/exe, NOT by pattern: a
# `pgrep -f 'java @.*sbt-args'` counts this script's own shell too and reported
# 8 while only 6 JVMs were alive, which is why the last death went unnoticed.
guard(){
  while sleep 60; do
    n=0
    for d in /proc/[0-9]*; do
      case "$(readlink "$d/exe" 2>/dev/null)" in */java) n=$((n+1));; esac
    done
    if [[ "$n" -lt 8 ]] || ! lsof -tiTCP:8085 -sTCP:LISTEN >/dev/null 2>&1; then
      log "!!! ENGINE LOST (java=$n, :8085 down) — killing pool, results after this point would be garbage"
      pkill -P $$ 2>/dev/null
      touch "$D/Q_ABORTED"
      return 1
    fi
    for p in 3004; do
      lsof -tiTCP:$p -sTCP:LISTEN >/dev/null 2>&1 || { log "!!! agent-service :$p DOWN — killing pool"; pkill -P $$ 2>/dev/null; touch "$D/Q_ABORTED"; return 1; }
    done
  done
}
guard & GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

# task-major: consecutive lines alternate arms, so all 6 progress together
emit_pending(){
  for T in "${TASKS[@]}"; do
    for A in "${ARMS[@]}"; do
      L="$D/poolQ_${A}__${T}.log"
      grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
    done
  done
}

run_pass(){
  emit_pending | xargs -P "$1" -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/poolQ_$0__$2.log" 2>&1
  '
}

run_pass 10
[[ -f "$D/Q_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }
log "main pass done — repair pass at P8 (lower: repair runs are the heavy tail)"
run_pass 6
[[ -f "$D/Q_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }

# reeval SERIALLY: concurrent `kb.py reeval` corrupted results/aggregated_results.csv
# (torn SUT names, 0-row arms) and produced a bogus 43.8% for an N3 rep.

# COMPLETENESS GATE. On 2026-07-30 a relaunched instance reached this loop with only
# ~29 of 104 answers per arm and happily reeval'd them, writing measures CSVs that
# scored 9.6%/12.5% as if complete. Scoring partial data is worse than not scoring:
# it looks like a result. Refuse unless every arm has nearly all answers.
incomplete=0
for A in "${ARMS[@]}"; do
  na=$(ls system_scratch/$A/*/response.txt 2>/dev/null | wc -l)
  if [ "$na" -lt 100 ]; then log "  INCOMPLETE $A: $na/104 answers"; incomplete=1; fi
done
if [ "$incomplete" -eq 1 ]; then
  log "REFUSING to reeval — at least one arm is incomplete. Fix and rerun; nothing scored."
  exit 1
fi
log "repair pass done. reeval (serial) + score:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleQ_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHQ ALL DONE ############"
