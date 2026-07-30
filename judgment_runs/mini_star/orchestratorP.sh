#!/bin/bash
# P-SERIES — the CODE budget. P0 control / P1 codeMaxChars 800 / P2 400, 3 reps, full 104.
#
# Byte accounting over ~100 traces/arm: `Code:` is 27-39% of the rendered dataflow
# (the LARGEST component at a 1k row budget) and `max_operator_result_char_limit`
# clamps table ROWS only, so code had never been under any budget. Code size is
# long-tailed (p50 286 B, p90 2,014 B, p99 6,081 B, max 16,400 B), so an 800 B cap
# removes ~49% of code bytes touching only 23.5% of blocks; 400 B removes ~64%.
#
# All three arms hit :3004 (the code-lean build). P0 sets no cap and is therefore
# byte-parity with D8F, present so the comparison never rests on cross-pool engine
# age — the mistake that made D8F reps 4-5 uninterpretable.
#
# Task-major emission and P16, for the reasons documented in HANDOFF 4.1/4.2.
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
PROG=$D/orchP_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

mapfile -t ARMS < "$D/ruleP.txt"
mapfile -t TASKS < "$D/tasks_full104.txt"
log "P-SERIES code budget — ${#ARMS[@]} arms x ${#TASKS[@]} tasks = $(( ${#ARMS[@]} * ${#TASKS[@]} )) runs, xargs -P 6, task-major interleave, :3004."

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
      touch "$D/P_ABORTED"
      return 1
    fi
    for p in 3004; do
      lsof -tiTCP:$p -sTCP:LISTEN >/dev/null 2>&1 || { log "!!! agent-service :$p DOWN — killing pool"; pkill -P $$ 2>/dev/null; touch "$D/P_ABORTED"; return 1; }
    done
  done
}
guard & GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

# task-major: consecutive lines alternate arms, so all 6 progress together
emit_pending(){
  for T in "${TASKS[@]}"; do
    for A in "${ARMS[@]}"; do
      L="$D/poolP_${A}__${T}.log"
      grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
    done
  done
}

run_pass(){
  emit_pending | xargs -P "$1" -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/poolP_$0__$2.log" 2>&1
  '
}

run_pass 6
[[ -f "$D/P_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }
log "main pass done — repair pass at P8 (lower: repair runs are the heavy tail)"
run_pass 4
[[ -f "$D/P_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }

# reeval SERIALLY: concurrent `kb.py reeval` corrupted results/aggregated_results.csv
# (torn SUT names, 0-row arms) and produced a bogus 43.8% for an N3 rep.
log "repair pass done. reeval (serial) + score:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleP_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHP ALL DONE ############"
