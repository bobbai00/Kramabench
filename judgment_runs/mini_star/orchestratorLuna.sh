#!/bin/bash
# LUNA SWEEP — gpt-5.6-luna anchor/C1-C4, 3 reps, full 104 = 1560 runs.
#
#   LunaAnchor  1K, DELTA,  no stats
#   LunaC1      5K, DELTA,  no stats             (+sampling)
#   LunaC2      1K, DELTA,  stats + hints        (+stats)
#   LunaC3      1K, LATEST, no stats, +code      (+latest)
#   LunaC4      5K, LATEST, stats + hints, +code (all three)
#
# Mirrors the gpt-5-mini anchor/C1-C4 factorial so the models compare knob-for-knob.
#
# MODEL PLUMBING: luna/terra reject function tools on /v1/chat/completions with ANY
# reasoning effort (medium, low, or omitted) — only "none" is accepted there, which
# would benchmark the model with reasoning OFF. litellm therefore routes them via
# `openai/responses/gpt-5.6-luna` to /v1/responses at reasoning_effort=medium, where
# tools and reasoning coexist. Verified: bat-and-ball answered 5 (not 10) with
# reasoning_tokens>0.
#
# P16 main / P8 repair: luna is a reasoning model over the Responses API, so runs
# spend more wall-clock in LLM-wait and fewer UDF workers execute concurrently than
# with gpt-5-mini. memwatch is armed regardless.
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
PROG=$D/orchLuna_progress.log
# NOTE: append, never truncate. A relaunch (e.g. by memwatch after an engine
# recycle) used to `: > "$PROG"` and erase the previous instance's audit trail,
# which is how a bogus "ALL DONE" came to sit alone in the log.
touch "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

mapfile -t ARMS < "$D/ruleLuna.txt"
mapfile -t TASKS < "$D/tasks_full104.txt"
log "LUNA sweep (gpt-5.6-luna anchor/C1-C4) — ${#ARMS[@]} arms x ${#TASKS[@]} tasks = $(( ${#ARMS[@]} * ${#TASKS[@]} )) runs, xargs -P 16, task-major interleave, :3004."

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
      touch "$D/LUNA_ABORTED"
      return 1
    fi
    for p in 3004; do
      lsof -tiTCP:$p -sTCP:LISTEN >/dev/null 2>&1 || { log "!!! agent-service :$p DOWN — killing pool"; pkill -P $$ 2>/dev/null; touch "$D/LUNA_ABORTED"; return 1; }
    done
  done
}
guard & GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

# task-major: consecutive lines alternate arms, so all 6 progress together
emit_pending(){
  for T in "${TASKS[@]}"; do
    for A in "${ARMS[@]}"; do
      L="$D/poolLuna_${A}__${T}.log"
      grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
    done
  done
}

run_pass(){
  emit_pending | xargs -P "$1" -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/poolLuna_$0__$2.log" 2>&1
  '
}

run_pass 16
[[ -f "$D/LUNA_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }
log "main pass done — repair pass at P8 (lower: repair runs are the heavy tail)"
run_pass 8
[[ -f "$D/LUNA_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }

# reeval SERIALLY: concurrent `kb.py reeval` corrupted results/aggregated_results.csv
# (torn SUT names, 0-row arms) and produced a bogus 43.8% for an N3 rep.

# COMPLETENESS GATE. On 2026-07-30 a relaunched instance reached this loop with only
# ~29 of 104 answers per arm and happily reeval'd them, writing measures CSVs that
# scored 9.6%/12.5% as if complete. Scoring partial data is worse than not scoring:
# it looks like a result. Threshold is 95, not 100: the timeout tail (astronomy-
# hard-11 and friends) legitimately leaves an arm at 99-101, and a gate that can
# never be satisfied stalls the pool forever. 95 still blocks the failure it exists
# for, which was ~29/104.
incomplete=0
for A in "${ARMS[@]}"; do
  na=$(ls system_scratch/$A/*/response.txt 2>/dev/null | wc -l)
  if [ "$na" -lt 95 ]; then log "  INCOMPLETE $A: $na/104 answers"; incomplete=1; fi
done
if [ "$incomplete" -eq 1 ]; then
  log "REFUSING to reeval — at least one arm is incomplete. Fix and rerun; nothing scored."
  exit 1
fi
log "repair pass done. reeval (serial) + score:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleLuna_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHLUNA ALL DONE ############"
