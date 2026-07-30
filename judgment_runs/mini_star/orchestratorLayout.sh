#!/bin/bash
# LAYOUT A/B — `Files read:` inside `Result:` vs above `Code:`. FULL 104, 3 reps each.
#   LayoutOld  :3003  agent-service 311ddd646  (fact at the bottom of Result:)
#   LayoutNew  :3002  agent-service c516d800f  (fact above Code:, with Inputs:)
# Config is D8F's exactly: LATEST 5k + code + files-read, no stats.
#
# WHY THIS EXISTS: D8F reps 4-5 (new layout) scored 65.6 vs reps 1-3 (old) 71.2,
# -5.6 pt at 2.90x SE — but reps 1-3 ran on a 1-3 h engine and reps 4-5 on a 10 h
# engine that died minutes later, and engine age alone moves scores ~2.3 pt. That
# comparison cannot separate layout from senescence.
#
# TWO DESIGN CHOICES THAT MAKE THIS ONE CLEAN, both learned from that failure:
#
#  1. TASK-MAJOR EMISSION. The previous orchestrators emitted arm-major (all of
#     arm 1, then arm 2, ...), so with xargs -P the early arms ran on a younger
#     engine than the late ones — the exact confound under test. Here the inner
#     loop is over ARMS, so all 6 arms advance in lockstep and engine age is held
#     constant BY CONSTRUCTION rather than corrected for afterwards.
#
#  2. P16, not P30. The N6 pool's own log claimed P12 while the code ran P30, and
#     its P20 repair pass hit `OutOfMemoryError: unable to create native thread`
#     on a 10 h engine, killing ComputingUnitMaster. P30 is only safe on a fresh
#     engine. 16 is the compromise; the abort guard below is the real protection.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orchLayout_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

mapfile -t ARMS < "$D/ruleLayout.txt"
mapfile -t TASKS < "$D/tasks_full104.txt"
log "LAYOUT A/B — ${#ARMS[@]} arms x ${#TASKS[@]} tasks = $(( ${#ARMS[@]} * ${#TASKS[@]} )) runs, xargs -P 16, task-major interleave, :3002 + :3003."

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
      touch "$D/LAYOUT_ABORTED"
      return 1
    fi
    for p in 3002 3003; do
      lsof -tiTCP:$p -sTCP:LISTEN >/dev/null 2>&1 || { log "!!! agent-service :$p DOWN — killing pool"; pkill -P $$ 2>/dev/null; touch "$D/LAYOUT_ABORTED"; return 1; }
    done
  done
}
guard & GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT

# task-major: consecutive lines alternate arms, so all 6 progress together
emit_pending(){
  for T in "${TASKS[@]}"; do
    for A in "${ARMS[@]}"; do
      L="$D/poolLayout_${A}__${T}.log"
      grep -q 'Total score' "$L" 2>/dev/null || echo "$A ${T%%-*} $T"
    done
  done
}

run_pass(){
  emit_pending | xargs -P "$1" -L 1 bash -c '
    timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --task_id "$2" --use_truth_subset --no_pipeline_eval \
      > "judgment_runs/mini_star/poolLayout_$0__$2.log" 2>&1
  '
}

run_pass 16
[[ -f "$D/LAYOUT_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }
log "main pass done — repair pass at P8 (lower: repair runs are the heavy tail)"
run_pass 8
[[ -f "$D/LAYOUT_ABORTED" ]] && { log "aborted — not scoring"; exit 1; }

# reeval SERIALLY: concurrent `kb.py reeval` corrupted results/aggregated_results.csv
# (torn SUT names, 0-row arms) and produced a bogus 43.8% for an N3 rep.
log "repair pass done. reeval (serial) + score:"
for A in "${ARMS[@]}"; do
  ./kb.py reeval --sut "$A" >> $D/ruleLayout_${A}.log 2>&1
  log "  $A -> $(overall "$A")"
done
log "############ ORCHLAYOUT ALL DONE ############"
