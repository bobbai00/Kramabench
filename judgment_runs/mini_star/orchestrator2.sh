#!/bin/bash
# Autonomous post-mini pipeline:
#  1. wait for mini fullrun ALL DONE
#  2. snapshot mini M1 (compute_scores per arm)
#  3. launch gpt-5.2 fullrun (long pole) in background
#  4. CONCURRENTLY run mini M3/M4 judge (OpenAI-only, no engine contention)
#  5. wait for gpt-5.2 fullrun done
#  6. run gpt-5.2 M3/M4 judge
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
D=judgment_runs/mini_star
PROG=$D/orch2_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

MINI=(DataflowSystemGPT5MiniDelta1kSchemaOnly DataflowSystemGPT5MiniDelta5kSchemaOnly \
      DataflowSystemGPT5MiniDeltaStats1kD2 DataflowSystemGPT5MiniLatest1kCodeInSnap)
G52=(DataflowSystemGPT52Delta1kSchemaOnly DataflowSystemGPT52Delta5kSchemaOnly \
     DataflowSystemGPT52DeltaStats1kD2 DataflowSystemGPT52Latest1kCodeInSnap)
TF=$D/all104.txt

# 1. wait for mini fullrun
log "waiting for mini fullrun ALL DONE..."
while ! grep -q 'ALL DONE' $D/fullrun_progress.log 2>/dev/null; do
  if ! pgrep -f '[f]ullrun.sh' >/dev/null && ! grep -q 'ALL DONE' $D/fullrun_progress.log 2>/dev/null; then
    log "WARN mini fullrun.sh gone without ALL DONE; proceeding with whatever scored"; break
  fi
  sleep 60
done
log "mini fullrun finished. mini M1 leaderboard:"
for A in "${MINI[@]}"; do log "  M1 $A -> $(overall "$A")"; done

# 3. launch gpt-5.2 fullrun (long pole) detached
log "launching gpt-5.2 fullrun (background)"
setsid bash $D/gpt52_fullrun.sh > $D/gpt52_fullrun.out 2>&1 < /dev/null &
disown

# 4. mini M3/M4 concurrently (judge = OpenAI only, no engine load)
log "mini M3/M4 judge start (concurrent with gpt-5.2 runs)"
./kb.py judge --sut "${MINI[@]}" --tasks-file "$TF" --lens both --force > $D/judge_mini_full.log 2>&1
log "mini M3/M4 judge done"

# 5. wait for gpt-5.2 fullrun
log "waiting for gpt-5.2 fullrun ALL DONE..."
while ! grep -q 'GPT52 ALL DONE' $D/gpt52_progress.log 2>/dev/null; do
  if ! pgrep -f '[g]pt52_fullrun.sh' >/dev/null && ! grep -q 'GPT52 ALL DONE' $D/gpt52_progress.log 2>/dev/null; then
    log "WARN gpt52_fullrun.sh gone without ALL DONE; proceeding"; break
  fi
  sleep 60
done
log "gpt-5.2 fullrun finished. gpt-5.2 M1 leaderboard:"
for A in "${G52[@]}"; do log "  M1 $A -> $(overall "$A")"; done

# 6. gpt-5.2 M3/M4
log "gpt-5.2 M3/M4 judge start"
./kb.py judge --sut "${G52[@]}" --tasks-file "$TF" --lens both --force > $D/judge_gpt52_full.log 2>&1
log "gpt-5.2 M3/M4 judge done"

log "############ ORCH2 ALL DONE ############"
