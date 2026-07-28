#!/bin/bash
# C4/C5/C6 stats-star for BOTH models. Waits for orch3 (gpt-5.2 C3) to finish,
# then runs 6 arms full+2retry+reeval, then M3/M4 judge on all 6.
#   C4 = Delta 5k + stats D2
#   C5 = Delta 2k + stats D2
#   C6 = Latest 1k + stats D2
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
export KB_MAX_PARALLEL=4
D=judgment_runs/mini_star
PROG=$D/orch4_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

# arms: mini first (cheaper), then gpt-5.2
ARMS=(DataflowSystemGPT5MiniDeltaStats5kD2 \
      DataflowSystemGPT5MiniDeltaStats2kD2 \
      DataflowSystemGPT5MiniLatestStats1kD2 \
      DataflowSystemGPT52DeltaStats5kD2 \
      DataflowSystemGPT52DeltaStats2kD2 \
      DataflowSystemGPT52LatestStats1kD2)
TF=$D/all104.txt

# 1. wait for orch3 to finish (gpt-5.2 C1/C3 + its M3/M4)
log "waiting for orch3 ALL DONE before starting C4/C5/C6..."
while ! grep -q 'ORCH3 ALL DONE' $D/orch3_progress.log 2>/dev/null; do
  if ! pgrep -f '[o]rchestrator3.sh' >/dev/null && ! grep -q 'ORCH3 ALL DONE' $D/orch3_progress.log 2>/dev/null; then
    log "WARN orch3 gone without ALL DONE; proceeding anyway"; break
  fi
  sleep 60
done
log "orch3 done. starting C4/C5/C6 runs."

# 2. run each arm full + 2 retries + reeval
for ARM in "${ARMS[@]}"; do
  log "===== ARM $ARM ====="
  log "FULL104 start"
  ./kb.py run --sut "$ARM" --parallel --watchdog-min 8 >> $D/c456_${ARM}.log 2>&1
  log "FULL104 done -> $(overall "$ARM")"
  for R in 1 2; do
    log "RETRY$R start"
    ./kb.py rerun-failed --sut "$ARM" --all-failed --parallel --watchdog-min 8 >> $D/c456_${ARM}.log 2>&1
    log "RETRY$R done -> $(overall "$ARM")"
  done
  ./kb.py reeval --sut "$ARM" >> $D/c456_${ARM}.log 2>&1
  log "ARM $ARM FINAL -> $(overall "$ARM")"
done

# 3. M3/M4 judge on all 6 arms
log "M3/M4 judge start (6 arms)"
./kb.py judge --sut "${ARMS[@]}" --tasks-file "$TF" --lens both --force > $D/judge_c456_full.log 2>&1
log "M3/M4 judge done"
log "############ ORCH4 ALL DONE ############"
