#!/bin/bash
# Post-C1 gpt-5.2 pipeline (C2 skipped per user):
#  1. wait for gpt52_fullrun.sh to stop (stopper kills it after arm2 C1 FINAL)
#  2. run C3 (NEW Latest1kCodeInSnap): full 104 + 2 retries + reeval
#  3. gpt-5.2 M3/M4 judge on anchor + C1 + C3 (skip C2)
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
export $(grep -vE '^#' .env | sed 's/^export //' | xargs)
export KB_MAX_PARALLEL=4
D=judgment_runs/mini_star
PROG=$D/orch3_progress.log
: > "$PROG"
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$PROG"; }
overall(){ .venv/bin/python compute_scores.py --sut "$1" 2>/dev/null | awk '/OVERALL/{print $2,$3,$4}'; }

ANCHOR=DataflowSystemGPT52Delta1kSchemaOnly
C1=DataflowSystemGPT52Delta5kSchemaOnly
C3=DataflowSystemGPT52Latest1kCodeInSnap
TF=$D/all104.txt

# 1. wait until gpt52_fullrun stopped (stopper handles kill after C1)
log "waiting for gpt52_fullrun to stop after C1..."
while pgrep -f '[g]pt52_fullrun.sh' >/dev/null; do sleep 30; done
log "gpt52_fullrun stopped. gpt-5.2 anchor+C1 M1:"
log "  anchor -> $(overall $ANCHOR)"
log "  C1     -> $(overall $C1)"

# 2. run C3 (new code arm): full + 2 retries + reeval
log "C3 ($C3) FULL104 start"
./kb.py run --sut "$C3" --parallel --watchdog-min 8 >> $D/gpt52_${C3}.log 2>&1
log "C3 FULL104 done -> $(overall $C3)"
for R in 1 2; do
  log "C3 RETRY$R start"
  ./kb.py rerun-failed --sut "$C3" --all-failed --parallel --watchdog-min 8 >> $D/gpt52_${C3}.log 2>&1
  log "C3 RETRY$R done -> $(overall $C3)"
done
./kb.py reeval --sut "$C3" >> $D/gpt52_${C3}.log 2>&1
log "C3 FINAL -> $(overall $C3)"

# 3. gpt-5.2 M3/M4 on anchor + C1 + C3
log "gpt-5.2 M3/M4 judge start (anchor + C1 + C3)"
./kb.py judge --sut "$ANCHOR" "$C1" "$C3" --tasks-file "$TF" --lens both --force > $D/judge_gpt52_full.log 2>&1
log "gpt-5.2 M3/M4 judge done"
log "############ ORCH3 ALL DONE ############"
