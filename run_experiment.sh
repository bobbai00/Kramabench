#!/usr/bin/env bash
# Canonical experiment driver (replaces the one-off run_rank34/run_e1/
# run_explore*/run_renderprefs/run_rpfresh scripts).
#
# Protocol per arm — the audit-standard fair comparison:
#   1. full run over all 6 workloads      (kb.py run --parallel, 8-min watchdog)
#   2. two symmetric --all-failed recovery rounds (isolate mode)
#   3. a scores snapshot
#
# Usage:
#   ./run_experiment.sh SUT [SUT ...]              # oracle mode (gold files)
#   ./run_experiment.sh --no-oracle SUT [SUT ...]  # exploration mode (lake glob)
#
# Arms run SEQUENTIALLY (symmetric stack conditions). Logs land under
# logs/exp-<tag>-<timestamp>/ with a driver.log of phase transitions.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python

ORACLE_FLAG=""
if [ "${1:-}" = "--no-oracle" ]; then
  ORACLE_FLAG="--no-oracle"
  shift
fi
[ $# -ge 1 ] || { echo "usage: $0 [--no-oracle] SUT [SUT ...]"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
TAG=$(echo "$1" | tr -cd 'A-Za-z0-9' | tail -c 24)
LOGD="logs/exp-$TAG-$TS"
mkdir -p "$LOGD"
DRIVER="$LOGD/driver.log"
log() { echo "[exp $(date +%H:%M:%S)] $*" | tee -a "$DRIVER"; }

log "logs -> $LOGD  (oracle_flag='${ORACLE_FLAG:-oracle}')"
for SUT in "$@"; do
  log "===== $SUT: FULL RUN start ====="
  $PY kb.py run --sut "$SUT" $ORACLE_FLAG --parallel --watchdog-min 8 > "$LOGD/$SUT-full.log" 2>&1
  log "$SUT full run exit $?"
  for round in 1 2; do
    log "===== $SUT: RECOVERY $round start ====="
    $PY kb.py rerun-failed --sut "$SUT" --all-failed $ORACLE_FLAG --parallel --isolate --watchdog-min 8 \
      > "$LOGD/$SUT-rec$round.log" 2>&1
    log "$SUT recovery $round exit $?"
  done
  $PY kb.py scores --sut "$SUT" > "$LOGD/$SUT-scores.log" 2>&1 || true
  log "$SUT scores snapshot written"
done
log "ALL ARMS DONE"
