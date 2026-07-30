#!/bin/bash
# Engine recycler for long pools. The engine leaks Python UDF workers (~750 MB each,
# never reaped), so available memory falls monotonically until the OOM killer picks a
# JVM and destroys the run. Measured 2026-07-30: 66 workers holding 49.9 GB took the
# box to 0 GB available; recycling the engine returned memory 60/62 -> 3/62 GB.
#
# This watchdog does what was otherwise done by hand: at a memory floor, stop the
# pool, recycle the engine, and relaunch the orchestrator. Resume-skip means no
# completed work is lost — only the handful of in-flight runs, which get retried.
#
# Ordering matters: kill the ORCHESTRATOR FIRST, then its xargs. Killing xargs first
# makes the orchestrator's `run_pass` return and it immediately advances to the next
# pass, spawning a fresh xargs — observed live.
#
# Every kill uses literal PIDs from an explicit list. Pattern matching (`pkill -f`,
# `pgrep -f`) matches this script's own command line and has killed the wrong process
# repeatedly.
set -uo pipefail
cd ~/Desktop/bobflow/Kramabench
D=judgment_runs/mini_star
LOG=$D/memwatch.log
FLOOR_GB=${1:-18}          # recycle below this many GB available
MAX_CYCLES=${2:-8}         # backstop against an infinite recycle loop
cycles=0

log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" >> "$LOG"; }

java_pids(){
  for d in /proc/[0-9]*; do
    case "$(readlink "$d/exe" 2>/dev/null)" in */java) echo "${d#/proc/}";; esac
  done
}
orch_pids(){ ps -eo pid=,args= | awk '{for(i=2;i<=NF;i++) if($i ~ /orchestratorP\.sh$/){print $1; break}}'; }
xargs_pids(){ ps -eo pid=,comm= | awk '$2=="xargs"{print $1}'; }
worker_pids(){ ps -eo pid=,args= | awk '/dataflow-agent\/\.venv\/bin\/python/{print $1}'; }

log "memwatch armed: floor=${FLOOR_GB}GB max_cycles=$MAX_CYCLES"

while true; do
  sleep 60
  grep -q 'ORCHP ALL DONE' $D/orchP_progress.log 2>/dev/null && { log "pool done, exiting"; exit 0; }
  [ -f $D/P_ABORTED ] && { log "pool aborted, exiting"; exit 0; }

  avail=$(free -g | awk 'NR==2{print $7}')
  [ "$avail" -ge "$FLOOR_GB" ] && continue

  w=$(worker_pids | wc -l)
  log "FLOOR HIT: ${avail}GB available, $w UDF workers -> recycling (cycle $((cycles+1)))"
  cycles=$((cycles+1))
  if [ "$cycles" -gt "$MAX_CYCLES" ]; then
    log "max cycles reached, refusing to recycle again; leaving pool stopped"
    O=$(orch_pids | tr '\n' ' '); [ -n "$O" ] && kill $O 2>/dev/null
    sleep 1
    X=$(xargs_pids | tr '\n' ' '); [ -n "$X" ] && kill $X 2>/dev/null
    exit 1
  fi

  # 1. stop the pool — orchestrator BEFORE xargs
  O=$(orch_pids | tr '\n' ' '); [ -n "$O" ] && kill $O 2>/dev/null
  sleep 2
  X=$(xargs_pids | tr '\n' ' '); [ -n "$X" ] && kill $X 2>/dev/null
  sleep 3
  O=$(orch_pids | tr '\n' ' '); [ -n "$O" ] && kill -9 $O 2>/dev/null
  X=$(xargs_pids | tr '\n' ' '); [ -n "$X" ] && kill -9 $X 2>/dev/null

  # 2. recycle the engine (this is what actually reclaims the leaked memory)
  J=$(java_pids | tr '\n' ' '); [ -n "$J" ] && kill $J 2>/dev/null
  sleep 12
  J=$(java_pids | tr '\n' ' '); [ -n "$J" ] && kill -9 $J 2>/dev/null
  sleep 4
  W=$(worker_pids | tr '\n' ' '); [ -n "$W" ] && kill -9 $W 2>/dev/null
  sleep 4
  log "after recycle: $(free -g | awk 'NR==2{print $7}')GB available, $(worker_pids | wc -l) workers"

  # 3. relaunch the engine and wait for the execution port
  nohup setsid /tmp/launch_stack2.sh > ~/Desktop/bobflow/dataflow-agent/logs/sbt-services.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do
    sleep 10
    lsof -tiTCP:8085 -sTCP:LISTEN >/dev/null 2>&1 && break
  done
  if ! lsof -tiTCP:8085 -sTCP:LISTEN >/dev/null 2>&1; then
    log "engine did NOT come back on :8085 — stopping, needs a human"
    exit 1
  fi
  jn=$(java_pids | wc -l)
  log "engine back: java=$jn :8085 up"

  # 4. resume the pool; resume-skip re-runs only unscored (arm,task) pairs
  rm -f $D/P_ABORTED
  nohup setsid ./judgment_runs/mini_star/orchestratorP.sh > /dev/null 2>&1 &
  sleep 20
  log "pool resumed: scored=$(grep -l 'Total score' $D/poolP_* 2>/dev/null | wc -l)/936"
done
