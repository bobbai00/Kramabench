#!/usr/bin/env bash
# Recover the tasks the watchdog left unanswered in the gpt-5.2 0/0 run.
# One isolated `-u` process per task, bounded parallelism + per-task timeout.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; . ./.env; set +a
PY=.venv/bin/python
SUT=DataflowSystemGPT52LatestColumnStatsOnly
MAXPAR=6; TIMEOUT=420
TS=$(date +%Y%m%d_%H%M%S); LOGD="logs/recover-gpt52-$TS"; mkdir -p "$LOGD"
IDS="astronomy-hard-12 environment-hard-10 environment-hard-11 environment-hard-12 environment-hard-13 environment-hard-14 environment-hard-15 environment-hard-16 environment-hard-17 environment-hard-18 environment-hard-19 environment-hard-20"
echo "[rec] recovering $(wc -w <<<"$IDS") unanswered tasks (maxpar=$MAXPAR, timeout=${TIMEOUT}s)"
run_task() {
  local tid="$1" wl="${1%%-*}" log="$LOGD/$1.log"
  echo "[rec] start $tid"
  timeout "$TIMEOUT" $PY -u evaluate.py --sut "$SUT" --workload "$wl" --task_id "$tid" \
      --no_pipeline_eval --verbose --use_truth_subset > "$log" 2>&1
  echo "[rec] done  $tid (exit $?)"
}
for tid in $IDS; do
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 3; done
  run_task "$tid" &
done
wait
echo "[rec] rebuilding cache + rescoring"
$PY kb.py reeval --sut "$SUT" 2>&1 | tail -12
