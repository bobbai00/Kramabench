#!/usr/bin/env bash
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; . ./.env; set +a
PY=.venv/bin/python
SUT=DataflowSystemGPT52DeltaColumnStatsDataHints
MAXPAR=6; TIMEOUT=480
TS=$(date +%Y%m%d_%H%M%S); LOGD="logs/recover-delta-$TS"; mkdir -p "$LOGD"
IDS="archeology-hard-5 archeology-easy-6 archeology-hard-7 archeology-easy-8 archeology-hard-9 archeology-easy-10 archeology-easy-11 archeology-hard-12 astronomy-hard-12"
echo "[rd] recovering $(wc -w <<<"$IDS") watchdog-killed tasks"
run_task(){ local tid="$1" wl="${1%%-*}"; echo "[rd] start $tid"; \
  timeout "$TIMEOUT" $PY -u evaluate.py --sut "$SUT" --workload "$wl" --task_id "$tid" \
    --no_pipeline_eval --verbose --use_truth_subset > "$LOGD/$tid.log" 2>&1; \
  echo "[rd] done  $tid (exit $?)"; }
for tid in $IDS; do while [ "$(jobs -r|wc -l)" -ge "$MAXPAR" ]; do sleep 3; done; run_task "$tid" & done
wait
echo "[rd] rescoring full 104"
$PY kb.py reeval --sut "$SUT" 2>&1 | sed -n '/====/,$p' | grep -E "archeology|astronomy|biomedical|environment|legal|wildfire|OVERALL|leaderboard" | tail -8
