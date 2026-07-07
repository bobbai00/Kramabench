#!/usr/bin/env bash
# Rerun score-0 tasks for legal/biomedical/astronomy/archeology (gpt-5.2 0/0 SUT),
# isolated per task, bounded parallelism + per-task timeout, then rescore.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; . ./.env; set +a
PY=.venv/bin/python
SUT=DataflowSystemGPT52LatestColumnStatsOnly
MAXPAR=6; TIMEOUT=420
TS=$(date +%Y%m%d_%H%M%S); LOGD="logs/rerun4-gpt52-$TS"; mkdir -p "$LOGD"
IDS=$($PY kb.py failed --sut "$SUT" --zero-only --ids-only 2>/dev/null | tail -1 \
      | tr ' ' '\n' | grep -E '^(legal|biomedical|astronomy|archeology)-' | tr '\n' ' ')
echo "[r4] rerunning $(wc -w <<<"$IDS") tasks: $IDS"
run_task(){ local tid="$1" wl="${1%%-*}"; echo "[r4] start $tid"; \
  timeout "$TIMEOUT" $PY -u evaluate.py --sut "$SUT" --workload "$wl" --task_id "$tid" \
    --no_pipeline_eval --verbose --use_truth_subset > "$LOGD/$tid.log" 2>&1; \
  echo "[r4] done  $tid (exit $?)"; }
for tid in $IDS; do while [ "$(jobs -r|wc -l)" -ge "$MAXPAR" ]; do sleep 3; done; run_task "$tid" & done
wait
echo "[r4] rescoring full 104"
$PY kb.py reeval --sut "$SUT" 2>&1 | tail -12
