#!/usr/bin/env bash
# Rerun score-0 environment tasks (gpt-5.2 0/0 SUT), isolated + parallel.
# NO reeval here — a single final reeval runs after this AND the other rerun finish
# (avoids two concurrent reevals racing on the same response_cache).
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; . ./.env; set +a
PY=.venv/bin/python
SUT=DataflowSystemGPT52LatestColumnStatsOnly
MAXPAR=6; TIMEOUT=420
TS=$(date +%Y%m%d_%H%M%S); LOGD="logs/rerun-env-gpt52-$TS"; mkdir -p "$LOGD"
IDS=$($PY kb.py failed --sut "$SUT" --zero-only --ids-only 2>/dev/null | tail -1 \
      | tr ' ' '\n' | grep -E '^environment-' | tr '\n' ' ')
echo "[env] rerunning $(wc -w <<<"$IDS") env tasks: $IDS"
run_task(){ local tid="$1"; echo "[env] start $tid"; \
  timeout "$TIMEOUT" $PY -u evaluate.py --sut "$SUT" --workload environment --task_id "$tid" \
    --no_pipeline_eval --verbose --use_truth_subset > "$LOGD/$tid.log" 2>&1; \
  echo "[env] done  $tid (exit $?)"; }
for tid in $IDS; do while [ "$(jobs -r|wc -l)" -ge "$MAXPAR" ]; do sleep 3; done; run_task "$tid" & done
wait
echo "[env] all env reruns finished (no reeval — final reeval runs separately)"
