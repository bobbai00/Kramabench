#!/usr/bin/env bash
# Rerun score-0 tasks for legal + environment (gpt-5.2 0/0 SUT), isolated + parallel,
# then a single rescore.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; . ./.env; set +a
PY=.venv/bin/python
SUT=DataflowSystemGPT52LatestColumnStatsOnly
MAXPAR=6; TIMEOUT=420
TS=$(date +%Y%m%d_%H%M%S); LOGD="logs/rerun-legalenv-gpt52-$TS"; mkdir -p "$LOGD"
IDS=$($PY kb.py failed --sut "$SUT" --zero-only --ids-only 2>/dev/null | tail -1 \
      | tr ' ' '\n' | grep -E '^(legal|environment)-' | tr '\n' ' ')
echo "[le] rerunning $(wc -w <<<"$IDS") tasks: $IDS"
run_task(){ local tid="$1" wl="${1%%-*}"; echo "[le] start $tid"; \
  timeout "$TIMEOUT" $PY -u evaluate.py --sut "$SUT" --workload "$wl" --task_id "$tid" \
    --no_pipeline_eval --verbose --use_truth_subset > "$LOGD/$tid.log" 2>&1; \
  echo "[le] done  $tid (exit $?)"; }
for tid in $IDS; do while [ "$(jobs -r|wc -l)" -ge "$MAXPAR" ]; do sleep 3; done; run_task "$tid" & done
wait
echo "[le] rescoring full 104"
$PY kb.py reeval --sut "$SUT" 2>&1 | sed -n '/====/,$p' | grep -E "archeology|astronomy|biomedical|environment|legal|wildfire|OVERALL|leaderboard" | tail -8
