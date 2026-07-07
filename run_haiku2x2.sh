#!/bin/bash
# Haiku-4.5 2x2 parameter study driver: run the legal workload for all four
# configs. PHASE selects run1 (full workload) or run2 (best-of-2 reruns of
# the task ids passed per SUT via the RERUN_* env vars). Scoring LLM (gpt-4o-mini)
# and the agent model both route through the local Anthropic proxy on :8099.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
export PATH="$(pwd)/.venv/bin:$PATH"
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8099/api

SUTS=(
  DataflowSystemHaiku45Annot2
  DataflowSystemHaiku45Annot2Lineage
  DataflowSystemHaiku45Annot2ThoughtReplay
  DataflowSystemHaiku45Annot2LineageThoughtReplay
)

PHASE="${1:-run1}"
LOGDIR="logs/$PHASE"
mkdir -p "$LOGDIR"

pids=()
for s in "${SUTS[@]}"; do
  if [ "$PHASE" = "run1" ]; then
    .venv/bin/python evaluate.py --sut "$s" --workload legal --no_pipeline_eval --verbose --use_truth_subset \
      > "$LOGDIR/$s.log" 2>&1 &
  else
    # run2: rerun only the failed task ids for this SUT (var RERUN_<s>)
    var="RERUN_${s}"
    ids="${!var:-}"
    if [ -n "$ids" ]; then
      .venv/bin/python evaluate.py --sut "$s" --workload legal --no_pipeline_eval --verbose --use_truth_subset \
        --task_id $ids > "$LOGDIR/$s.log" 2>&1 &
    else
      echo "[$s] no rerun tasks" > "$LOGDIR/$s.log"
      : &
    fi
  fi
  pids+=($!)
  echo "launched $s (pid ${pids[-1]})"
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[DONE] ${SUTS[$i]}"; else echo "[EXIT≠0] ${SUTS[$i]}"; fail=$((fail+1)); fi
done
echo "$PHASE COMPLETE ($fail non-zero exits)"
