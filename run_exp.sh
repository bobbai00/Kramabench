#!/bin/bash
# Generic SUT x workload runner. SUTS + WORKLOADS via env (space-separated).
# PHASE run1 = full workloads; run2 = best-of-2 reruns of failed task ids passed
# via RERUN_<SUT>__<WORKLOAD> env vars. LLM + judge route through the proxy on :8099.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
export PATH="$(pwd)/.venv/bin:$PATH"
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8099/api

PHASE="${1:-run1}"
read -ra SUTS <<< "${SUTS:-}"
read -ra WORKLOADS <<< "${WORKLOADS:-}"
LOGDIR="logs/exp_$PHASE"; mkdir -p "$LOGDIR"

pids=()
for s in "${SUTS[@]}"; do
  for w in "${WORKLOADS[@]}"; do
    log="$LOGDIR/${s}__${w}.log"
    if [ "$PHASE" = "run1" ]; then
      .venv/bin/python evaluate.py --sut "$s" --workload "$w" --no_pipeline_eval --verbose --use_truth_subset > "$log" 2>&1 &
    else
      var="RERUN_${s}__${w}"; ids="${!var:-}"
      if [ -n "$ids" ]; then
        .venv/bin/python evaluate.py --sut "$s" --workload "$w" --no_pipeline_eval --verbose --use_truth_subset --task_id $ids > "$log" 2>&1 &
      else echo "[$s/$w] no rerun tasks" > "$log"; : & fi
    fi
    pids+=($!); echo "launched $s / $w (pid ${pids[-1]})"
  done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "$PHASE COMPLETE ($fail non-zero exits)"
