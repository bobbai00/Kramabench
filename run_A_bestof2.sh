#!/bin/bash
# Best-of-2 reruns for A (LATEST+both). Parallelized: normal-speed env failures
# in one process, each slow env-hard-16/17/19/20 in its own process, wildfire
# failures in one process. Staggered 3s so measures-CSV timestamps don't collide.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
export PATH="$(pwd)/.venv/bin:$PATH" OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:8099/api
S=DataflowSystemHaiku45Annot2LineageThoughtReplay
LOG=logs/A_bo2; mkdir -p "$LOG"
NORMAL_ENV="environment-easy-1 environment-easy-3 environment-hard-7 environment-hard-8 environment-hard-9 environment-hard-10 environment-hard-13"
WF="wildfire-easy-9 wildfire-hard-14 wildfire-hard-17 wildfire-hard-19 wildfire-hard-21"
pids=()
run() { .venv/bin/python evaluate.py --sut $S --workload "$1" --no_pipeline_eval --verbose --use_truth_subset --task_id $2 > "$LOG/$3.log" 2>&1 & pids+=($!); echo "launched $3 pid ${pids[-1]}"; }
run environment "$NORMAL_ENV" env_normal
sleep 3; run environment "environment-hard-16" env_h16
sleep 3; run environment "environment-hard-17" env_h17
sleep 3; run environment "environment-hard-19" env_h19
sleep 3; run environment "environment-hard-20" env_h20
sleep 3; run wildfire "$WF" wildfire
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "A_BO2 COMPLETE ($fail non-zero exits)"
