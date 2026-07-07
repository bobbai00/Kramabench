#!/bin/bash
# Run the two gpt-5-mini systems (replay ON K=5 vs OFF) over ONLY the 38 tasks the
# Haiku both-flags system failed. PARALLEL=true => run_dataflow_tasks.sh fans the
# workload groups out. Agent (gpt-5-mini) routes via the proxy->OpenAI; judge
# (gpt-4o-mini) hits real OpenAI directly (OPENAI_BASE_URL unset, real key).
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; source .env 2>/dev/null; set +a
unset OPENAI_BASE_URL
export PATH="$(pwd)/.venv/bin:$PATH"
FAILED=$(tr '\n' ' ' < /tmp/A_failed.txt)
echo "running ${FAILED}" | tr ' ' '\n' | grep -c . | xargs echo "task count:"

SUT=DataflowSystemGPT5MiniAnnot2LineageThoughtReplay PARALLEL=true ./run_dataflow_tasks.sh $FAILED > logs/gpt5_replay_driver.log 2>&1 &
p1=$!
SUT=DataflowSystemGPT5MiniAnnot2Lineage PARALLEL=true ./run_dataflow_tasks.sh $FAILED > logs/gpt5_noreplay_driver.log 2>&1 &
p2=$!
wait $p1; r1=$?
wait $p2; r2=$?
echo "GPT5 BOTH COMPLETE (replay exit=$r1, noreplay exit=$r2)"
