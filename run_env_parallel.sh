#!/bin/bash
# Parallelize A's environment run: the 15 fast tasks in one process + each slow
# env-hard-16..20 in its own concurrent process. Staggered 3s so per-process
# measures CSV timestamps don't collide. max_steps unchanged (=50) for fairness.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
export PATH="$(pwd)/.venv/bin:$PATH" OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://localhost:8099/api
S=DataflowSystemHaiku45Annot2LineageThoughtReplay
LOG=logs/exp2c_envpar; mkdir -p "$LOG"
FAST=$(.venv/bin/python -c "import json;ids=[x['id'] for x in json.load(open('workload/environment.json'))];slow={'environment-hard-%d'%i for i in range(16,21)};print(' '.join(i for i in ids if i not in slow))")
pids=()
.venv/bin/python evaluate.py --sut $S --workload environment --no_pipeline_eval --verbose --use_truth_subset --task_id $FAST > "$LOG/fast.log" 2>&1 &
pids+=($!); echo "launched fast batch (15 tasks) pid ${pids[-1]}"
for t in environment-hard-16 environment-hard-17 environment-hard-18 environment-hard-19 environment-hard-20; do
  sleep 3
  .venv/bin/python evaluate.py --sut $S --workload environment --no_pipeline_eval --verbose --use_truth_subset --task_id "$t" > "$LOG/$t.log" 2>&1 &
  pids+=($!); echo "launched $t pid ${pids[-1]}"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "ENVPAR COMPLETE ($fail non-zero exits)"
