#!/bin/bash
# C9 = latest + 5k + code-in-snapshot + column stats, 5 clean single-shot reps.
# Global 4-wide task pool across all reps (engine-safe concurrency, as used by
# the earlier knob-star runs). No retries: raw single-shot, like the other reps.
cd /home/bob/Desktop/bobflow/Kramabench
set -a; source .env; set +a
B=DataflowSystemGPT5MiniLatestStats5kD2Code
: > judgment_runs/mini_star/c9_jobs.txt
for i in 0 1 2 3 4; do
  while read -r wl tid; do echo "${B}Replicate$i $wl $tid"; done < judgment_runs/mini_star/c9_tasks.txt
done >> judgment_runs/mini_star/c9_jobs.txt
wc -l < judgment_runs/mini_star/c9_jobs.txt
xargs -P 4 -L 1 bash -c '
  timeout 900 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/poolc9_$0__$2.log" 2>&1
' < judgment_runs/mini_star/c9_jobs.txt
echo "ALL C9 REPS DONE"
