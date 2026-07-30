#!/bin/bash
# Rep5-7 for anchor / C1 / C2 under the NEW prompt (terse summaries on DELTA).
# Paired with their Rep0-4, which ran under the OLD verbose-summary wording.
cd /home/bob/Desktop/bobflow/Kramabench
set -a; source .env; set +a
: > judgment_runs/mini_star/rep567_jobs.txt
for B in DataflowSystemGPT5MiniDelta1kSchemaOnly DataflowSystemGPT5MiniDelta5kSchemaOnly DataflowSystemGPT5MiniDeltaStats1kD2; do
  for i in 5 6 7; do
    while read -r wl tid; do echo "${B}Replicate$i $wl $tid"; done < judgment_runs/mini_star/c9_tasks.txt
  done
done >> judgment_runs/mini_star/rep567_jobs.txt
wc -l < judgment_runs/mini_star/rep567_jobs.txt
xargs -P 4 -L 1 bash -c '
  timeout 1200 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
    --task_id "$2" --use_truth_subset --no_pipeline_eval \
    > "judgment_runs/mini_star/pool567_$0__$2.log" 2>&1
' < judgment_runs/mini_star/rep567_jobs.txt
echo "REP567 DONE"
