#!/bin/bash
# Re-evaluate specific tasks for DataflowSystemGpt52
# Using --use_system_cache to re-evaluate existing outputs with updated metrics

# Usage: ./evaluate_dataflow_tasks.sh <workload> <task_id> [task_id ...]
# Example: ./evaluate_dataflow_tasks.sh legal legal-hard-1 legal-hard-3

if [ $# -lt 2 ]; then
    echo "Usage: $0 <workload> <task_id> [task_id ...]"
    echo "Example: $0 legal legal-hard-1 legal-hard-3"
    exit 1
fi

WORKLOAD="$1"
shift
TASK_IDS=("$@")

echo "=========================================="
echo "Evaluating: $WORKLOAD | Tasks: ${TASK_IDS[*]}"
echo "=========================================="
python evaluate.py --sut DataflowSystemGpt52 --workload "$WORKLOAD" --no_pipeline_eval --verbose --use_system_cache --task_id "${TASK_IDS[@]}"
echo ""

echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
