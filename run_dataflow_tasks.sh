#!/bin/bash
export DATAFLOW_MAX_CELL_CHARS=10000
export DATAFLOW_MAX_RESULT_CHARS=5000
export DATAFLOW_MAX_STEPS=100
export DATAFLOW_FINE_GRAINED_PROMPT=false

# Configuration
# Set ORACLE_MODE=true to use ground truth subset files (--use_truth_subset)
# Set ORACLE_MODE=false to use all files in dataset directory
ORACLE_MODE=${ORACLE_MODE:-true}

# Usage: ./run_dataflow_tasks.sh <workload> <task_id> [task_id ...]
# Example: ./run_dataflow_tasks.sh legal legal-hard-1 legal-hard-3

if [ $# -lt 2 ]; then
    echo "Usage: $0 <workload> <task_id> [task_id ...]"
    echo "Example: $0 legal legal-hard-1 legal-hard-3"
    exit 1
fi

WORKLOAD="$1"
shift
TASK_IDS=("$@")

# Build extra arguments based on mode
EXTRA_ARGS=""
if [ "$ORACLE_MODE" = "true" ]; then
    EXTRA_ARGS="--use_truth_subset"
    echo "Running in ORACLE MODE (using ground truth subset files)"
else
    echo "Running in STANDARD MODE (using all files)"
fi

echo "=========================================="
echo "Running: $WORKLOAD | Tasks: ${TASK_IDS[*]}"
echo "=========================================="
python evaluate.py --sut DataflowSystemGpt52 --workload "$WORKLOAD" --no_pipeline_eval --verbose --task_id "${TASK_IDS[@]}" $EXTRA_ARGS
echo ""

echo "=========================================="
echo "Run complete!"
echo "=========================================="
