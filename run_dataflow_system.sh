#!/bin/bash
export DATAFLOW_MAX_CELL_CHARS=2000
export DATAFLOW_MAX_RESULT_CHARS=10000
export DATAFLOW_MAX_STEPS=100

# Configuration
# Set ORACLE_MODE=true to use ground truth subset files (--use_truth_subset)
# Set ORACLE_MODE=false to use all files in dataset directory
ORACLE_MODE=${ORACLE_MODE:-true}

WORKLOADS=("archeology" "biomedical" "environment" "wildfire")
#WORKLOADS=("archeology" "astronomy" "biomedical" "environment" "legal" "wildfire")

# Build extra arguments based on mode
EXTRA_ARGS=""
if [ "$ORACLE_MODE" = "true" ]; then
    EXTRA_ARGS="--use_truth_subset"
    echo "Running in ORACLE MODE (using ground truth subset files)"
else
    echo "Running in STANDARD MODE (using all files)"
fi

for workload in "${WORKLOADS[@]}"; do
    echo "=========================================="
    echo "Running: $workload"
    echo "=========================================="
    python evaluate.py --sut DataflowSystemO4Mini --workload "$workload" --no_pipeline_eval --verbose $EXTRA_ARGS
    echo ""
done

echo "=========================================="
echo "All runs complete!"
echo "=========================================="
