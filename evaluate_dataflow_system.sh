#!/bin/bash
# Re-evaluate all workloads for DataflowSystemGptO3
# Using --use_system_cache to re-evaluate existing outputs with updated metrics

WORKLOADS=("archeology" "astronomy" "biomedical" "environment" "legal" "wildfire")

for workload in "${WORKLOADS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $workload"
    echo "=========================================="
    python evaluate.py --sut DataflowSystemGptO3 --workload "$workload" --no_pipeline_eval --verbose --use_system_cache
    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
