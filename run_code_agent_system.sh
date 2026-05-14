#!/bin/bash

# Kill all child processes on Ctrl+C
trap 'echo ""; echo "Interrupted! Killing all subprocesses..."; kill 0; exit 1' INT TERM

export CODE_AGENT_MAX_STEPS=50
export CUSTOMIZED_PROMPT_ENABLED=true
export FINE_GRAINED_PROMPT_ENABLED=false
export CODE_AGENT_MAX_PRINT_OUTPUTS_LENGTH=1000

# Configuration
SUT=${SUT:-CodeAgentSystemGpt5MiniMediumResultChars1000}
# Set ORACLE_MODE=true to use ground truth subset files (--use_truth_subset)
# Set ORACLE_MODE=false to use all files in dataset directory
ORACLE_MODE=${ORACLE_MODE:-true}
# Set PARALLEL=true to run all workloads in parallel (default: true)
PARALLEL=${PARALLEL:-true}

WORKLOADS=("legal" "environment" "wildfire" "astronomy" "archeology" "biomedical")
# WORKLOADS=("astronomy")
#WORKLOADS=("astronomy" "legal")

# Build extra arguments based on mode
EXTRA_ARGS=""
if [ "$ORACLE_MODE" = "true" ]; then
    EXTRA_ARGS="--use_truth_subset"
    echo "Running in ORACLE MODE (using ground truth subset files)"
else
    echo "Running in STANDARD MODE (using all files)"
fi

# Create a log directory
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if [ "$PARALLEL" = "true" ]; then
    echo "Running ${#WORKLOADS[@]} workloads in PARALLEL for $SUT"
    echo "Logs: $LOG_DIR/"
    echo "=========================================="

    PIDS=()
    for workload in "${WORKLOADS[@]}"; do
        echo "Starting: $SUT | $workload (log: $LOG_DIR/${workload}.log)"
        python evaluate.py --sut "$SUT" --workload "$workload" --no_pipeline_eval --verbose $EXTRA_ARGS \
            > "$LOG_DIR/${workload}.log" 2>&1 &
        PIDS+=("$!:$workload")
    done

    echo "=========================================="
    echo "Waiting for all workloads to finish..."
    echo "=========================================="

    FAILED=0
    for entry in "${PIDS[@]}"; do
        pid="${entry%%:*}"
        workload="${entry##*:}"
        if wait "$pid"; then
            echo "[DONE] $workload (exit 0)"
        else
            exit_code=$?
            echo "[FAIL] $workload (exit $exit_code)"
            FAILED=$((FAILED + 1))
        fi
    done

    echo "=========================================="
    if [ $FAILED -eq 0 ]; then
        echo "All ${#WORKLOADS[@]} workloads completed successfully!"
    else
        echo "$FAILED/${#WORKLOADS[@]} workloads failed. Check logs in $LOG_DIR/"
    fi
    echo "=========================================="
else
    echo "Running ${#WORKLOADS[@]} workloads SEQUENTIALLY for $SUT"
    echo "=========================================="

    for workload in "${WORKLOADS[@]}"; do
        echo "=========================================="
        echo "Running: $SUT | $workload"
        echo "=========================================="
        python evaluate.py --sut "$SUT" --workload "$workload" --no_pipeline_eval --verbose $EXTRA_ARGS \
            | tee "$LOG_DIR/${workload}.log"
        echo ""
    done

    echo "=========================================="
    echo "All runs complete!"
    echo "=========================================="
fi
