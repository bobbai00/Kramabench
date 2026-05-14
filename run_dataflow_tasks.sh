#!/bin/bash

# Kill all child processes on Ctrl+C
trap 'echo ""; echo "Interrupted! Killing all subprocesses..."; kill 0; exit 1' INT TERM

export DATAFLOW_MAX_CELL_CHARS=${DATAFLOW_MAX_CELL_CHARS:-10000}
export DATAFLOW_MAX_RESULT_CHARS=${DATAFLOW_MAX_RESULT_CHARS:-1000}
export DATAFLOW_MAX_STEPS=${DATAFLOW_MAX_STEPS:-100}
export DATAFLOW_FINE_GRAINED_PROMPT=${DATAFLOW_FINE_GRAINED_PROMPT:-false}
export DATAFLOW_COLLECT_REACT_STEPS=true

# Configuration
# Set SUT to override system under test (default: DataflowSystemHaiku45CtxOptD1Limit500)
SUT=${SUT:-DataflowSystemHaiku45CtxOptD1Limit500}
# Set ORACLE_MODE=true to use ground truth subset files (--use_truth_subset)
# Set ORACLE_MODE=false to use all files in dataset directory
ORACLE_MODE=${ORACLE_MODE:-true}
# Set PARALLEL=true to run workload groups in parallel (default: false for legacy, true for auto-group mode)
PARALLEL=${PARALLEL:-auto}

# Usage: ./run_dataflow_tasks.sh <task_id> [task_id ...]
#   Workload is parsed from task IDs (e.g. legal-hard-1 -> legal)
#   Tasks are grouped by workload and run in parallel.
# Legacy usage also supported: ./run_dataflow_tasks.sh <workload> <task_id> [task_id ...]
# Example: ./run_dataflow_tasks.sh legal-hard-1 legal-hard-3 astronomy-easy-1
# Example: SUT=DataflowSystemGpt52ResultChars2000 ./run_dataflow_tasks.sh legal-hard-1
# Example: PARALLEL=true SUT=DataflowSystemGpt52 ./run_dataflow_tasks.sh legal legal-hard-1 legal-hard-3

if [ $# -lt 1 ]; then
    echo "Usage: $0 <task_id> [task_id ...]"
    echo "Example: $0 legal-hard-1 legal-hard-3 astronomy-easy-1"
    exit 1
fi

# Build extra arguments based on mode
EXTRA_ARGS=""
if [ "$ORACLE_MODE" = "true" ]; then
    EXTRA_ARGS="--use_truth_subset"
    echo "Running in ORACLE MODE (using ground truth subset files)"
else
    echo "Running in STANDARD MODE (using all files)"
fi

# Check if first arg is a workload name (legacy mode: workload + task_ids)
KNOWN_WORKLOADS="archeology astronomy biomedical environment legal wildfire"
FIRST_ARG="$1"
IS_LEGACY=false
for wl in $KNOWN_WORKLOADS; do
    if [ "$FIRST_ARG" = "$wl" ] || [ "$FIRST_ARG" = "${wl}-tiny" ]; then
        IS_LEGACY=true
        break
    fi
done

if [ "$IS_LEGACY" = "true" ]; then
    # Legacy mode: first arg is workload, rest are task IDs
    WORKLOAD="$1"
    shift
    TASK_IDS=("$@")

    # In legacy mode, PARALLEL defaults to false (single workload)
    if [ "$PARALLEL" = "auto" ]; then
        PARALLEL=false
    fi

    echo "=========================================="
    echo "Running: $SUT | $WORKLOAD | Tasks: ${TASK_IDS[*]}"
    echo "DATAFLOW_MAX_RESULT_CHARS=$DATAFLOW_MAX_RESULT_CHARS"
    echo "=========================================="
    python evaluate.py --sut "$SUT" --workload "$WORKLOAD" --no_pipeline_eval --verbose --task_id "${TASK_IDS[@]}" $EXTRA_ARGS
    echo ""
else
    # New mode: parse workload from task IDs, group and run per workload
    declare -A WORKLOAD_TASKS
    for task_id in "$@"; do
        # Parse workload: everything before -easy- or -hard-
        workload=$(echo "$task_id" | sed -E 's/-(easy|hard)-[0-9]+$//')
        if [ -z "$workload" ]; then
            echo "ERROR: Cannot parse workload from task ID: $task_id"
            exit 1
        fi
        WORKLOAD_TASKS[$workload]+="$task_id "
    done

    # In auto-group mode, PARALLEL defaults to true (multiple workloads)
    if [ "$PARALLEL" = "auto" ]; then
        PARALLEL=true
    fi

    # Create log directory for parallel output
    LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"

    if [ "$PARALLEL" = "true" ]; then
        echo "Running ${#WORKLOAD_TASKS[@]} workload groups in PARALLEL"
        echo "Logs: $LOG_DIR/"
        echo "=========================================="

        PIDS=()
        for workload in $(echo "${!WORKLOAD_TASKS[@]}" | tr ' ' '\n' | sort -u); do
            read -ra TASK_IDS <<< "${WORKLOAD_TASKS[$workload]}"
            echo "Starting: $SUT | $workload | Tasks: ${TASK_IDS[*]} (log: $LOG_DIR/${workload}.log)"
            python evaluate.py --sut "$SUT" --workload "$workload" --no_pipeline_eval --verbose --task_id "${TASK_IDS[@]}" $EXTRA_ARGS \
                > "$LOG_DIR/${workload}.log" 2>&1 &
            PIDS+=("$!:$workload")
        done

        echo "=========================================="
        echo "Waiting for all workloads to finish..."
        echo "=========================================="

        # Wait for each process and report status
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
            echo "All ${#WORKLOAD_TASKS[@]} workloads completed successfully!"
        else
            echo "$FAILED/${#WORKLOAD_TASKS[@]} workloads failed. Check logs in $LOG_DIR/"
        fi
        echo "=========================================="
    else
        # Sequential mode
        for workload in $(echo "${!WORKLOAD_TASKS[@]}" | tr ' ' '\n' | sort -u); do
            read -ra TASK_IDS <<< "${WORKLOAD_TASKS[$workload]}"
            echo "=========================================="
            echo "Running: $SUT | $workload | Tasks: ${TASK_IDS[*]}"
            echo "DATAFLOW_MAX_RESULT_CHARS=$DATAFLOW_MAX_RESULT_CHARS"
            echo "=========================================="
            python evaluate.py --sut "$SUT" --workload "$workload" --no_pipeline_eval --verbose --task_id "${TASK_IDS[@]}" $EXTRA_ARGS \
                | tee "$LOG_DIR/${workload}.log"
            echo ""
        done
    fi
fi

echo "=========================================="
echo "Run complete!"
echo "=========================================="
