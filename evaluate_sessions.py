#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session-Based Benchmark Evaluation for KramaBench.

This script runs session-based benchmarks where multiple queries (subtasks)
are executed sequentially while maintaining agent state between queries.

Usage:
    python evaluate_sessions.py --system CodeAgentSession --workload environment
    python evaluate_sessions.py --system DataflowAgentSession --workload biomedical --verbose

The session-based benchmark evaluates:
1. Individual subtask accuracy (chain_accuracy)
2. Final answer accuracy (final_score)
3. Full session success (all_correct)
4. Error propagation (first_error_step)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from benchmark.session import SessionResult, SessionBenchmarkResult
from benchmark.session_runner import SessionRunner
from benchmark.session_evaluator import SessionEvaluator
from systems.code_agent_session import CodeAgentSessionRunner, CodeAgentSessionSystem
from systems.dataflow_agent_session import DataflowAgentSessionRunner, DataflowAgentSessionSystem


# Available systems
SYSTEMS = {
    "CodeAgentSession": CodeAgentSessionSystem,
    "DataflowAgentSession": DataflowAgentSessionSystem,
}

# Paths
BASE_DIR = Path(__file__).parent
WORKLOAD_DIR = BASE_DIR / "workload"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SCRATCH_DIR = BASE_DIR / "system_scratch"


def load_workload(workload_name: str) -> List[Dict[str, Any]]:
    """
    Load tasks from a workload JSON file.

    Args:
        workload_name: Name of the workload (e.g., "environment", "biomedical")

    Returns:
        List of task dictionaries
    """
    # Handle -tiny suffix
    base_workload = workload_name.replace("-tiny", "")
    workload_path = WORKLOAD_DIR / f"{base_workload}.json"

    if not workload_path.exists():
        raise FileNotFoundError(f"Workload not found: {workload_path}")

    with open(workload_path) as f:
        tasks = json.load(f)

    # If -tiny, return only first 3 tasks
    if "-tiny" in workload_name:
        tasks = tasks[:3]

    return tasks


def filter_tasks_with_subtasks(tasks: List[Dict]) -> List[Dict]:
    """
    Filter to only tasks that have subtasks.

    Args:
        tasks: List of all tasks

    Returns:
        List of tasks with non-empty subtasks
    """
    return [t for t in tasks if t.get("subtasks")]


def save_session_result(
    result: SessionResult,
    system_name: str,
    output_dir: Path,
) -> None:
    """
    Save individual session result to file.

    Args:
        result: SessionResult to save
        system_name: Name of the system
        output_dir: Directory to save results
    """
    task_dir = output_dir / result.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save result JSON
    result_path = task_dir / "session_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def save_benchmark_result(
    result: SessionBenchmarkResult,
    output_dir: Path,
) -> Path:
    """
    Save aggregated benchmark results.

    Args:
        result: SessionBenchmarkResult to save
        output_dir: Directory to save results

    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_benchmark_{result.workload}_{timestamp}.json"
    result_path = output_dir / filename

    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    return result_path


def print_session_summary(result: SessionResult) -> None:
    """Print summary of a single session."""
    status = "✓" if result.all_correct else "✗"
    print(f"\n  {status} {result.task_id}")
    print(f"    Subtasks: {result.subtasks_correct}/{result.subtasks_total} correct ({result.chain_accuracy:.1%})")
    print(f"    Final: {result.final_score:.2f}")
    if result.first_error_step:
        print(f"    First error: step {result.first_error_step}")
    if result.session_error:
        print(f"    Error: {result.session_error[:50]}...")


def print_benchmark_summary(result: SessionBenchmarkResult) -> None:
    """Print summary of benchmark results."""
    print("\n" + "=" * 70)
    print("SESSION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"System: {result.system_name}")
    print(f"Workload: {result.workload}")
    print(f"Total sessions: {result.total_sessions}")
    print()
    print("Session-Level Metrics:")
    print(f"  All correct (subtasks + final): {result.sessions_all_correct}/{result.total_sessions} ({result.sessions_all_correct/result.total_sessions:.1%})")
    print(f"  Final answer correct: {result.sessions_final_correct}/{result.total_sessions} ({result.sessions_final_correct/result.total_sessions:.1%})")
    print(f"  Average chain accuracy: {result.avg_chain_accuracy:.1%}")
    print(f"  Average final score: {result.avg_final_score:.2f}")
    print()
    print("Subtask-Level Metrics:")
    print(f"  Total subtasks: {result.total_subtasks}")
    print(f"  Subtasks correct: {result.subtasks_correct}")
    print(f"  Subtask accuracy: {result.subtask_accuracy:.1%}")
    print()
    print(f"Total time: {result.total_elapsed_seconds:.1f}s")
    print("=" * 70)


def run_session_benchmark(
    system_name: str,
    workload: str,
    model_type: Optional[str] = None,
    max_steps: int = 50,
    success_threshold: float = 0.9,
    verbose: bool = False,
    task_ids: Optional[List[str]] = None,
) -> SessionBenchmarkResult:
    """
    Run session-based benchmark.

    Args:
        system_name: Name of the system to evaluate
        workload: Name of the workload
        model_type: LLM model type (optional, uses system default)
        max_steps: Maximum steps per query
        success_threshold: Threshold for considering answer correct
        verbose: Enable verbose output
        task_ids: Specific task IDs to run (optional, runs all if None)

    Returns:
        SessionBenchmarkResult with all results
    """
    # Initialize system
    if system_name not in SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(SYSTEMS.keys())}")

    system_class = SYSTEMS[system_name]
    system_kwargs = {"max_steps": max_steps, "verbose": verbose}
    if model_type:
        system_kwargs["model_type"] = model_type

    system = system_class(**system_kwargs)

    # Set dataset directory
    base_workload = workload.replace("-tiny", "")
    dataset_dir = DATA_DIR / base_workload / "input"
    if dataset_dir.exists():
        system.process_dataset(str(dataset_dir))
    else:
        # Some workloads may not have separate input dirs
        system.process_dataset(str(DATA_DIR / base_workload))

    # Create output directory
    output_dir = SCRATCH_DIR / f"{system_name}_sessions" / workload
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter tasks
    tasks = load_workload(workload)
    tasks = filter_tasks_with_subtasks(tasks)

    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]

    print(f"\nRunning session benchmark: {system_name} on {workload}")
    print(f"Tasks with subtasks: {len(tasks)}")
    print(f"Success threshold: {success_threshold}")

    # Initialize evaluator
    evaluator = SessionEvaluator(success_threshold=success_threshold)

    # Run sessions
    session_results = []
    benchmark_start = time.time()

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] Session: {task['id']}")
        print(f"Subtasks: {len(task.get('subtasks', []))}")
        print(f"{'='*60}")

        # Create fresh runner for each session
        runner = system.create_runner()

        # Run session
        result = runner.run_session(task)

        # Evaluate
        result = evaluator.evaluate_session(result, task)

        # Save individual result
        save_session_result(result, system_name, output_dir)

        # Print summary
        print_session_summary(result)

        session_results.append(result)

    # Aggregate results
    benchmark_result = evaluator.aggregate_results(
        session_results=session_results,
        system_name=system.name,
        workload=workload,
    )
    benchmark_result.total_elapsed_seconds = time.time() - benchmark_start

    # Save aggregated results
    results_output_dir = RESULTS_DIR / f"{system_name}_sessions"
    results_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_benchmark_result(benchmark_result, results_output_dir)

    print(f"\nResults saved to: {result_path}")
    print_benchmark_summary(benchmark_result)

    return benchmark_result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run session-based benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_sessions.py --system CodeAgentSession --workload environment
  python evaluate_sessions.py --system DataflowAgentSession --workload biomedical --verbose
  python evaluate_sessions.py --system CodeAgentSession --workload legal-tiny --model gpt-4o
        """,
    )

    parser.add_argument(
        "--system",
        choices=list(SYSTEMS.keys()),
        required=True,
        help="System to evaluate",
    )
    parser.add_argument(
        "--workload",
        required=True,
        help="Workload name (e.g., environment, biomedical, legal-tiny)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model type (uses system default if not specified)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per query (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Success threshold (default: 0.9)",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Specific task ID to run (can be specified multiple times)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        run_session_benchmark(
            system_name=args.system,
            workload=args.workload,
            model_type=args.model,
            max_steps=args.max_steps,
            success_threshold=args.threshold,
            verbose=args.verbose,
            task_ids=args.tasks,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
