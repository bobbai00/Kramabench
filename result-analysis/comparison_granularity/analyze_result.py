#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze granularity comparison results.

This script compares metrics (tokens, steps, elapsed time) and accuracy between
coarse-grained and fine-grained agent executions.

Usage:
    python analyze_result.py <system_name>

Example:
    python analyze_result.py CodeAgentSystemGemini25Pro
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path to import benchmark modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import (
    Success,
    F1,
    F1Approximate,
    RAEScore,
    LLMParaphrase,
    RougeScore,
)


def load_stats(task_dir: Path) -> Optional[Dict]:
    """Load stats.json from a task directory."""
    stats_file = task_dir / "stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {stats_file}: {e}")
    return None


def load_answer(task_dir: Path) -> Optional[str]:
    """Load answer from answer.json."""
    answer_file = task_dir / "answer.json"
    if answer_file.exists():
        try:
            with open(answer_file, "r") as f:
                data = json.load(f)
                return data.get("answer", "")
        except (json.JSONDecodeError, IOError):
            pass
    return None


def load_workload(workload_name: str = "legal") -> Dict[str, Dict]:
    """Load workload file to get ground truth answers."""
    workload_path = PROJECT_ROOT / "workload" / f"{workload_name}.json"
    if not workload_path.exists():
        print(f"Warning: Workload file not found: {workload_path}")
        return {}

    with open(workload_path, "r") as f:
        tasks = json.load(f)

    return {task["id"]: task for task in tasks}


def count_effective_lines(code: str) -> int:
    """
    Count effective lines of code, excluding:
    - Empty lines
    - Comment-only lines (starting with #)
    - Print statements (lines starting with print)
    - Lines that are only whitespace
    """
    if not code:
        return 0

    effective_count = 0
    for line in code.split('\n'):
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Skip comment-only lines
        if stripped.startswith('#'):
            continue
        # Skip print statements
        if stripped.startswith('print(') or stripped.startswith('print '):
            continue
        # Count this line
        effective_count += 1

    return effective_count


def load_reasoning_trace(task_dir: Path) -> Optional[List[Dict]]:
    """Load reasoning_trace.json from a task directory."""
    trace_file = task_dir / "reasoning_trace.json"
    if trace_file.exists():
        try:
            with open(trace_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def analyze_code_stats(base_dir: Path, system_name: str) -> Dict[str, Any]:
    """
    Analyze code statistics for all tasks.

    Returns:
        Dict with per-cell and per-task code line statistics
    """
    system_dir = base_dir / system_name
    if not system_dir.exists():
        return {}

    all_cell_lines = []  # Lines per cell across all tasks
    task_stats = {}  # Per-task statistics

    for task_dir in sorted(system_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        trace = load_reasoning_trace(task_dir)
        if not trace:
            continue

        task_cell_lines = []
        for step in trace:
            code = step.get('code', '')
            if code:
                effective_lines = count_effective_lines(code)
                task_cell_lines.append(effective_lines)
                all_cell_lines.append(effective_lines)

        if task_cell_lines:
            task_stats[task_dir.name] = {
                "num_cells": len(task_cell_lines),
                "total_lines": sum(task_cell_lines),
                "avg_lines_per_cell": sum(task_cell_lines) / len(task_cell_lines),
                "min_lines": min(task_cell_lines),
                "max_lines": max(task_cell_lines),
                "cell_lines": task_cell_lines,
            }

    return {
        "all_cell_lines": all_cell_lines,
        "task_stats": task_stats,
    }


def compute_code_summary(code_stats: Dict[str, Any]) -> Dict:
    """Compute summary statistics for code analysis."""
    if not code_stats or not code_stats.get("all_cell_lines"):
        return {}

    all_lines = code_stats["all_cell_lines"]
    task_stats = code_stats["task_stats"]

    # Per-cell statistics
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def trimmed_avg(lst):
        if len(lst) <= 2:
            return avg(lst)
        sorted_lst = sorted(lst)
        return avg(sorted_lst[1:-1])

    def median(lst):
        if not lst:
            return 0
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
        return sorted_lst[n//2]

    # Per-task total lines
    task_totals = [t["total_lines"] for t in task_stats.values()]

    return {
        "per_cell": {
            "count": len(all_lines),
            "avg": avg(all_lines),
            "trimmed_avg": trimmed_avg(all_lines),
            "median": median(all_lines),
            "min": min(all_lines) if all_lines else 0,
            "max": max(all_lines) if all_lines else 0,
        },
        "per_task": {
            "count": len(task_totals),
            "avg": avg(task_totals),
            "trimmed_avg": trimmed_avg(task_totals),
            "median": median(task_totals),
            "min": min(task_totals) if task_totals else 0,
            "max": max(task_totals) if task_totals else 0,
        },
    }


def get_metric_for_answer_type(answer_type: str):
    """Get the appropriate metric based on answer_type."""
    metric_map = {
        "numeric_exact": Success(),
        "numeric_approximate": RAEScore(),
        "string_exact": Success(),
        "string_approximate": LLMParaphrase(),
        "list_exact": F1(),
        "list_approximate": F1Approximate(),
    }
    return metric_map.get(answer_type, Success())


def evaluate_task(predicted: Any, target: Any, answer_type: str) -> Tuple[float, str]:
    """
    Evaluate a single task prediction against ground truth.

    Returns:
        Tuple of (score, metric_name)
    """
    metric = get_metric_for_answer_type(answer_type)
    try:
        result = metric(predicted, target)
        score = result[0] if result[0] is not None else 0.0
        return (score, metric.name)
    except Exception as e:
        print(f"Warning: Metric evaluation failed: {e}")
        return (0.0, metric.name)


def collect_stats(base_dir: Path, system_name: str) -> Dict[str, Dict]:
    """Collect stats for all tasks in a directory."""
    system_dir = base_dir / system_name
    if not system_dir.exists():
        print(f"Warning: Directory not found: {system_dir}")
        return {}

    stats = {}
    for task_dir in sorted(system_dir.iterdir()):
        if task_dir.is_dir():
            task_stats = load_stats(task_dir)
            if task_stats:
                stats[task_dir.name] = task_stats
    return stats


def collect_answers(base_dir: Path, system_name: str) -> Dict[str, Any]:
    """Collect answers for all tasks in a directory."""
    system_dir = base_dir / system_name
    if not system_dir.exists():
        return {}

    answers = {}
    for task_dir in sorted(system_dir.iterdir()):
        if task_dir.is_dir():
            answer = load_answer(task_dir)
            if answer is not None:
                answers[task_dir.name] = answer
    return answers


def evaluate_accuracy(
    answers: Dict[str, Any],
    workload: Dict[str, Dict],
    verbose: bool = False
) -> Dict[str, Dict]:
    """
    Evaluate accuracy for all tasks.

    Returns:
        Dict with task_id -> {score, metric, difficulty, answer_type}
    """
    results = {}

    for task_id, predicted in answers.items():
        if task_id not in workload:
            if verbose:
                print(f"Warning: No ground truth for {task_id}")
            continue

        task = workload[task_id]
        target = task.get("answer")
        answer_type = task.get("answer_type", "string_exact")

        # Determine difficulty from task_id
        difficulty = "hard" if "-hard-" in task_id else "easy"

        score, metric_name = evaluate_task(predicted, target, answer_type)

        results[task_id] = {
            "score": score,
            "metric": metric_name,
            "difficulty": difficulty,
            "answer_type": answer_type,
            "predicted": str(predicted)[:100],
            "target": str(target)[:100],
        }

        if verbose:
            status = "PASS" if score >= 0.99 else ("PARTIAL" if score > 0 else "FAIL")
            print(f"  {task_id}: {status} (score={score:.2f}, type={answer_type})")

    return results


def compute_accuracy_summary(eval_results: Dict[str, Dict]) -> Dict:
    """Compute accuracy summary by difficulty."""
    if not eval_results:
        return {}

    # Split by difficulty
    easy_scores = [r["score"] for r in eval_results.values() if r["difficulty"] == "easy"]
    hard_scores = [r["score"] for r in eval_results.values() if r["difficulty"] == "hard"]
    all_scores = [r["score"] for r in eval_results.values()]

    # Count passes (score >= 0.99 for exact match metrics)
    easy_passes = sum(1 for s in easy_scores if s >= 0.99)
    hard_passes = sum(1 for s in hard_scores if s >= 0.99)
    all_passes = sum(1 for s in all_scores if s >= 0.99)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "all": {
            "count": len(all_scores),
            "passes": all_passes,
            "accuracy": all_passes / len(all_scores) if all_scores else 0,
            "avg_score": avg(all_scores),
        },
        "easy": {
            "count": len(easy_scores),
            "passes": easy_passes,
            "accuracy": easy_passes / len(easy_scores) if easy_scores else 0,
            "avg_score": avg(easy_scores),
        },
        "hard": {
            "count": len(hard_scores),
            "passes": hard_passes,
            "accuracy": hard_passes / len(hard_scores) if hard_scores else 0,
            "avg_score": avg(hard_scores),
        },
    }


def compute_summary(stats: Dict[str, Dict]) -> Dict:
    """Compute summary statistics."""
    if not stats:
        return {}

    input_tokens = [s.get("input_tokens", 0) for s in stats.values()]
    output_tokens = [s.get("output_tokens", 0) for s in stats.values()]
    total_tokens = [s.get("total_tokens", 0) for s in stats.values()]
    num_steps = [s.get("num_steps", 0) for s in stats.values()]
    elapsed = [s.get("elapsed_seconds", 0) for s in stats.values()]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def trimmed_avg(lst):
        """Average excluding min and max values."""
        if len(lst) <= 2:
            return avg(lst)
        sorted_lst = sorted(lst)
        trimmed = sorted_lst[1:-1]  # Remove min and max
        return sum(trimmed) / len(trimmed) if trimmed else 0

    def median(lst):
        if not lst:
            return 0
        sorted_lst = sorted(lst)
        n = len(sorted_lst)
        if n % 2 == 0:
            return (sorted_lst[n//2 - 1] + sorted_lst[n//2]) / 2
        return sorted_lst[n//2]

    def compute_metric_stats(lst):
        return {
            "sum": sum(lst),
            "avg": avg(lst),
            "trimmed_avg": trimmed_avg(lst),
            "median": median(lst),
            "min": min(lst),
            "max": max(lst),
        }

    return {
        "count": len(stats),
        "input_tokens": compute_metric_stats(input_tokens),
        "output_tokens": compute_metric_stats(output_tokens),
        "total_tokens": compute_metric_stats(total_tokens),
        "num_steps": compute_metric_stats(num_steps),
        "elapsed_seconds": compute_metric_stats(elapsed),
    }


def print_summary(name: str, summary: Dict, accuracy: Dict = None, code_summary: Dict = None):
    """Print a formatted summary."""
    if not summary:
        print(f"\n{name}: No data available")
        return

    print(f"\n{'='*70}")
    print(f"{name} (n={summary['count']} tasks)")
    print(f"{'='*70}")

    # Print accuracy if available
    if accuracy:
        print("\nACCURACY:")
        print(f"  Overall:  {accuracy['all']['passes']}/{accuracy['all']['count']} = {accuracy['all']['accuracy']*100:.1f}% (avg_score={accuracy['all']['avg_score']:.2f})")
        print(f"  Easy:     {accuracy['easy']['passes']}/{accuracy['easy']['count']} = {accuracy['easy']['accuracy']*100:.1f}% (avg_score={accuracy['easy']['avg_score']:.2f})")
        print(f"  Hard:     {accuracy['hard']['passes']}/{accuracy['hard']['count']} = {accuracy['hard']['accuracy']*100:.1f}% (avg_score={accuracy['hard']['avg_score']:.2f})")
        print()

    # Print code statistics if available
    if code_summary:
        print("CODE STATISTICS (effective lines, excl. print/comments):")
        pc = code_summary["per_cell"]
        pt = code_summary["per_task"]
        print(f"  Per Cell:  avg={pc['avg']:.1f}, trimmed_avg={pc['trimmed_avg']:.1f}, median={pc['median']:.1f}, min={pc['min']}, max={pc['max']} (n={pc['count']} cells)")
        print(f"  Per Task:  avg={pt['avg']:.1f}, trimmed_avg={pt['trimmed_avg']:.1f}, median={pt['median']:.1f}, min={pt['min']}, max={pt['max']} (n={pt['count']} tasks)")
        print()

    metrics = [
        ("Input Tokens", "input_tokens"),
        ("Output Tokens", "output_tokens"),
        ("Total Tokens", "total_tokens"),
        ("Num Steps", "num_steps"),
        ("Elapsed (sec)", "elapsed_seconds"),
    ]

    print(f"{'Metric':<15} {'Avg':>12} {'Trimmed Avg':>12} {'Median':>12} {'Min':>10} {'Max':>10}")
    print("-" * 71)

    for label, key in metrics:
        s = summary[key]
        if key == "elapsed_seconds":
            print(f"{label:<15} {s['avg']:>12.1f} {s['trimmed_avg']:>12.1f} {s['median']:>12.1f} {s['min']:>10.1f} {s['max']:>10.1f}")
        else:
            print(f"{label:<15} {s['avg']:>12.1f} {s['trimmed_avg']:>12.1f} {s['median']:>12.1f} {s['min']:>10,} {s['max']:>10,}")


def print_comparison(coarse_summary: Dict, fine_summary: Dict, coarse_accuracy: Dict = None, fine_accuracy: Dict = None, coarse_code: Dict = None, fine_code: Dict = None):
    """Print comparison between coarse and fine-grained results."""
    if not coarse_summary or not fine_summary:
        print("\nCannot compute comparison - missing data")
        return

    print(f"\n{'='*70}")
    print("COMPARISON: Fine-Grained vs Coarse-Grained")
    print(f"{'='*70}")

    # Accuracy comparison
    if coarse_accuracy and fine_accuracy:
        print("\nACCURACY COMPARISON:")
        print(f"{'Category':<12} {'Coarse':>15} {'Fine':>15} {'Diff':>12}")
        print("-" * 54)

        for cat in ["all", "easy", "hard"]:
            c_acc = coarse_accuracy[cat]["accuracy"] * 100
            f_acc = fine_accuracy[cat]["accuracy"] * 100
            diff = f_acc - c_acc
            c_str = f"{coarse_accuracy[cat]['passes']}/{coarse_accuracy[cat]['count']} ({c_acc:.1f}%)"
            f_str = f"{fine_accuracy[cat]['passes']}/{fine_accuracy[cat]['count']} ({f_acc:.1f}%)"
            print(f"{cat.upper():<12} {c_str:>15} {f_str:>15} {diff:>+11.1f}%")
        print()

    # Code statistics comparison
    if coarse_code and fine_code:
        print("CODE COMPARISON (effective lines, excl. print/comments):")
        print(f"{'Metric':<20} {'Coarse':>12} {'Fine':>12} {'Diff':>12} {'Ratio':>10}")
        print("-" * 66)

        # Per cell comparison
        c_cell = coarse_code["per_cell"]["trimmed_avg"]
        f_cell = fine_code["per_cell"]["trimmed_avg"]
        diff = f_cell - c_cell
        ratio = f_cell / c_cell if c_cell > 0 else float('inf')
        print(f"{'Lines/Cell (trim)':20} {c_cell:>12.1f} {f_cell:>12.1f} {diff:>+12.1f} {ratio:>10.2f}x")

        # Per task comparison
        c_task = coarse_code["per_task"]["trimmed_avg"]
        f_task = fine_code["per_task"]["trimmed_avg"]
        diff = f_task - c_task
        ratio = f_task / c_task if c_task > 0 else float('inf')
        print(f"{'Lines/Task (trim)':20} {c_task:>12.1f} {f_task:>12.1f} {diff:>+12.1f} {ratio:>10.2f}x")

        # Total cells
        c_cells = coarse_code["per_cell"]["count"]
        f_cells = fine_code["per_cell"]["count"]
        diff = f_cells - c_cells
        ratio = f_cells / c_cells if c_cells > 0 else float('inf')
        print(f"{'Total Cells':20} {c_cells:>12} {f_cells:>12} {diff:>+12} {ratio:>10.2f}x")
        print()

    metrics = [
        ("Input Tokens", "input_tokens"),
        ("Output Tokens", "output_tokens"),
        ("Total Tokens", "total_tokens"),
        ("Num Steps", "num_steps"),
        ("Elapsed (sec)", "elapsed_seconds"),
    ]

    print("RESOURCE COMPARISON (using trimmed avg - excludes min/max):")
    print(f"{'Metric':<15} {'Coarse':>12} {'Fine':>12} {'Diff':>12} {'Ratio':>10}")
    print("-" * 61)

    for label, key in metrics:
        coarse_avg = coarse_summary[key]["trimmed_avg"]
        fine_avg = fine_summary[key]["trimmed_avg"]
        diff = fine_avg - coarse_avg
        ratio = fine_avg / coarse_avg if coarse_avg > 0 else float('inf')

        if key == "elapsed_seconds":
            print(f"{label:<15} {coarse_avg:>12.1f} {fine_avg:>12.1f} {diff:>+12.1f} {ratio:>10.2f}x")
        else:
            print(f"{label:<15} {coarse_avg:>12.1f} {fine_avg:>12.1f} {diff:>+12.1f} {ratio:>10.2f}x")


def print_per_task_comparison(coarse_stats: Dict, fine_stats: Dict, coarse_eval: Dict = None, fine_eval: Dict = None):
    """Print per-task comparison."""
    # Find common tasks
    common_tasks = sorted(set(coarse_stats.keys()) & set(fine_stats.keys()))

    if not common_tasks:
        print("\nNo common tasks to compare")
        return

    print(f"\n{'='*100}")
    print("PER-TASK COMPARISON (showing common tasks)")
    print(f"{'='*100}")

    has_eval = coarse_eval and fine_eval

    if has_eval:
        print(f"{'Task':<20} {'C-Steps':>8} {'F-Steps':>8} {'C-Tokens':>10} {'F-Tokens':>10} {'C-Score':>8} {'F-Score':>8}")
        print("-" * 82)
    else:
        print(f"{'Task':<20} {'Coarse Steps':>12} {'Fine Steps':>12} {'Coarse Tokens':>14} {'Fine Tokens':>14}")
        print("-" * 72)

    for task in common_tasks:
        c = coarse_stats[task]
        f = fine_stats[task]

        if has_eval:
            c_score = coarse_eval.get(task, {}).get("score", 0)
            f_score = fine_eval.get(task, {}).get("score", 0)
            print(f"{task:<20} {c.get('num_steps', 0):>8} {f.get('num_steps', 0):>8} {c.get('total_tokens', 0):>10,} {f.get('total_tokens', 0):>10,} {c_score:>8.2f} {f_score:>8.2f}")
        else:
            print(f"{task:<20} {c.get('num_steps', 0):>12} {f.get('num_steps', 0):>12} {c.get('total_tokens', 0):>14,} {f.get('total_tokens', 0):>14,}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze granularity comparison results"
    )
    parser.add_argument(
        "system_name",
        help="Name of the system to analyze (e.g., CodeAgentSystemGemini25Pro)"
    )
    parser.add_argument(
        "--workload",
        "-w",
        default="legal",
        help="Workload name to load ground truth from (default: legal)"
    )
    parser.add_argument(
        "--per-task",
        action="store_true",
        help="Show per-task comparison"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose evaluation output"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip accuracy evaluation"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for JSON results"
    )

    args = parser.parse_args()

    # Determine script directory
    script_dir = Path(__file__).parent
    coarse_dir = script_dir / "coarse-grained"
    fine_dir = script_dir / "fine-grained"

    # Collect stats
    print(f"Analyzing system: {args.system_name}")
    print(f"Coarse-grained dir: {coarse_dir / args.system_name}")
    print(f"Fine-grained dir: {fine_dir / args.system_name}")

    coarse_stats = collect_stats(coarse_dir, args.system_name)
    fine_stats = collect_stats(fine_dir, args.system_name)

    print(f"\nFound {len(coarse_stats)} coarse-grained tasks")
    print(f"Found {len(fine_stats)} fine-grained tasks")

    # Compute summaries
    coarse_summary = compute_summary(coarse_stats)
    fine_summary = compute_summary(fine_stats)

    # Evaluate accuracy if not skipped
    coarse_accuracy = None
    fine_accuracy = None
    coarse_eval = None
    fine_eval = None

    if not args.no_eval:
        print(f"\nLoading workload: {args.workload}")
        workload = load_workload(args.workload)
        print(f"Loaded {len(workload)} tasks from workload")

        # Collect answers
        coarse_answers = collect_answers(coarse_dir, args.system_name)
        fine_answers = collect_answers(fine_dir, args.system_name)

        print(f"\nEvaluating coarse-grained accuracy...")
        coarse_eval = evaluate_accuracy(coarse_answers, workload, args.verbose)
        coarse_accuracy = compute_accuracy_summary(coarse_eval)

        print(f"\nEvaluating fine-grained accuracy...")
        fine_eval = evaluate_accuracy(fine_answers, workload, args.verbose)
        fine_accuracy = compute_accuracy_summary(fine_eval)

    # Analyze code statistics
    print(f"\nAnalyzing code statistics...")
    coarse_code_stats = analyze_code_stats(coarse_dir, args.system_name)
    fine_code_stats = analyze_code_stats(fine_dir, args.system_name)
    coarse_code_summary = compute_code_summary(coarse_code_stats)
    fine_code_summary = compute_code_summary(fine_code_stats)

    # Print summaries
    print_summary("COARSE-GRAINED", coarse_summary, coarse_accuracy, coarse_code_summary)
    print_summary("FINE-GRAINED", fine_summary, fine_accuracy, fine_code_summary)

    # Print comparison
    print_comparison(coarse_summary, fine_summary, coarse_accuracy, fine_accuracy, coarse_code_summary, fine_code_summary)

    # Per-task comparison if requested
    if args.per_task:
        print_per_task_comparison(coarse_stats, fine_stats, coarse_eval, fine_eval)

    # Save to JSON if requested
    if args.output:
        output_data = {
            "system_name": args.system_name,
            "workload": args.workload,
            "coarse_grained": {
                "stats": coarse_stats,
                "summary": coarse_summary,
                "accuracy": coarse_accuracy,
                "evaluation": coarse_eval,
                "code_summary": coarse_code_summary,
            },
            "fine_grained": {
                "stats": fine_stats,
                "summary": fine_summary,
                "accuracy": fine_accuracy,
                "evaluation": fine_eval,
                "code_summary": fine_code_summary,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
