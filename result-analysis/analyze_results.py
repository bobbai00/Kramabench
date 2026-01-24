#!/usr/bin/env python3
"""
Result Analysis Script for KramaBench O4Mini Comparison

This script:
1. Reads results from CodeAgentSystemO4Mini and DataflowSystemO4Mini
2. Categorizes tasks into 4 cases:
   - 1_both_succeed: Both systems succeeded (score >= threshold)
   - 2_dataflow_succeed: Only DataflowSystem succeeded
   - 3_code_succeed: Only CodeAgent succeeded
   - 4_both_failed: Both systems failed
3. Randomly selects 5 tasks per case
4. Copies traces to organized folders
5. Generates stats.json with counts per domain and score statistics
"""

import os
import json
import shutil
import random
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path("/Users/baijiadong/Desktop/chenlab/KramaBench")
RESULTS_DIR = BASE_DIR / "results"
SCRATCH_DIR = BASE_DIR / "system_scratch"
WORKLOAD_DIR = BASE_DIR / "workload"
OUTPUT_DIR = BASE_DIR / "result-analysis" / "o4mini"

CODE_AGENT = "CodeAgentSystemO4Mini"
DATAFLOW_SYSTEM = "DataflowSystemO4Mini"

DOMAINS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]

CASE_DIRS = {
    "both_succeed": "1_both_succeed",
    "dataflow_succeed": "2_dataflow_succeed",
    "code_succeed": "3_code_succeed",
    "both_failed": "4_both_failed"
}

NUM_SAMPLES_PER_CASE = 5

# Success threshold for partial scores (1.0 = exact match only)
DEFAULT_SUCCESS_THRESHOLD = 1.0

# Metrics priority order (first match wins)
METRIC_PRIORITY = ['success', 'f1', 'f1_approximate', 'llm_paraphrase']

# Accurate (exact) metrics - tasks using these are counted in accurate-only stats
ACCURATE_METRICS = {'success'}

# Approximate metrics - tasks using these are excluded from accurate-only stats
APPROXIMATE_METRICS = {'f1', 'f1_approximate', 'llm_paraphrase', 'rae_score',
                       'mean_absolute_error', 'mean_squared_error'}


def get_latest_result_file(system_dir: Path, domain: str) -> Path:
    """Get the latest result CSV file for a domain."""
    pattern = f"{domain}_measures_*.csv"
    files = list(system_dir.glob(pattern))
    if not files:
        return None
    # Sort by modification time (latest first)
    return max(files, key=lambda f: f.stat().st_mtime)


def extract_score_metrics(csv_path: Path) -> dict:
    """Extract score metrics from a result CSV file.

    Returns dict mapping task_id -> {
        'score': float (0.0 to 1.0),
        'metric_type': str (which metric was used),
        'is_accurate': bool (True if using exact/accurate metric),
        'all_metrics': dict (all metrics for this task)
    }
    """
    if csv_path is None or not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    score_metrics = {}

    for task_id in df['task_id'].unique():
        task_df = df[df['task_id'] == task_id]

        # Collect all metrics for this task
        all_metrics = {}
        for _, row in task_df.iterrows():
            metric_name = row['metric']
            metric_value = row['value']
            all_metrics[metric_name] = metric_value

        # Find the primary score metric using priority order
        score = 0.0
        metric_type = 'none'

        for metric_name in METRIC_PRIORITY:
            if metric_name in all_metrics:
                score = all_metrics[metric_name]
                metric_type = metric_name
                break

        # Determine if this task uses accurate (exact) evaluation
        is_accurate = metric_type in ACCURATE_METRICS

        score_metrics[task_id] = {
            'score': score,
            'metric_type': metric_type,
            'is_accurate': is_accurate,
            'all_metrics': all_metrics
        }

    return score_metrics


def extract_success_metrics(csv_path: Path, threshold: float = 1.0) -> dict:
    """Extract success metrics from a result CSV file.

    Returns dict mapping task_id -> success (True/False based on threshold)
    """
    score_metrics = extract_score_metrics(csv_path)
    return {
        task_id: data['score'] >= threshold
        for task_id, data in score_metrics.items()
    }


def load_workload(domain: str) -> dict:
    """Load workload JSON and return dict mapping task_id -> task info."""
    workload_path = WORKLOAD_DIR / f"{domain}.json"
    if not workload_path.exists():
        return {}

    with open(workload_path, 'r') as f:
        tasks = json.load(f)

    return {task['id']: task for task in tasks}


def categorize_tasks(code_results: dict, dataflow_results: dict) -> dict:
    """Categorize tasks into 4 cases based on success/failure."""
    categories = {
        "both_succeed": [],
        "dataflow_succeed": [],
        "code_succeed": [],
        "both_failed": []
    }

    all_tasks = set(code_results.keys()) | set(dataflow_results.keys())

    for task_id in all_tasks:
        code_success = code_results.get(task_id, False)
        dataflow_success = dataflow_results.get(task_id, False)

        if code_success and dataflow_success:
            categories["both_succeed"].append(task_id)
        elif dataflow_success and not code_success:
            categories["dataflow_succeed"].append(task_id)
        elif code_success and not dataflow_success:
            categories["code_succeed"].append(task_id)
        else:
            categories["both_failed"].append(task_id)

    return categories


def categorize_tasks_with_scores(code_scores: dict, dataflow_scores: dict, threshold: float) -> dict:
    """Categorize tasks with detailed score information.

    Returns dict with categories and score details for each task.
    """
    categories = {
        "both_succeed": [],
        "dataflow_succeed": [],
        "code_succeed": [],
        "both_failed": []
    }

    # Separate categories for accurate-only tasks
    accurate_categories = {
        "both_succeed": [],
        "dataflow_succeed": [],
        "code_succeed": [],
        "both_failed": []
    }

    task_details = {}
    all_tasks = set(code_scores.keys()) | set(dataflow_scores.keys())

    for task_id in all_tasks:
        code_data = code_scores.get(task_id, {'score': 0.0, 'metric_type': 'none', 'is_accurate': False})
        dataflow_data = dataflow_scores.get(task_id, {'score': 0.0, 'metric_type': 'none', 'is_accurate': False})

        code_score = code_data['score']
        dataflow_score = dataflow_data['score']
        code_success = code_score >= threshold
        dataflow_success = dataflow_score >= threshold

        # Task is accurate if BOTH systems evaluate it with accurate metrics
        # (they should use the same metric type for the same task)
        is_accurate = code_data.get('is_accurate', False) and dataflow_data.get('is_accurate', False)

        task_details[task_id] = {
            'code_score': code_score,
            'dataflow_score': dataflow_score,
            'code_metric': code_data['metric_type'],
            'dataflow_metric': dataflow_data['metric_type'],
            'code_success': code_success,
            'dataflow_success': dataflow_success,
            'is_accurate': is_accurate,
            'score_diff': dataflow_score - code_score  # positive = dataflow better
        }

        # Determine category
        if code_success and dataflow_success:
            category = "both_succeed"
        elif dataflow_success and not code_success:
            category = "dataflow_succeed"
        elif code_success and not dataflow_success:
            category = "code_succeed"
        else:
            category = "both_failed"

        categories[category].append(task_id)

        # Also add to accurate categories if applicable
        if is_accurate:
            accurate_categories[category].append(task_id)

    return categories, accurate_categories, task_details


def compute_score_statistics(task_details: dict, domains: list) -> dict:
    """Compute aggregate score statistics."""
    stats = {
        'overall': {},
        'by_domain': {}
    }

    # Overall statistics
    code_scores = [d['code_score'] for d in task_details.values()]
    dataflow_scores = [d['dataflow_score'] for d in task_details.values()]

    if code_scores:
        stats['overall']['code_agent'] = {
            'mean': sum(code_scores) / len(code_scores),
            'min': min(code_scores),
            'max': max(code_scores),
            'count': len(code_scores)
        }

    if dataflow_scores:
        stats['overall']['dataflow'] = {
            'mean': sum(dataflow_scores) / len(dataflow_scores),
            'min': min(dataflow_scores),
            'max': max(dataflow_scores),
            'count': len(dataflow_scores)
        }

    # Per-domain statistics
    for domain in domains:
        domain_code = [d['code_score'] for tid, d in task_details.items() if tid.startswith(domain)]
        domain_dataflow = [d['dataflow_score'] for tid, d in task_details.items() if tid.startswith(domain)]

        if domain_code:
            stats['by_domain'][domain] = {
                'code_agent_mean': sum(domain_code) / len(domain_code),
                'dataflow_mean': sum(domain_dataflow) / len(domain_dataflow) if domain_dataflow else 0,
                'count': len(domain_code)
            }

    return stats


def copy_task_files(task_id: str, task_info: dict, output_dir: Path):
    """Copy task files to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write prompt.txt
    prompt_path = output_dir / "prompt.txt"
    with open(prompt_path, 'w') as f:
        f.write(task_info.get('query', 'N/A'))

    # Write answer.txt
    answer_path = output_dir / "answer.txt"
    with open(answer_path, 'w') as f:
        answer = task_info.get('answer', 'N/A')
        if isinstance(answer, list):
            f.write(json.dumps(answer, indent=2))
        else:
            f.write(str(answer))

    # Copy CodeAgent trace
    code_trace_src = SCRATCH_DIR / CODE_AGENT / task_id / "reasoning_trace.json"
    code_trace_dst = output_dir / "code_trace.json"
    if code_trace_src.exists():
        shutil.copy(code_trace_src, code_trace_dst)
    else:
        # Write empty placeholder
        with open(code_trace_dst, 'w') as f:
            json.dump({"error": "Trace not found"}, f)

    # Copy DataflowSystem trace
    dataflow_trace_src = SCRATCH_DIR / DATAFLOW_SYSTEM / task_id / "messages.json"
    dataflow_trace_dst = output_dir / "dataflow_trace.json"
    if dataflow_trace_src.exists():
        shutil.copy(dataflow_trace_src, dataflow_trace_dst)
    else:
        # Write empty placeholder
        with open(dataflow_trace_dst, 'w') as f:
            json.dump({"error": "Trace not found"}, f)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze KramaBench results')
    parser.add_argument('--threshold', type=float, default=DEFAULT_SUCCESS_THRESHOLD,
                        help=f'Success threshold (0.0-1.0, default: {DEFAULT_SUCCESS_THRESHOLD})')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES_PER_CASE,
                        help=f'Number of samples per case (default: {NUM_SAMPLES_PER_CASE})')
    args = parser.parse_args()

    threshold = args.threshold
    num_samples = args.samples

    print("=" * 60)
    print("KramaBench O4Mini Result Analysis")
    print(f"Success Threshold: {threshold}")
    print("=" * 60)

    # Collect all results
    all_code_scores = {}
    all_dataflow_scores = {}
    all_workloads = {}

    code_results_dir = RESULTS_DIR / CODE_AGENT
    dataflow_results_dir = RESULTS_DIR / DATAFLOW_SYSTEM

    print("\n[1/5] Loading result metrics...")
    for domain in DOMAINS:
        print(f"  Loading {domain}...")

        # Load CodeAgent results (with scores)
        code_csv = get_latest_result_file(code_results_dir, domain)
        code_scores = extract_score_metrics(code_csv)
        all_code_scores.update(code_scores)

        # Load DataflowSystem results (with scores)
        dataflow_csv = get_latest_result_file(dataflow_results_dir, domain)
        dataflow_scores = extract_score_metrics(dataflow_csv)
        all_dataflow_scores.update(dataflow_scores)

        # Load workload
        workload = load_workload(domain)
        all_workloads.update(workload)

        print(f"    CodeAgent: {len(code_scores)} tasks, DataflowSystem: {len(dataflow_scores)} tasks")

    print(f"\n  Total: CodeAgent={len(all_code_scores)}, DataflowSystem={len(all_dataflow_scores)}")

    # Categorize tasks with score details
    print("\n[2/5] Categorizing tasks...")
    categories, accurate_categories, task_details = categorize_tasks_with_scores(
        all_code_scores, all_dataflow_scores, threshold
    )

    # Count accurate vs approximate tasks
    accurate_task_ids = {tid for tid, d in task_details.items() if d['is_accurate']}
    approximate_task_ids = {tid for tid, d in task_details.items() if not d['is_accurate']}

    print(f"\n  All tasks breakdown:")
    for cat_name, tasks in categories.items():
        print(f"    {cat_name}: {len(tasks)} tasks")

    print(f"\n  Accurate-only tasks breakdown (using 'success' metric):")
    for cat_name, tasks in accurate_categories.items():
        print(f"    {cat_name}: {len(tasks)} tasks")

    print(f"\n  Task evaluation type: {len(accurate_task_ids)} accurate, {len(approximate_task_ids)} approximate")

    # Compute score statistics
    print("\n[3/5] Computing statistics...")
    score_stats = compute_score_statistics(task_details, DOMAINS)

    # Build stats object
    stats = {
        "config": {
            "threshold": threshold,
            "code_agent": CODE_AGENT,
            "dataflow_system": DATAFLOW_SYSTEM
        },
        "total_tasks": len(all_code_scores),
        "accurate_tasks_count": len(accurate_task_ids),
        "approximate_tasks_count": len(approximate_task_ids),
        "categories": {},
        "categories_accurate_only": {},
        "domain_breakdown": {},
        "domain_breakdown_accurate_only": {},
        "score_statistics": score_stats
    }

    # All tasks categories
    for cat_name, tasks in categories.items():
        stats["categories"][cat_name] = len(tasks)

    # Accurate-only categories
    for cat_name, tasks in accurate_categories.items():
        stats["categories_accurate_only"][cat_name] = len(tasks)

    # Breakdown by domain (all tasks)
    print("\n  All tasks domain breakdown:")
    for domain in DOMAINS:
        domain_stats = {cat: 0 for cat in categories.keys()}
        for cat_name, tasks in categories.items():
            for task_id in tasks:
                if task_id.startswith(domain):
                    domain_stats[cat_name] += 1

        # Add score means for this domain
        domain_score_stats = score_stats['by_domain'].get(domain, {})
        domain_stats['code_mean_score'] = round(domain_score_stats.get('code_agent_mean', 0), 4)
        domain_stats['dataflow_mean_score'] = round(domain_score_stats.get('dataflow_mean', 0), 4)

        stats["domain_breakdown"][domain] = domain_stats
        print(f"    {domain}: {domain_stats}")

    # Breakdown by domain (accurate-only)
    print("\n  Accurate-only domain breakdown:")
    for domain in DOMAINS:
        domain_stats = {cat: 0 for cat in accurate_categories.keys()}
        for cat_name, tasks in accurate_categories.items():
            for task_id in tasks:
                if task_id.startswith(domain):
                    domain_stats[cat_name] += 1

        # Count accurate tasks in this domain
        domain_accurate_count = sum(1 for tid in accurate_task_ids if tid.startswith(domain))
        domain_stats['accurate_task_count'] = domain_accurate_count

        stats["domain_breakdown_accurate_only"][domain] = domain_stats
        print(f"    {domain}: {domain_stats}")

    # Clean up existing output directories
    print("\n[4/5] Setting up output directories...")
    for case_dir in CASE_DIRS.values():
        case_path = OUTPUT_DIR / case_dir
        # Remove existing task folders but keep the directory
        if case_path.exists():
            for item in case_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)

    # Randomly select and copy tasks (ONLY from accurate-evaluated tasks)
    print("\n[5/5] Selecting and copying representative tasks (accurate-evaluated only)...")
    random.seed(42)  # For reproducibility

    selected_tasks = {}
    for cat_name, tasks in accurate_categories.items():
        case_dir = CASE_DIRS[cat_name]
        case_path = OUTPUT_DIR / case_dir
        case_path.mkdir(parents=True, exist_ok=True)

        # Select up to num_samples random tasks from accurate-evaluated tasks only
        num_to_select = min(num_samples, len(tasks))
        selected = random.sample(tasks, num_to_select) if tasks else []

        # Include score details for selected tasks
        selected_with_scores = []
        for task_id in selected:
            details = task_details.get(task_id, {})
            selected_with_scores.append({
                'task_id': task_id,
                'code_score': details.get('code_score', 0),
                'dataflow_score': details.get('dataflow_score', 0),
                'code_metric': details.get('code_metric', 'unknown'),
                'dataflow_metric': details.get('dataflow_metric', 'unknown'),
                'is_accurate': details.get('is_accurate', False),
                'score_diff': details.get('score_diff', 0)
            })

        selected_tasks[cat_name] = selected_with_scores

        print(f"\n  {cat_name} ({num_to_select} selected from {len(tasks)} accurate tasks):")
        for item in selected_with_scores:
            task_id = item['task_id']
            task_info = all_workloads.get(task_id, {"query": "N/A", "answer": "N/A"})
            task_output_dir = case_path / task_id
            copy_task_files(task_id, task_info, task_output_dir)

            # Also write score info
            score_info_path = task_output_dir / "scores.json"
            with open(score_info_path, 'w') as f:
                json.dump({
                    'code_score': item['code_score'],
                    'dataflow_score': item['dataflow_score'],
                    'code_metric': item['code_metric'],
                    'dataflow_metric': item['dataflow_metric'],
                    'is_accurate': item['is_accurate'],
                    'score_diff': item['score_diff'],
                    'threshold': threshold,
                    'code_success': item['code_score'] >= threshold,
                    'dataflow_success': item['dataflow_score'] >= threshold
                }, f, indent=2)

            metric_str = f"[{item['code_metric']}]"
            score_str = f"(Code: {item['code_score']:.2f}, DF: {item['dataflow_score']:.2f})"
            print(f"    - {task_id} {metric_str} {score_str}")

    # Add selected tasks to stats
    stats["selected_samples"] = selected_tasks

    # Write stats.json
    stats_path = OUTPUT_DIR / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Stats file: {stats_path}")
    print(f"{'=' * 60}")

    # Print summary table - ALL TASKS
    print("\n" + "=" * 90)
    print("SUMMARY TABLE - ALL TASKS (including approximate metrics)")
    print("=" * 90)
    print(f"{'Domain':<15} {'Both OK':<10} {'DF Only':<10} {'Code Only':<10} {'Both Fail':<10} {'Code Mean':<12} {'DF Mean':<12}")
    print("-" * 90)
    for domain in DOMAINS:
        d = stats["domain_breakdown"][domain]
        print(f"{domain:<15} {d['both_succeed']:<10} {d['dataflow_succeed']:<10} {d['code_succeed']:<10} {d['both_failed']:<10} {d['code_mean_score']:<12.4f} {d['dataflow_mean_score']:<12.4f}")
    print("-" * 90)
    total = stats["categories"]
    overall = score_stats['overall']
    code_mean = overall.get('code_agent', {}).get('mean', 0)
    df_mean = overall.get('dataflow', {}).get('mean', 0)
    print(f"{'TOTAL':<15} {total['both_succeed']:<10} {total['dataflow_succeed']:<10} {total['code_succeed']:<10} {total['both_failed']:<10} {code_mean:<12.4f} {df_mean:<12.4f}")

    # Print summary table - ACCURATE ONLY
    print("\n" + "=" * 90)
    print("SUMMARY TABLE - ACCURATE ONLY (using 'success' metric)")
    print("=" * 90)
    print(f"{'Domain':<15} {'Both OK':<10} {'DF Only':<10} {'Code Only':<10} {'Both Fail':<10} {'Accurate#':<12}")
    print("-" * 90)
    for domain in DOMAINS:
        d = stats["domain_breakdown_accurate_only"][domain]
        print(f"{domain:<15} {d['both_succeed']:<10} {d['dataflow_succeed']:<10} {d['code_succeed']:<10} {d['both_failed']:<10} {d['accurate_task_count']:<12}")
    print("-" * 90)
    total_acc = stats["categories_accurate_only"]
    total_acc_count = stats["accurate_tasks_count"]
    print(f"{'TOTAL':<15} {total_acc['both_succeed']:<10} {total_acc['dataflow_succeed']:<10} {total_acc['code_succeed']:<10} {total_acc['both_failed']:<10} {total_acc_count:<12}")


if __name__ == "__main__":
    main()
