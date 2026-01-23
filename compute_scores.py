#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-workload and overall scores from aggregated_results.csv.

This script reuses the scoring logic from evaluate.py to calculate:
1. Per-workload scores for each system
2. Overall scores for each system

Usage:
    python compute_scores.py
    python compute_scores.py --output scores.csv
    python compute_scores.py --sut DataflowSystemGptO3
"""

import argparse
import os
import pandas as pd

# Score metrics used in evaluate.py (lines 54-60)
SCORE_METRICS = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]


def compute_scores_for_system(df: pd.DataFrame, system_name: str) -> pd.DataFrame:
    """
    Compute per-workload and overall scores for a single system.

    Uses the same formula as evaluate.py:
        score = sum(value_support * value_mean) / total_support * 100

    Args:
        df: DataFrame filtered to a single system
        system_name: Name of the system

    Returns:
        DataFrame with workload scores and overall score
    """
    results = []

    total_score = 0
    total_support = 0

    workloads = sorted(df['workload'].unique())

    for workload in workloads:
        wl_df = df[(df['workload'] == workload) & (df['metric'].isin(SCORE_METRICS))]

        if len(wl_df) == 0:
            continue

        # Same formula as evaluate.py lines 56-59
        wl_score = (wl_df['value_support'] * wl_df['value_mean']).sum()
        wl_support = wl_df['total_value_support'].sum()
        wl_pct = (wl_score / wl_support * 100) if wl_support > 0 else 0

        total_score += wl_score
        total_support += wl_support

        # Clean workload name (remove .json suffix)
        workload_clean = workload.replace('.json', '')

        results.append({
            'sut': system_name,
            'workload': workload_clean,
            'score': round(wl_pct, 2),
            'correct': round(wl_score, 2),
            'total': int(wl_support)
        })

    # Add overall score
    overall_pct = (total_score / total_support * 100) if total_support > 0 else 0
    results.append({
        'sut': system_name,
        'workload': 'OVERALL',
        'score': round(overall_pct, 2),
        'correct': round(total_score, 2),
        'total': int(total_support)
    })

    return pd.DataFrame(results)


def compute_all_scores(aggregated_path: str, sut_filter: str = None) -> pd.DataFrame:
    """
    Compute scores for all systems in the aggregated results.

    Args:
        aggregated_path: Path to aggregated_results.csv
        sut_filter: Optional filter to compute scores for a single system

    Returns:
        DataFrame with all scores
    """
    df = pd.read_csv(aggregated_path)

    systems = df['sut'].unique()
    if sut_filter:
        systems = [s for s in systems if s == sut_filter]
        if not systems:
            raise ValueError(f"System '{sut_filter}' not found in {aggregated_path}")

    all_scores = []
    for system_name in sorted(systems):
        system_df = df[df['sut'] == system_name]
        scores_df = compute_scores_for_system(system_df, system_name)
        all_scores.append(scores_df)

    return pd.concat(all_scores, ignore_index=True)


def print_scores_table(scores_df: pd.DataFrame) -> None:
    """Print scores in a formatted table."""
    systems = scores_df['sut'].unique()

    for system_name in systems:
        system_scores = scores_df[scores_df['sut'] == system_name]

        print(f"\n{'=' * 60}")
        print(f" {system_name}")
        print(f"{'=' * 60}")
        print(f"{'Workload':<20} {'Score':>10} {'Correct':>12} {'Total':>8}")
        print(f"{'-' * 60}")

        for _, row in system_scores.iterrows():
            workload = row['workload']
            if workload == 'OVERALL':
                print(f"{'-' * 60}")
            print(f"{workload:<20} {row['score']:>9.1f}% {row['correct']:>12.1f} {row['total']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute scores from aggregated_results.csv"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="results/aggregated_results.csv",
        help="Path to aggregated_results.csv (default: results/aggregated_results.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV path for scores (optional, prints to stdout if not specified)"
    )
    parser.add_argument(
        "--sut",
        type=str,
        default=None,
        help="Filter to a specific system under test"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Aggregated results not found at {args.input}")

    # Compute scores
    scores_df = compute_all_scores(args.input, args.sut)

    # Output results
    if args.output:
        scores_df.to_csv(args.output, index=False)
        print(f"Scores saved to {args.output}")

    # Always print to stdout
    print_scores_table(scores_df)

    # Print comparison summary if multiple systems
    systems = scores_df['sut'].unique()
    if len(systems) > 1:
        print(f"\n{'=' * 60}")
        print(" COMPARISON SUMMARY")
        print(f"{'=' * 60}")

        # Get overall scores for each system
        overall_scores = scores_df[scores_df['workload'] == 'OVERALL'][['sut', 'score']]
        overall_scores = overall_scores.sort_values('score', ascending=False)

        print(f"{'System':<35} {'Overall Score':>15}")
        print(f"{'-' * 60}")
        for _, row in overall_scores.iterrows():
            print(f"{row['sut']:<35} {row['score']:>14.1f}%")


if __name__ == "__main__":
    main()
