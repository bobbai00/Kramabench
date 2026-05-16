#!/usr/bin/env python3
"""
List failed tasks for a given KramaBench System Under Test (SUT), grouped by workload.

Sources of truth:
  - results/{SUT}/{workload}_measures_*.csv  (latest timestamp)  -> per-(task,metric) scores
  - workload/{workload}.json                                     -> answer_type per task
  - system_scratch/{SUT}/{task_id}/                              -> agent crash / empty-answer signals

A task is reported as "failed" when its primary score metric for the task's
answer_type is below --threshold (default 1.0 -> anything less than perfect),
OR when system_scratch artifacts show the agent did not produce a real answer.

Each failed task gets a reason:
  - execution_error : answer.json contains "Error: ..." (agent raised in serve_query)
  - empty_answer    : response.txt empty or "(empty response)"
  - wrong_answer    : primary score == 0 with a real answer
  - partial         : 0 < primary score < threshold (only meaningful for f1 / rae_score / f1_approximate)
  - missing_eval    : task is in the workload but no measures row exists

Only top-level tasks are listed (subtask rows in the measures CSV are skipped).

Usage:
  python scripts/list_failed_tasks.py --sut DataflowSystemHaiku45
  python scripts/list_failed_tasks.py --sut DataflowSystem --workload legal
  python scripts/list_failed_tasks.py --sut DataflowSystem --errors-only
  python scripts/list_failed_tasks.py --sut DataflowSystem --threshold 0.5 --json
"""

import argparse
import datetime
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

PRIMARY_METRIC = {
    "numeric_exact": "success",
    "string_exact": "success",
    "numeric_approximate": "rae_score",
    "string_approximate": "llm_paraphrase",
    "list_exact": "f1",
    "list_approximate": "f1_approximate",
}

DEFAULT_WORKLOADS = [
    "archeology", "astronomy", "biomedical",
    "environment", "legal", "wildfire",
]


def find_latest_measures(results_dir: str, sut: str, workload: str) -> Optional[str]:
    pattern = os.path.join(results_dir, sut, f"{workload}_measures_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None

    def ts(p: str) -> datetime.datetime:
        s = os.path.basename(p).rsplit("_measures_", 1)[1].replace(".csv", "")
        return datetime.datetime.strptime(s, "%Y%m%d_%H%M%S")

    files.sort(key=ts, reverse=True)
    return files[0]


def load_workload_answer_types(project_root: str, workload: str) -> Dict[str, str]:
    path = os.path.join(project_root, "workload", f"{workload}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        tasks = json.load(f)
    return {t["id"]: t.get("answer_type", "") for t in tasks}


def classify_scratch(scratch_dir: str) -> Optional[str]:
    """Inspect system_scratch artifacts; return reason or None if nothing fired."""
    answer_path = os.path.join(scratch_dir, "answer.json")
    if os.path.exists(answer_path):
        try:
            with open(answer_path) as f:
                ans = json.load(f)
            val = ans.get("answer", "")
            if isinstance(val, str) and val.startswith("Error:"):
                return "execution_error"
        except Exception:
            pass

    response_path = os.path.join(scratch_dir, "response.txt")
    if os.path.exists(response_path):
        try:
            with open(response_path) as f:
                txt = f.read().strip()
            if not txt or txt == "(empty response)":
                return "empty_answer"
        except Exception:
            pass
    return None


def collect_failures(
    project_root: str,
    sut: str,
    workload: str,
    threshold: float,
    errors_only: bool,
) -> List[Dict]:
    answer_types = load_workload_answer_types(project_root, workload)
    if not answer_types:
        return []

    measures_path = find_latest_measures(os.path.join(project_root, "results"), sut, workload)
    measures = None
    if measures_path:
        measures = pd.read_csv(measures_path)
        measures["value_num"] = pd.to_numeric(measures["value"], errors="coerce")

    failures: List[Dict] = []
    seen = set()

    if measures is not None:
        for tid in measures["task_id"].unique():
            if tid not in answer_types:
                continue  # skip subtasks
            seen.add(tid)
            atype = answer_types[tid]
            primary = PRIMARY_METRIC.get(atype)
            row = measures[(measures["task_id"] == tid) & (measures["metric"] == primary)] \
                if primary else None
            score = None
            if row is not None and not row.empty:
                v = row["value_num"].iloc[0]
                score = None if pd.isna(v) else float(v)

            scratch_dir = os.path.join(project_root, "system_scratch", sut, tid)
            scratch_reason = classify_scratch(scratch_dir) if os.path.isdir(scratch_dir) else None

            if scratch_reason is not None:
                reason = scratch_reason
            elif score is None:
                reason = "missing_eval"
            elif score >= threshold:
                continue
            elif score == 0:
                reason = "wrong_answer"
            else:
                reason = "partial"

            if errors_only and reason not in ("execution_error", "empty_answer"):
                continue

            failures.append({
                "workload": workload,
                "task_id": tid,
                "answer_type": atype,
                "primary_metric": primary,
                "score": score,
                "reason": reason,
            })

    for tid, atype in answer_types.items():
        if tid in seen:
            continue
        scratch_dir = os.path.join(project_root, "system_scratch", sut, tid)
        scratch_reason = classify_scratch(scratch_dir) if os.path.isdir(scratch_dir) else None
        reason = scratch_reason or "missing_eval"
        if errors_only and reason not in ("execution_error", "empty_answer"):
            continue
        failures.append({
            "workload": workload,
            "task_id": tid,
            "answer_type": atype,
            "primary_metric": PRIMARY_METRIC.get(atype),
            "score": None,
            "reason": reason,
        })

    return failures


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--sut", required=True, help="System name (e.g. DataflowSystemHaiku45)")
    ap.add_argument("--workload", default=None,
                    help="Single workload; default: all six built-in workloads")
    ap.add_argument("--project-root", default=os.getcwd(),
                    help="Project root (default: cwd)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="Primary-metric score strictly below this counts as failure (default 1.0)")
    ap.add_argument("--errors-only", action="store_true",
                    help="Only report execution_error / empty_answer")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of text")
    args = ap.parse_args()

    workloads = [args.workload] if args.workload else DEFAULT_WORKLOADS
    all_failures: List[Dict] = []
    for wl in workloads:
        all_failures.extend(collect_failures(
            args.project_root, args.sut, wl, args.threshold, args.errors_only
        ))

    if args.json:
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for f in all_failures:
            grouped[f["workload"]].append(f)
        print(json.dumps(grouped, indent=2))
        return

    if not all_failures:
        scope = args.workload or "any workload"
        print(f"No failed tasks for SUT '{args.sut}' on {scope} (threshold < {args.threshold}).")
        return

    by_workload: Dict[str, List[Dict]] = defaultdict(list)
    for f in all_failures:
        by_workload[f["workload"]].append(f)

    print(f"Failed tasks for SUT: {args.sut}  (score < {args.threshold})")
    if args.errors_only:
        print("  filter: execution_error / empty_answer only")
    print()

    reason_totals: Dict[str, int] = defaultdict(int)
    for wl in sorted(by_workload):
        rows = by_workload[wl]
        print(f"== {wl} ({len(rows)} failed) ==")
        for f in rows:
            reason_totals[f["reason"]] += 1
            score_str = f"{f['score']:.3f}" if isinstance(f["score"], float) else "n/a"
            metric = f["primary_metric"] or "-"
            print(f"  {f['task_id']:<32s} reason={f['reason']:<16s} "
                  f"{metric}={score_str}  ({f['answer_type']})")
        print()

    print(f"Total failed: {len(all_failures)} across {len(by_workload)} workload(s)")
    print("By reason: " + ", ".join(f"{k}={v}" for k, v in sorted(reason_totals.items())))


if __name__ == "__main__":
    main()
