#!/usr/bin/env python
"""Re-score stored answers with the current metric stack — no SUT, no agent.

Same scoring code path as `evaluate.py` (benchmark.Evaluator._evaluate_result_for_task),
but it reads each task's `answer.json` out of system_scratch/<SUT>/<task>/ instead of
booting the system under test. Use it after changing a metric or the judge so every
arm is scored by one consistent function.

Writes, per (SUT, workload):
  - system_scratch/<SUT>/<task>/evaluation.json   (same shape evaluate.py writes)
  - results/<SUT>/<workload>_measures_<ts>.csv

Does NOT touch results/aggregated_results.csv — run compute_scores.py for that.

    python scripts/rescore_from_scratch.py --sut ARM [ARM ...] [--workload legal ...]
"""
import argparse
import datetime
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.benchmark import Evaluator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOADS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]


def load_answer(sut, task_id):
    p = os.path.join(ROOT, "system_scratch", sut, task_id, "answer.json")
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def rescore(sut, workload, verbose=False):
    workload_path = os.path.join(ROOT, f"workload/{workload}.json")
    with open(workload_path) as f:
        tasks = json.load(f)

    ev = Evaluator(
        workload_path=workload_path,
        task_fixture_directory=os.path.join(ROOT, "benchmark/fixtures"),
        results_directory=os.path.join(ROOT, "results", sut),
        run_subtasks=False,
        evaluate_pipeline=False,
    )

    rows, n_scored, n_missing = [], 0, 0
    for task in tasks:
        tid = task["id"]
        ans = load_answer(sut, tid)
        if ans is None:
            n_missing += 1
            continue
        # Mirror the response envelope evaluate.py hands the Evaluator, and keep
        # the SUT-side stats already recorded in the previous evaluation.json so
        # the file stays self-contained.
        prev_path = os.path.join(ROOT, "system_scratch", sut, tid, "evaluation.json")
        prev = {}
        if os.path.exists(prev_path):
            try:
                with open(prev_path) as f:
                    prev = json.load(f)
            except Exception:
                pass
        response = {"task_id": tid, "model_output": ans, "code": ""}
        for k in ("token_usage_sut", "token_usage_sut_input", "token_usage_sut_output", "runtime"):
            if k in prev:
                response[k] = prev[k]

        result = ev._evaluate_result_for_task(response, task, evaluate_pipeline=False)[0]
        n_scored += 1

        with open(prev_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        for metric, value in result.items():
            if metric in ("task_id", "code", "model_output", "subresponses"):
                continue
            rows.append({"sut": sut, "workload": workload, "task_id": tid,
                         "metric": metric, "value": value})

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", sut)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{workload}_measures_{ts}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    if verbose:
        print(f"  {sut}/{workload}: scored {n_scored}, missing answer {n_missing} -> {os.path.relpath(out, ROOT)}")
    return n_scored, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", nargs="+", required=True)
    ap.add_argument("--workload", nargs="+", default=WORKLOADS)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    for sut in a.sut:
        if not os.path.isdir(os.path.join(ROOT, "system_scratch", sut)):
            print(f"SKIP {sut}: no scratch dir")
            continue
        tot = miss = 0
        for wl in a.workload:
            s, m = rescore(sut, wl, verbose=not a.quiet)
            tot += s
            miss += m
        print(f"{sut}: {tot} scored, {miss} missing")


if __name__ == "__main__":
    main()
