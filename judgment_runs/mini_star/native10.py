#!/usr/bin/env python3
"""KramaBench-native score restricted to the focused-10, using the freshest raw
measures files (results/<SUT>/<wl>_measures_*.csv) and compute_scores.py's exact
formula: score = sum(support*mean)/total_support*100, one SCORE_METRIC row per task."""
import os, glob, sys
import pandas as pd
KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
D = os.path.dirname(__file__)
SCORE_METRICS = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]
ARMS = {"anchor": "DataflowSystemGPT5MiniDelta1kSchemaOnly",
        "C1_5k": "DataflowSystemGPT5MiniDelta5kSchemaOnly",
        "C2_stats": "DataflowSystemGPT5MiniDeltaStats1kD2",
        "C3_latest_code": "DataflowSystemGPT5MiniLatest1kCodeInSnap"}
TASKS = [l.strip() for l in open(os.path.join(D, "tasks10.txt")) if l.strip()]
WLS = sorted(set(t.rsplit("-", 2)[0] for t in TASKS))

def freshest_measures(sut):
    """Newest measures csv per workload for this SUT."""
    rows = []
    for wl in WLS:
        fs = sorted(glob.glob(os.path.join(KB, "results", sut, f"{wl}_measures_*.csv")))
        if not fs:
            continue
        rows.append(pd.read_csv(fs[-1]))  # newest by timestamp in name
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

print(f"{'arm':<16}{'score%':>8}{'correct':>9}{'total':>6}  per-task")
summary = {}
for k, sut in ARMS.items():
    df = freshest_measures(sut)
    df = df[df["task_id"].isin(TASKS) & df["metric"].isin(SCORE_METRICS)]
    # one score-metric row per task; support = 1 each
    per = df.groupby("task_id")["value"].max()  # a task has exactly 1 score metric
    per = per.reindex(TASKS)  # missing task -> NaN
    correct = per.fillna(0).sum()
    total = len(TASKS)
    pct = correct / total * 100
    summary[k] = dict(score=pct, correct=correct, total=total,
                      per={t: (None if pd.isna(per[t]) else round(float(per[t]), 2)) for t in TASKS})
    ptxt = " ".join(f"{t.split('-',1)[0][:3]}{'.'.join(t.split('-')[1:])}={'NA' if pd.isna(per[t]) else format(per[t],'.2f')}" for t in TASKS)
    print(f"{k:<16}{pct:>7.1f}%{correct:>9.2f}{total:>6}")
print("\n=== per-task native score matrix (this reeval) ===")
print(f"{'task':<22}" + "".join(f"{k:>16}" for k in ARMS))
for t in TASKS:
    print(f"{t:<22}" + "".join(f"{(('%.2f'%summary[k]['per'][t]) if summary[k]['per'][t] is not None else 'NA'):>16}" for k in ARMS))
import json
json.dump(summary, open(os.path.join(D, "native10_summary.json"), "w"), indent=1)
