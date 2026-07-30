#!/usr/bin/env python3
"""Easy/hard split using KramaBench's OWN scores: reads the freshest native
measures CSVs (results/<SUT>/<wl>_measures_*.csv, written by evaluate.py's
metric pass) and applies compute_scores.py's exact formula
(sum(value)/n over SCORE_METRICS rows), partitioned by -easy-/-hard- task ids.
Validation: easy+hard recombined must equal compute_scores.py OVERALL.
"""
import glob, os, sys
import pandas as pd
KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
SCORE_METRICS = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]  # compute_scores.py:21
WLS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]

WL_N = {"archeology": 12, "astronomy": 12, "biomedical": 9, "environment": 20,
        "legal": 30, "wildfire": 21}  # full task counts per workload

def native_rows(sut):
    """Freshest FULL-workload measures CSV per workload (skips partial CSVs
    written by --task_id runs/warm-ups, which contain only a task subset)."""
    dfs = []
    for wl in WLS:
        for f in sorted(glob.glob(f"{KB}/results/{sut}/{wl}_measures_*.csv"), reverse=True):
            d = pd.read_csv(f)
            if d[d["metric"].isin(SCORE_METRICS)]["task_id"].nunique() >= WL_N[wl]:
                dfs.append(d)
                break
    d = pd.concat(dfs, ignore_index=True)
    return d[d["metric"].isin(SCORE_METRICS)]

def split(sut):
    d = native_rows(sut)
    # one score-metric row per task (native long format); official formula:
    # sum(value)/total_support, support=1 per task row
    easy = d[d["task_id"].str.contains("-easy-")]
    hard = d[d["task_id"].str.contains("-hard-")]
    e = easy["value"].sum() / len(easy) * 100
    h = hard["value"].sum() / len(hard) * 100
    o = d["value"].sum() / len(d) * 100
    return e, len(easy), h, len(hard), o, len(d)

if __name__ == "__main__":
    for label, sut in [l.split(None, 1) for l in sys.stdin.read().strip().split("\n")]:
        e, ne, h, nh, o, n = split(sut)
        print(f"{label:<14} easy {e:5.1f}% (n={ne})  hard {h:5.1f}% (n={nh})  OVERALL {o:5.1f}% (n={n})")
