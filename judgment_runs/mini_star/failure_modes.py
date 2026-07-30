#!/usr/bin/env python3
"""What KIND of failure dominates? The prerequisite for any context lever.

If failures are mostly "produced an answer, wrong value", then no amount of render
tuning fixes them — the bottleneck is reasoning or the task itself. If they are
"thrashed on errors" or "ran out of steps", the render plausibly matters.

Method: pool every scored era-2 arm-rep. For each (task, arm-rep) classify the run
from its trace, then group by task pass-rate so flippy tasks are separated from
never-pass ones. Accuracy always comes from KramaBench's own measures CSVs.
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

import pandas as pd

KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
SM = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]
WL_N = {"archeology": 12, "astronomy": 12, "biomedical": 9,
        "environment": 20, "legal": 30, "wildfire": 21}
STEP_CAP = 25


def scored_arms():
    """Era-2 arm-reps with full CSVs (the comparable, current-vintage set)."""
    pats = ["D8Latest5kCode", "D8FileIO", "D12LatestStats1kCode", "D12FileIO",
            "N1Latest5kStats", "N2Delta5kStats", "N3SrcRich5k2k",
            "N4Latest2kStats", "N5SrcRich2k1k", "N6Latest3kStats"]
    out = []
    for p in pats:
        for r in range(1, 6):
            sut = f"DataflowSystemGPT5Mini{p}Replicate{r}"
            if os.path.isdir(f"{KB}/system_scratch/{sut}"):
                out.append(sut)
    return out


def task_scores(sut):
    dfs = []
    for wl, n in WL_N.items():
        for f in sorted(glob.glob(f"{KB}/results/{sut}/{wl}_measures_*.csv"), reverse=True):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if d[d["metric"].isin(SM)]["task_id"].nunique() >= n:
                dfs.append(d)
                break
    if len(dfs) < len(WL_N):
        return None
    d = pd.concat(dfs, ignore_index=True)
    d = d[d["metric"].isin(SM)]
    return d.groupby("task_id")["value"].mean().to_dict()


def classify(tdir):
    """One run -> (mode, n_steps, n_errors, n_ops)."""
    resp_p = os.path.join(tdir, "response.txt")
    stats_p = os.path.join(tdir, "stats.json")
    trace_p = os.path.join(tdir, "react_steps.json")
    resp = ""
    if os.path.exists(resp_p):
        try:
            resp = open(resp_p, errors="ignore").read().strip()
        except Exception:
            pass
    steps = 0
    if os.path.exists(stats_p):
        try:
            steps = json.load(open(stats_p)).get("num_steps", 0) or 0
        except Exception:
            pass

    n_err, n_ops = 0, 0
    try:
        d = json.load(open(trace_p))
        ops = set()
        for stp in d.get("steps", []):
            for tr in stp.get("toolResults") or []:
                if isinstance(tr, dict) and tr.get("isError"):
                    n_err += 1
                out = str(tr.get("output", "")) if isinstance(tr, dict) else ""
                if "rror" in out and "Error" in out:
                    n_err += 1
            for tc in stp.get("toolCalls") or []:
                if isinstance(tc, dict) and tc.get("toolName") == "createOrModifyOperator":
                    oid = (tc.get("input") or {}).get("operatorId")
                    if oid:
                        ops.add(oid)
        n_ops = len(ops)
    except Exception:
        pass

    if not os.path.exists(resp_p):
        mode = "no_answer_file"
    elif resp == "" or resp == "(empty response)":
        mode = "empty_response"
    elif steps >= STEP_CAP:
        mode = "hit_step_cap"
    elif n_err >= 5:
        mode = "answered_after_thrash"
    else:
        mode = "answered_clean"
    return mode, steps, n_err, n_ops


def main():
    arms = scored_arms()
    per_task_scores = defaultdict(list)
    runs = {}          # (task, sut) -> (mode, steps, err, ops)
    kept = 0
    for sut in arms:
        sc = task_scores(sut)
        if sc is None:
            continue
        kept += 1
        for task, val in sc.items():
            per_task_scores[task].append(val)
            tdir = f"{KB}/system_scratch/{sut}/{task}"
            if os.path.isdir(tdir):
                runs[(task, sut)] = classify(tdir)

    print(f"FAILURE MODES — {kept} era-2 arm-reps, {len(per_task_scores)} tasks, "
          f"{len(runs)} classified runs")
    print("Score = KramaBench's own metric mean per task.\n")

    # bucket tasks by how often they pass
    def bucket(vals):
        m = st.mean(vals)
        if m >= 0.95:
            return "always_pass"
        if m <= 0.05:
            return "never_pass"
        return "flips"

    buckets = defaultdict(list)
    for t, vals in per_task_scores.items():
        buckets[bucket(vals)].append(t)

    print(f"{'bucket':<14}{'tasks':>7}   modes over its runs")
    print("-" * 100)
    for b in ("always_pass", "flips", "never_pass"):
        tasks = set(buckets[b])
        modes = Counter(m for (t, _), (m, *_ ) in runs.items() if t in tasks)
        tot = sum(modes.values()) or 1
        desc = "  ".join(f"{k}={v/tot*100:.0f}%" for k, v in modes.most_common())
        print(f"{b:<14}{len(tasks):>7}   {desc}")

    # The decisive split: among runs that scored ~0, how many produced a clean answer?
    print("\nAmong FAILING runs (score < 0.5), what happened:")
    fail_modes = Counter()
    fail_steps, fail_errs = [], []
    for sut in arms:
        sc = task_scores(sut)
        if sc is None:
            continue
        for task, val in sc.items():
            key = (task, sut)
            if key not in runs or val >= 0.5:
                continue
            mode, steps, err, _ = runs[key]
            fail_modes[mode] += 1
            fail_steps.append(steps)
            fail_errs.append(err)
    tot = sum(fail_modes.values()) or 1
    for k, v in fail_modes.most_common():
        print(f"  {k:<24}{v:5}  {v/tot*100:5.1f}%")
    if fail_steps:
        print(f"  failing-run steps: mean {st.mean(fail_steps):.1f}   "
              f"tool errors: mean {st.mean(fail_errs):.1f}")

    print("\nAmong PASSING runs (score >= 0.5), for contrast:")
    pass_modes = Counter()
    pass_steps, pass_errs = [], []
    for sut in arms:
        sc = task_scores(sut)
        if sc is None:
            continue
        for task, val in sc.items():
            key = (task, sut)
            if key not in runs or val < 0.5:
                continue
            mode, steps, err, _ = runs[key]
            pass_modes[mode] += 1
            pass_steps.append(steps)
            pass_errs.append(err)
    tot = sum(pass_modes.values()) or 1
    for k, v in pass_modes.most_common():
        print(f"  {k:<24}{v:5}  {v/tot*100:5.1f}%")
    if pass_steps:
        print(f"  passing-run steps: mean {st.mean(pass_steps):.1f}   "
              f"tool errors: mean {st.mean(pass_errs):.1f}")

    # flippiest tasks — the only place a render change can plausibly show up
    print("\nFlippiest tasks (pass-rate nearest 50%, n>=6 reps) — where a lever could bite:")
    flippy = sorted(
        ((t, st.mean(v), len(v)) for t, v in per_task_scores.items() if len(v) >= 6),
        key=lambda x: abs(x[1] - 0.5))[:15]
    for t, m, n in flippy:
        ms = Counter(runs[(t, s)][0] for s in arms if (t, s) in runs)
        print(f"  {t:<26} pass={m*100:5.1f}%  n={n:2}  {dict(ms)}")


if __name__ == "__main__":
    main()
