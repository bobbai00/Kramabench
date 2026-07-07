#!/usr/bin/env python3
"""Best-of-2 analysis for arbitrary SUTs x workloads.

  analyze_exp.py failures <SUT> <WORKLOAD>   -> space-separated failed task ids (run1)
  analyze_exp.py report "<SUT1> <SUT2>" "<WL1> <WL2>"  -> correctness vs token cost table

Best-of-2: per task, the run (across all that SUT/workload's measures CSVs) with the
max primary score (tie -> fewer tokens).
"""
import sys, glob, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PRIMARY = {
    "numeric_exact": "success", "string_exact": "success",
    "numeric_approximate": "rae_score", "string_approximate": "llm_paraphrase",
    "list_exact": "f1", "list_approximate": "f1_approximate",
}

def atypes(wl):
    d = json.load(open(os.path.join(ROOT, "workload", f"{wl}.json")))
    return {t["id"]: t.get("answer_type") for t in d}, [t["id"] for t in d]

def per_run(csv, wl, at):
    df = pd.read_csv(csv)
    df = df[df["workload"] == wl]
    out = {}
    for tid, g in df.groupby("task_id"):
        if tid not in at:
            continue
        m = PRIMARY.get(at[tid], "success")
        v = {r["metric"]: r["value"] for _, r in g.iterrows()}
        s = v.get(m)
        if s is None:
            continue
        out[tid] = {"s": float(s), "tok": float(v.get("token_usage_sut", 0) or 0),
                    "tin": float(v.get("token_usage_sut_input", 0) or 0),
                    "tout": float(v.get("token_usage_sut_output", 0) or 0)}
    return out

def csvs(sut, wl):
    return sorted(glob.glob(os.path.join(ROOT, "results", sut, f"{wl}_measures_*.csv")))

def latest(sut, wl):
    c = csvs(sut, wl)
    return c[-1] if c else None

def best_of(sut, wl, at, ids):
    runs = [per_run(c, wl, at) for c in csvs(sut, wl)]
    out = {}
    for tid in ids:
        recs = [r[tid] for r in runs if tid in r]
        out[tid] = max(recs, key=lambda r: (r["s"], -r["tok"])) if recs else None
    return out

def cmd_failures(sut, wl):
    # Union across ALL of this SUT/workload's measures CSVs (handles env runs
    # split into several partial CSVs), best-of per task; a task fails if its
    # best score < 1.0 (or it never appeared).
    at, ids = atypes(wl)
    bo = best_of(sut, wl, at, ids)
    print(" ".join(tid for tid in ids if (bo.get(tid) or {}).get("s", 0.0) < 1.0))

def cmd_report(suts, wls):
    for wl in wls:
        at, ids = atypes(wl)
        print(f"\n{'='*100}\nWorkload: {wl} ({len(ids)} tasks) — best-of-2 correctness vs token cost\n{'='*100}")
        print(f"{'SUT':<46}{'pass/N':<10}{'meanScore':<11}{'totTok':<11}{'meanTok':<10}{'tok/correct':<12}")
        for sut in suts:
            bo = best_of(sut, wl, at, ids)
            recs = [v for v in bo.values() if v]
            n = len(recs)
            if n == 0:
                print(f"{sut:<46}(no results)"); continue
            p = sum(1 for r in recs if r["s"] >= 1.0)
            mean = sum(r["s"] for r in recs) / n
            tot = sum(r["tok"] for r in recs)
            tpc = tot / p if p else float("nan")
            print(f"{sut:<46}{str(p)+'/'+str(n):<10}{mean*100:>7.1f}%   {tot:>10.0f} {tot/n:>9.0f} {tpc:>11.0f}")

if __name__ == "__main__":
    if sys.argv[1] == "failures":
        cmd_failures(sys.argv[2], sys.argv[3])
    else:
        suts = sys.argv[2].split()
        wls = sys.argv[3].split()
        cmd_report(suts, wls)
