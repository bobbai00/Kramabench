#!/usr/bin/env python3
"""Best-of-2 analysis for the Haiku-4.5 2x2 study on the legal workload.

Reads ALL results/<SUT>/legal_measures_*.csv for each SUT (run1 + any reruns),
and per task keeps the BEST run (max primary score; tie -> fewer tokens). Then
reports correctness vs token cost across the 2x2.

Subcommands:
  failures <SUT>   print space-separated task ids whose run1 (latest CSV) primary
                   score < 1.0 (or missing) -> the best-of-2 rerun set.
  report           print the 2x2 correctness-vs-cost table (best-of-2 + run1-only).
"""
import sys, glob, json, os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PRIMARY_METRIC = {
    "numeric_exact": "success", "string_exact": "success",
    "numeric_approximate": "rae_score", "string_approximate": "llm_paraphrase",
    "list_exact": "f1", "list_approximate": "f1_approximate",
}
SUTS = [
    ("DataflowSystemHaiku45Annot2",                    False, False),
    ("DataflowSystemHaiku45Annot2Lineage",             True,  False),
    ("DataflowSystemHaiku45Annot2ThoughtReplay",       False, True),
    ("DataflowSystemHaiku45Annot2LineageThoughtReplay",True,  True),
]
WORKLOAD = "legal"

def answer_types():
    d = json.load(open(os.path.join(ROOT, "workload", f"{WORKLOAD}.json")))
    return {t["id"]: t.get("answer_type") for t in d}, [t["id"] for t in d]

def per_run_records(csv_path, atypes):
    """task_id -> {score, tokens, tin, tout} for one measures CSV."""
    df = pd.read_csv(csv_path)
    df = df[df["workload"] == WORKLOAD]
    out = {}
    for tid, g in df.groupby("task_id"):
        if tid not in atypes:  # skip subtask / unknown rows
            continue
        m = PRIMARY_METRIC.get(atypes[tid], "success")
        vals = {row["metric"]: row["value"] for _, row in g.iterrows()}
        score = vals.get(m)
        if score is None:
            continue
        out[tid] = {
            "score": float(score),
            "tokens": float(vals.get("token_usage_sut", 0) or 0),
            "tin": float(vals.get("token_usage_sut_input", 0) or 0),
            "tout": float(vals.get("token_usage_sut_output", 0) or 0),
        }
    return out

def latest_csv(sut):
    files = sorted(glob.glob(os.path.join(ROOT, "results", sut, f"{WORKLOAD}_measures_*.csv")))
    return files[-1] if files else None

def all_runs(sut, atypes):
    return [per_run_records(f, atypes) for f in
            sorted(glob.glob(os.path.join(ROOT, "results", sut, f"{WORKLOAD}_measures_*.csv")))]

def best_of(runs, tid):
    recs = [r[tid] for r in runs if tid in r]
    if not recs:
        return None
    return max(recs, key=lambda r: (r["score"], -r["tokens"]))

def cmd_failures(sut):
    atypes, ids = answer_types()
    csv = latest_csv(sut)
    if not csv:
        print(""); return
    recs = per_run_records(csv, atypes)
    failed = [tid for tid in ids if recs.get(tid, {}).get("score", 0.0) < 1.0]
    print(" ".join(failed))

def cmd_report():
    atypes, ids = answer_types()
    print(f"\n{'='*112}\nHaiku-4.5 2x2 — legal workload ({len(ids)} tasks). Correctness vs token cost.\n{'='*112}")
    hdr = f"{'SUT':<46} {'lineage':<8} {'replay':<7} {'pass(best/30)':<14} {'meanScore':<10} {'totTok':<10} {'meanTok':<9} {'in/out':<14}"
    def block(title, pick):
        print(f"\n-- {title} --\n{hdr}")
        for sut, lineage, replay in SUTS:
            runs = all_runs(sut, atypes)
            if not runs:
                print(f"{sut:<46} {'on' if lineage else 'off':<8} {'on' if replay else 'off':<7} (no results yet)")
                continue
            recs = [pick(runs, tid) for tid in ids]
            recs = [r for r in recs if r]
            n = len(recs)
            passed = sum(1 for r in recs if r["score"] >= 1.0)
            mean = sum(r["score"] for r in recs)/n if n else 0
            tot = sum(r["tokens"] for r in recs)
            mtok = tot/n if n else 0
            tin = sum(r["tin"] for r in recs); tout = sum(r["tout"] for r in recs)
            print(f"{sut:<46} {'on' if lineage else 'off':<8} {'on' if replay else 'off':<7} "
                  f"{str(passed)+'/'+str(n):<14} {mean*100:>7.1f}%   {tot:>9.0f} {mtok:>8.0f}  {tin:>7.0f}/{tout:<7.0f}")
    block("RUN 1 ONLY (single run)", lambda runs, tid: runs[0].get(tid) if runs else None)
    block("BEST OF 2 (max score per task across runs)", best_of)
    # 2x2 grid (best-of-2 pass rate)
    print(f"\n-- 2x2 grid: pass rate (best-of-2) --")
    grid = {}
    for sut, lineage, replay in SUTS:
        runs = all_runs(sut, atypes)
        recs = [best_of(runs, tid) for tid in ids] if runs else []
        recs = [r for r in recs if r]
        grid[(lineage, replay)] = (sum(1 for r in recs if r["score"]>=1.0), len(recs))
    print(f"{'':<16}{'replay=off':<14}{'replay=on':<14}")
    for lineage in (False, True):
        row = f"{'lineage='+('on' if lineage else 'off'):<16}"
        for replay in (False, True):
            p,n = grid.get((lineage,replay),(0,0))
            row += f"{str(p)+'/'+str(n):<14}"
        print(row)

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "failures":
        cmd_failures(sys.argv[2])
    else:
        cmd_report()
