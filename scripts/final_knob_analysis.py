#!/usr/bin/env python3
"""Rigorous C1/C2 knob analysis on the subtask eval (+ main accuracy), MATCHED:
each pair's delta is computed only over tasks BOTH arms completed (removes the
transient-failure selection bias). Reads KramaBench's own scores."""
import json, glob, argparse, statistics as st
from pathlib import Path
KB = Path(__file__).resolve().parent.parent
SK = ["f1", "success", "rae_score", "llm_paraphrase", "f1_approximate"]

def load_W():
    W = {}
    for f in glob.glob(str(KB / "workload/*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")): continue
        for t in (json.load(open(f)) if True else []):
            if isinstance(t, dict) and t.get("id"): W[t["id"]] = t
    return W

def sc(p):
    if not Path(p).exists(): return None
    try: d = json.load(open(p))
    except: return None
    for k in SK:
        if k in d and isinstance(d[k], (int, float)): return d[k]
    return None

def arm(sut, tasks, W):
    base = KB / "system_scratch" / sut
    pt = {}  # task -> (main_score_or_None, submean_or_None, n_sub)
    for t in tasks:
        m = sc(base / t / "evaluation.json")
        subs = [sc(base / x["id"] / "evaluation.json") for x in W[t].get("subtasks", [])]
        subs = [v for v in subs if v is not None]
        pt[t] = (m, st.mean(subs) if subs else None, len(subs))
    return pt

def completed(pt, t):  # task "completed" if it has subtask scores
    return pt[t][1] is not None

def report_pair(name, A, B, la, lb, tasks):
    matched = [t for t in tasks if completed(A, t) and completed(B, t)]
    sa = [A[t][1] for t in matched]; sb = [B[t][1] for t in matched]
    # main pass on matched (only where both have a main score)
    mm = [t for t in matched if A[t][0] is not None and B[t][0] is not None]
    ma = sum(1 for t in mm if A[t][0] >= 0.9) / max(1, len(mm))
    mb = sum(1 for t in mm if B[t][0] >= 0.9) / max(1, len(mm))
    diffs = [B[t][1] - A[t][1] for t in matched]
    up = sum(1 for d in diffs if d > 0.05); dn = sum(1 for d in diffs if d < -0.05)
    print(f"\n=== {name}: {la} -> {lb} (matched {len(matched)} tasks both completed) ===")
    print(f"  subtask-mean: {la} {st.mean(sa):.3f}  {lb} {st.mean(sb):.3f}   Δ={st.mean(sb)-st.mean(sa):+.3f}")
    print(f"  main-pass (n={len(mm)}): {la} {ma:.1%}  {lb} {mb:.1%}   Δ={mb-ma:+.1%}")
    print(f"  per-task subtask movement (|Δ|>0.05): {up} up / {dn} down / {len(matched)-up-dn} flat")
    print(f"  mean per-task |Δ|={st.mean([abs(d) for d in diffs]):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="DataflowSystemGPT5MiniDelta1kSchemaOnly")
    ap.add_argument("--c1", default="DataflowSystemGPT5MiniDelta5kSchemaOnly")
    ap.add_argument("--c2", default="DataflowSystemGPT5MiniDeltaStats1kD2")
    ap.add_argument("--tasks-file", default=str(KB / "judgment_runs/levers_report/tasks50.txt"))
    a = ap.parse_args()
    W = load_W()
    tasks = [t for t in open(a.tasks_file).read().split() if t in W]
    AN = arm(a.anchor, tasks, W); C1 = arm(a.c1, tasks, W); C2 = arm(a.c2, tasks, W)
    # completion per arm
    for nm, pt in [("anchor(1k)", AN), ("C1 ray(5k)", C1), ("C2 ray(stats)", C2)]:
        print(f"{nm}: completed {sum(1 for t in tasks if completed(pt,t))}/{len(tasks)} tasks")
    report_pair("C1 (rows knob)", AN, C1, "1k", "5k", tasks)
    report_pair("C2 (stats knob)", AN, C2, "schema", "stats", tasks)

if __name__ == "__main__":
    main()
