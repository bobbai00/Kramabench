#!/usr/bin/env python3
"""Mediation test at mini: does the C1 (rows) knob's answer improvement travel
THROUGH evidence delivery? For each matched task, compute M4 evidence-delivery
(gold subtask values found in the rendered context) in the 1k vs 5k arm, then
cross-tab the per-task M4 delta against the subtask-score movement.

If the knob->answer effect is mediated by delivery, M4Δ should concentrate in
the answer-improved bucket and be ~0 in the flat bucket.

Uses SOURCE_ONLY renders (loader ops) by default to avoid circularity (a correct
arm's own computed answer counting as "delivered"). --all-renders includes the
agent's derived outputs (the "elicitation" view).

Run: .venv/bin/python scripts/m4_mediation.py [--all-renders]
"""
import argparse, json, glob, importlib.util, statistics as st
from pathlib import Path
KB = Path(__file__).resolve().parent.parent
LR = KB / "judgment_runs/levers_report"

def load(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m

m4 = load("m4", KB / "scripts/m4_evidence_delivery.py")
SK = ["f1", "success", "rae_score", "llm_paraphrase", "f1_approximate"]
A = "DataflowSystemGPT5MiniDelta1kSchemaOnly"
B = "DataflowSystemGPT5MiniDelta5kSchemaOnly"

def sc(p):
    p = Path(p)
    if not p.exists(): return None
    try: d = json.load(open(p))
    except: return None
    for k in SK:
        if k in d and isinstance(d[k], (int, float)): return d[k]
    return None

def submean(sut, t, W):
    vs = [sc(KB / "system_scratch" / sut / x["id"] / "evaluation.json") for x in W[t].get("subtasks", [])]
    vs = [v for v in vs if v is not None]
    return st.mean(vs) if vs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-renders", action="store_true")
    args = ap.parse_args()
    m4.SOURCE_ONLY = not args.all_renders
    W = m4.m1.load_workload()
    contam = set((LR / "variance_subset12.txt").read_text().split())  # anchor traces overwritten by variance run
    tasks = [t for t in (LR / "tasks50.txt").read_text().split() if t in W and t not in contam]
    rows = []
    for t in tasks:
        sa, sb = submean(A, t, W), submean(B, t, W)
        if sa is None or sb is None: continue           # matched only
        ra, rb = m4.score_task(A, t, W[t]), m4.score_task(B, t, W[t])
        if not ra or not rb or ra["coverage"] is None or rb["coverage"] is None: continue
        rows.append(dict(t=t, m4a=ra["coverage"], m4b=rb["coverage"], da=sb - sa, dm=rb["coverage"] - ra["coverage"]))
    mode = "ALL-RENDERS (elicitation)" if args.all_renders else "SOURCE-ONLY (raw loader renders)"
    print(f"=== M4 mediation, {mode}; matched, trace-clean n={len(rows)} ===")
    print(f"arm M4 means: 1k {st.mean([r['m4a'] for r in rows]):.3f} -> 5k {st.mean([r['m4b'] for r in rows]):.3f}  (Δ={st.mean([r['dm'] for r in rows]):+.3f})")
    buckets = [("answer UP  (Δsub>+0.05)", [r for r in rows if r["da"] > 0.05]),
               ("answer flat(|Δ|<=0.05)", [r for r in rows if abs(r["da"]) <= 0.05]),
               ("answer DOWN(Δsub<-0.05)", [r for r in rows if r["da"] < -0.05])]
    print(f"\n{'bucket':26s} {'n':>3s} {'mean M4Δ':>9s} {'M4Δ>+0.02':>10s} {'M4Δ<-0.02':>10s}")
    for name, b in buckets:
        if not b: print(f"{name:26s} {0:3d}"); continue
        up = sum(1 for r in b if r["dm"] > 0.02); dn = sum(1 for r in b if r["dm"] < -0.02)
        print(f"{name:26s} {len(b):3d} {st.mean([r['dm'] for r in b]):+9.3f} {up:10d} {dn:10d}")
    # the flipped tasks, individually
    print(f"\n--- answer-UP tasks (did delivery rise with the flip?) ---")
    print(f"{'task':22s} {'Δsub':>7s} {'M4 1k':>6s} {'M4 5k':>6s} {'M4Δ':>7s}")
    for r in sorted([r for r in rows if r["da"] > 0.05], key=lambda r: -r["da"]):
        print(f"{r['t']:22s} {r['da']:+7.3f} {r['m4a']:6.2f} {r['m4b']:6.2f} {r['dm']:+7.3f}")

if __name__ == "__main__":
    main()
