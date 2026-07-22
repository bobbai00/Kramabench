#!/usr/bin/env python3
"""Join the M3/M4 judge metrics with the answer metrics (M1 main, M2 subtask)
for a C1-style A-vs-B pair: per-arm means, knob deltas, movement buckets, and
the failure-mode split. Reads the judge cache written by scripts/judge_metrics.py.

Run: .venv/bin/python scripts/judge_vs_answers.py [--a SUT --b SUT --tasks-file F]
"""
import argparse, json, glob, statistics as st
from pathlib import Path
KB = Path(__file__).resolve().parent.parent
SK = ["f1", "success", "rae_score", "llm_paraphrase", "f1_approximate"]

def load_W():
    W = {}
    for f in glob.glob(str(KB / "workload/*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")): continue
        for t in json.load(open(f)):
            if isinstance(t, dict) and t.get("id"): W[t["id"]] = t
    return W

def jload(p):
    p = Path(p)
    try: return json.load(open(p)) if p.exists() else None
    except Exception: return None

def sc(p):
    d = jload(p)
    if not d: return None
    for k in SK:
        if isinstance(d.get(k), (int, float)): return float(d[k])
    return None

def submean(arm, t, W):
    vs = [sc(KB / "system_scratch" / arm / x["id"] / "evaluation.json") for x in W[t].get("subtasks", [])]
    vs = [v for v in vs if v is not None]
    return st.mean(vs) if vs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="DataflowSystemGPT5MiniDelta1kSchemaOnly")
    ap.add_argument("--b", default="DataflowSystemGPT5MiniDelta5kSchemaOnly")
    # default = the 29 matched trace-clean tasks (mini C1 anchor traces for the 12
    # variance-subset tasks are from a different run than their scored answers).
    ap.add_argument("--tasks-file", default=str(KB / "judgment_runs/levers_report/tasks_judge29.txt"))
    g = ap.parse_args()
    W = load_W()
    tasks = [t for t in open(g.tasks_file).read().split() if t in W]
    rows = []
    for t in tasks:
        ja = jload(KB / "system_scratch" / g.a / t / "judge_m3m4.json")
        jb = jload(KB / "system_scratch" / g.b / t / "judge_m3m4.json")
        sa, sb = submean(g.a, t, W), submean(g.b, t, W)
        ma, mb = sc(KB / "system_scratch" / g.a / t / "evaluation.json"), sc(KB / "system_scratch" / g.b / t / "evaluation.json")
        if not ja or not jb or sa is None or sb is None: continue
        rows.append(dict(t=t, ja=ja, jb=jb, dsub=sb - sa, ma=ma, mb=mb))
    print(f"matched tasks with judge cache: {len(rows)}")
    # 1. knob deltas on each metric
    print(f"\n== C1 knob delta (1k -> 5k) on every metric ==")
    for key, nm in [("m3", "M3 evidence-in-context"), ("m4_process", "M4 step-performed (process)"),
                    ("m4_deliverable", "M4 step-performed (deliverable)")]:
        d = [r["jb"][key] - r["ja"][key] for r in rows]
        up = sum(1 for x in d if x > 0.05); dn = sum(1 for x in d if x < -0.05)
        print(f"  {nm:32s}: {st.mean([r['ja'][key] for r in rows]):.3f} -> {st.mean([r['jb'][key] for r in rows]):.3f}"
              f"   Δ={st.mean(d):+.3f}   ({up}up/{dn}dn/{len(d)-up-dn}flat)")
    d = [r["dsub"] for r in rows]
    print(f"  {'M2 subtask answer score':32s}: Δ={st.mean(d):+.3f}   "
          f"({sum(1 for x in d if x>0.05)}up/{sum(1 for x in d if x<-0.05)}dn/{len(d)-sum(1 for x in d if abs(x)>0.05)}flat)")
    # 2. mediation: does M3 gain concentrate where answers improved?
    print(f"\n== M3 delta by answer movement bucket ==")
    for nm, sel in [("answer UP  (Δsub>+0.05)", lambda r: r["dsub"] > 0.05),
                    ("answer flat(|Δ|<=0.05)", lambda r: abs(r["dsub"]) <= 0.05),
                    ("answer DOWN(Δsub<-0.05)", lambda r: r["dsub"] < -0.05)]:
        b = [r for r in rows if sel(r)]
        if not b: print(f"  {nm:26s} n=0"); continue
        dm = [r["jb"]["m3"] - r["ja"]["m3"] for r in b]
        print(f"  {nm:26s} n={len(b):2d}  mean M3Δ={st.mean(dm):+.3f}  "
              f"({sum(1 for x in dm if x>0.05)}up/{sum(1 for x in dm if x<-0.05)}dn)")
    # 3. failure modes per arm
    print(f"\n== Failure modes (failed tasks, answer<0.9) ==")
    for arm, jk, mk in [(g.a, "ja", "ma"), (g.b, "jb", "mb")]:
        modes = {}
        for r in rows:
            if r[mk] is None or r[mk] >= 0.9: continue
            j = r[jk]
            m = ("mode1-step-missing" if j["m4_process"] < 0.999 else
                 "mode2-value-absent" if j["m3"] < 0.999 else "mode3-had-all-still-failed")
            modes.setdefault(m, []).append(r["t"])
        tot = sum(len(v) for v in modes.values()) or 1
        print(f"  {arm}:")
        for m in sorted(modes):
            print(f"    {m:28s} {len(modes[m]):2d} ({len(modes[m])/tot:.0%})  {modes[m][:5]}")
    # 4. biggest per-task M3 movers
    print(f"\n== biggest M3 movers (with answer movement) ==")
    print(f"{'task':22s} {'M3 1k':>6s} {'M3 5k':>6s} {'M3Δ':>7s} {'Δsub':>7s}")
    for r in sorted(rows, key=lambda r: -abs(r["jb"]["m3"] - r["ja"]["m3"]))[:8]:
        print(f"{r['t']:22s} {r['ja']['m3']:6.2f} {r['jb']['m3']:6.2f} "
              f"{r['jb']['m3']-r['ja']['m3']:+7.2f} {r['dsub']:+7.2f}")

if __name__ == "__main__":
    main()
