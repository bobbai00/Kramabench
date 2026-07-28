#!/usr/bin/env python3
"""M2 subtask eval: mean answer-type subtask score per arm over the focused-10.
Subtask dirs are <task>-<n>; score via kb.answer_scores (answer-type aware)."""
import json, os, re, sys
sys.path.insert(0, os.path.expanduser("~/Desktop/bobflow/Kramabench"))
import kb
D = os.path.dirname(__file__)
ARMS = {"anchor": "DataflowSystemGPT5MiniDelta1kSchemaOnly",
        "C1_5k": "DataflowSystemGPT5MiniDelta5kSchemaOnly",
        "C2_stats": "DataflowSystemGPT5MiniDeltaStats1kD2",
        "C3_latest_code": "DataflowSystemGPT5MiniLatest1kCodeInSnap"}
TASKS = [l.strip() for l in open(os.path.join(D, "tasks10.txt")) if l.strip()]

def sub_scores(sut):
    allsc = kb.answer_scores(sut)  # includes subtask dirs
    per_task = {}
    for t in TASKS:
        pat = re.compile(rf"^{re.escape(t)}-\d+$")
        subs = {k: v for k, v in allsc.items() if pat.match(k)}
        per_task[t] = subs
    return per_task

data = {k: sub_scores(v) for k, v in ARMS.items()}
# per-task subtask mean + count, per arm
print("per-task subtask mean (n subtasks) :")
print(f"{'task':<22}" + "".join(f"{a:>18}" for a in ARMS))
for t in TASKS:
    row = ""
    for a in ARMS:
        s = data[a][t]
        m = sum(s.values())/len(s) if s else float('nan')
        row += f"{m:>13.2f}({len(s):>2})"
    print(f"{t:<22}{row}")

print("\n=== per-arm overall M2 (micro-avg over all subtasks of the 10) ===")
summ = {}
for a in ARMS:
    vals = [v for t in TASKS for v in data[a][t].values()]
    micro = sum(vals)/len(vals) if vals else 0
    # macro: mean of per-task means
    tmeans = [sum(data[a][t].values())/len(data[a][t]) for t in TASKS if data[a][t]]
    macro = sum(tmeans)/len(tmeans) if tmeans else 0
    npass = sum(1 for v in vals if v >= 0.9)
    summ[a] = dict(micro=micro, macro=macro, n=len(vals), npass=npass)
    print(f"  {a:<16} micro={micro:.3f} macro={macro:.3f} n_sub={len(vals)} pass>={0.9}:{npass}")

anc = summ["anchor"]
print("\n=== M2 deltas vs anchor ===")
for a in ("C1_5k", "C2_stats", "C3_latest_code"):
    print(f"  {a:<16} micro {anc['micro']:.3f}->{summ[a]['micro']:.3f} ({summ[a]['micro']-anc['micro']:+.3f})  macro {anc['macro']:.3f}->{summ[a]['macro']:.3f} ({summ[a]['macro']-anc['macro']:+.3f})")
json.dump(summ, open(os.path.join(D, "m2_summary.json"), "w"), indent=1)
