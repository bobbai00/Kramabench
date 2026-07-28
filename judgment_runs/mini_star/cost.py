#!/usr/bin/env python3
"""M1 + cost summary over the focused-10, from the final (post-recovery) scratch.
Cost from stats.json:cost_usd; tokens from evaluation.json; pass from best-of-rounds."""
import json, os
D = os.path.dirname(__file__)
ROOT = os.path.expanduser("~/Desktop/bobflow/Kramabench/system_scratch")
ARMS = {"anchor": "DataflowSystemGPT5MiniDelta1kSchemaOnly",
        "C1_5k": "DataflowSystemGPT5MiniDelta5kSchemaOnly",
        "C2_stats": "DataflowSystemGPT5MiniDeltaStats1kD2",
        "C3_latest_code": "DataflowSystemGPT5MiniLatest1kCodeInSnap"}
TASKS = [l.strip() for l in open(os.path.join(D, "tasks10.txt")) if l.strip()]
best = json.load(open(os.path.join(D, "flaky_result.json")))["best"]

def load(p):
    try: return json.load(open(p))
    except Exception: return {}

print(f"{'arm':<16}{'passE':>7}{'cost$(10tk)':>13}{'tok_in':>10}{'tok_out':>9}{'steps~':>8}")
summ = {}
for k, sut in ARMS.items():
    cost = tin = tout = steps = 0.0
    for t in TASKS:
        d = os.path.join(ROOT, sut, t)
        st, ev = load(os.path.join(d, "stats.json")), load(os.path.join(d, "evaluation.json"))
        cost += float(st.get("cost_usd") or 0)
        tin += float(ev.get("token_usage_sut_input") or 0)
        tout += float(ev.get("token_usage_sut_output") or 0)
        rs = load(os.path.join(d, "react_steps.json"))
        steps += sum(1 for s in (rs.get("steps") or []) if s.get("role") == "agent")
    passe = sum(1 for t in TASKS if best[k][t] >= 0.9)
    summ[k] = dict(passE=passe, cost=cost, tin=tin, tout=tout, steps=steps)
    print(f"{k:<16}{passe:>6}P{cost:>13.4f}{int(tin):>10}{int(tout):>9}{int(steps):>8}")

anc = summ["anchor"]
print("\n=== deltas vs anchor (Delta1k) — final scratch, current cost regime ===")
for k in ("C1_5k", "C2_stats", "C3_latest_code"):
    s = summ[k]
    dc = (s["cost"]/anc["cost"]-1)*100 if anc["cost"] else 0
    print(f"  {k:<16} pass {anc['passE']}->{s['passE']}  cost {anc['cost']:.4f}->{s['cost']:.4f} ({dc:+.1f}%)  steps {int(anc['steps'])}->{int(s['steps'])}")
json.dump(summ, open(os.path.join(D, "cost_summary.json"), "w"), indent=1)
