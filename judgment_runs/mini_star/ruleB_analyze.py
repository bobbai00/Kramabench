#!/usr/bin/env python3
"""Rule B decisive table: accuracy (subset-scoped) + cache + reasoning, matched
against the C8 Latest5k+code baseline on the SAME 20 hard tasks."""
import json, glob, os, statistics as st

D = os.path.dirname(__file__)
TASKS = open(os.path.join(D, "subset_hard.txt")).read().split()
ATM = {"numeric_exact": "success", "string_exact": "success", "list_exact": "f1",
       "numeric_approximate": "rae_score", "list_approximate": "f1_approximate",
       "string_approximate": "llm_paraphrase"}
FALLBACK = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]
PIN, PCACHE, POUT = 0.25e-6, 0.025e-6, 2e-6

def jload(p):
    try: return json.load(open(p))
    except Exception: return None

def arm_rep_score(sut):
    """Mean answer-type score over the 20 subset tasks for one arm-rep."""
    vals = []
    for t in TASKS:
        d = f"system_scratch/{sut}/{t}"
        ev, gt = jload(f"{d}/evaluation.json"), jload(f"{d}/ground_truth.json")
        if not ev: continue
        k = ATM.get((gt or {}).get("answer_type") or "")
        v = ev.get(k) if k and isinstance(ev.get(k), (int, float)) else None
        if v is None:
            cands = [float(ev[m]) for m in FALLBACK if isinstance(ev.get(m), (int, float))]
            v = max(cands) if cands else None
        if v is not None: vals.append(float(v))
    return (st.mean(vals) * 100 if vals else None), len(vals)

def arm_rep_stats(sut):
    tin = cach = tout = reas = steps = 0.0; n = 0
    for t in TASKS:
        s = jload(f"system_scratch/{sut}/{t}/stats.json")
        if not s: continue
        tin += s.get("input_tokens", 0); cach += s.get("cached_tokens", 0)
        tout += s.get("output_tokens", 0); reas += s.get("reasoning_tokens", 0)
        steps += s.get("num_steps", 0); n += 1
    if not n: return None
    return dict(n=n, tin=tin/n, cached=cach/n, tout=tout/n, reas=reas/n, steps=steps/n,
                cachepct=cach/tin*100 if tin else 0,
                cost=((tin-cach)*PIN + cach*PCACHE + tout*POUT)/n)

ARMS = {
    "baseline C8 (latest5k+code)": [f"DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate{r}" for r in range(5)],
    "B1 + code history":          [f"DataflowSystemGPT5MiniB1CodeHistReplicate{r}" for r in (1, 2, 3)],
    "B2 + data history":          [f"DataflowSystemGPT5MiniB2ResultHistReplicate{r}" for r in (1, 2, 3)],
    "B3 + thought replay":        [f"DataflowSystemGPT5MiniB3ReplayReplicate{r}" for r in (1, 2, 3)],
}

print(f"20 hard tasks; accuracy = KramaBench answer-type metric, subset-scoped\n")
print(f"{'arm':<28}{'acc mean±std':>16}{'cache%':>9}{'reason/task':>13}{'uncached-in':>12}{'$/task':>9}{'steps':>7}")
print("-" * 95)
rows = {}
for label, suts in ARMS.items():
    accs, sts = [], []
    for s in suts:
        a, n = arm_rep_score(s)
        if a is not None: accs.append(a)
        x = arm_rep_stats(s)
        if x: sts.append(x)
    if not accs: print(f"{label:<28} no data"); continue
    m = st.mean(accs); sd = st.pstdev(accs) if len(accs) > 1 else 0
    cache = st.mean([x["cachepct"] for x in sts]); reas = st.mean([x["reas"] for x in sts])
    unc = st.mean([x["tin"] - x["cached"] for x in sts]); cost = st.mean([x["cost"] for x in sts])
    steps = st.mean([x["steps"] for x in sts])
    rows[label] = dict(acc=m, sd=sd, cache=cache, reas=reas, unc=unc, cost=cost, steps=steps, reps=len(accs))
    print(f"{label:<28}{m:>10.1f} ±{sd:4.1f}{cache:>9.1f}{reas:>13,.0f}{unc:>12,.0f}{cost:>9.4f}{steps:>7.1f}")

b = rows.get("baseline C8 (latest5k+code)")
if b:
    print(f"\n=== deltas vs baseline ===")
    for k, v in rows.items():
        if k == "baseline C8 (latest5k+code)": continue
        print(f"  {k:<24} acc {v['acc']-b['acc']:+5.1f}  cache {v['cache']-b['cache']:+6.1f}pp  "
              f"reasoning {v['reas']-b['reas']:+7,.0f}  uncached-in {v['unc']-b['unc']:+7,.0f}  "
              f"cost {(v['cost']/b['cost']-1)*100:+5.1f}%")
