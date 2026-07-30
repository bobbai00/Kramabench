#!/usr/bin/env python3
"""Rule A decisive table: "rich source / lean interior" (A1RolePolicy, 3 reps)
against its own matched control (A0Control, 3 reps) on the SAME 20 hard tasks.

Both arms ran on the SAME service (:3002 @ 4af1e98da, src_dirty=False), same
model, same sampling cap — the ONLY difference is the per-operator render policy,
so the comparison is single-vintage and the manipulation is verified
(verify_rulea.py: policy sources 12 rows + stats, control uncapped + no stats).

Adds a footprint section the Rule B analysis did not need: Rule A moves bytes
BETWEEN operator roles, so the verdict depends on whether the source gain and the
interior saving land where the ablation predicted.
"""
import json, glob, os, re, statistics as st

D = os.path.dirname(__file__)
TASKS = open(os.path.join(D, "subset_hard.txt")).read().split()
ATM = {"numeric_exact": "success", "string_exact": "success", "list_exact": "f1",
       "numeric_approximate": "rae_score", "list_approximate": "f1_approximate",
       "string_approximate": "llm_paraphrase"}
FALLBACK = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]
PIN, PCACHE, POUT = 0.25e-6, 0.025e-6, 2e-6
OPB = re.compile(r"### Operator `([^`]+)` \(([^)]+)\)(.*?)(?=\n### |\Z)", re.S)


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def task_score(sut, t):
    d = f"system_scratch/{sut}/{t}"
    ev, gt = jload(f"{d}/evaluation.json"), jload(f"{d}/ground_truth.json")
    if not ev:
        return None
    k = ATM.get((gt or {}).get("answer_type") or "")
    v = ev.get(k) if k and isinstance(ev.get(k), (int, float)) else None
    if v is None:
        c = [float(ev[m]) for m in FALLBACK if isinstance(ev.get(m), (int, float))]
        v = max(c) if c else None
    return float(v) if v is not None else None


def arm_rep_score(sut):
    vals = [task_score(sut, t) for t in TASKS]
    vals = [v for v in vals if v is not None]
    return (st.mean(vals) * 100 if vals else None), len(vals)


def arm_rep_stats(sut):
    tin = cach = tout = reas = steps = 0.0
    n = 0
    for t in TASKS:
        s = jload(f"system_scratch/{sut}/{t}/stats.json")
        if not s:
            continue
        tin += s.get("input_tokens", 0); cach += s.get("cached_tokens", 0)
        tout += s.get("output_tokens", 0); reas += s.get("reasoning_tokens", 0)
        steps += s.get("num_steps", 0); n += 1
    if not n:
        return None
    return dict(n=n, tin=tin / n, cached=cach / n, tout=tout / n, reas=reas / n,
                steps=steps / n, cachepct=cach / tin * 100 if tin else 0,
                cost=((tin - cach) * PIN + cach * PCACHE + tout * POUT) / n)


def footprint(suts):
    """Did the policy actually shape the render? Per-operator-block MEANS, split
    source vs non-source, reporting the three quantities the policy sets:
    sample rows, stats presence, block chars.

    Caveat that matters: the two arms build DIFFERENT pipelines (the agent is free
    to choose operators), so counts n_src/n_int are not matched and the char means
    conflate render policy with pipeline shape. The rows/stats rates are the
    trustworthy manipulation evidence — those are set directly by the policy.
    """
    out = {}
    for bucket in ("src", "int"):
        out[bucket] = dict(chars=[], rows=[], stats=0, n=0)
    for sut in suts:
        for t in TASKS:
            doc = jload(f"system_scratch/{sut}/{t}/react_steps.json")
            if not doc:
                continue
            S = [s for s in doc.get("steps", []) if s.get("inputMessages")]
            if not S:
                continue
            ctx = "\n".join(str(m.get("content", "")) for m in S[-1]["inputMessages"])
            for m in OPB.finditer(ctx):
                body = m.group(3)
                b = "src" if "DataLoading" in m.group(2) else "int"
                out[b]["chars"].append(len(body))
                out[b]["rows"].append(len(re.findall(r"\n\s*\d+\t", body)))
                out[b]["stats"] += 1 if "Column Schema and stats" in body else 0
                out[b]["n"] += 1
    for b in out:
        d = out[b]
        d["mchars"] = st.mean(d["chars"]) if d["chars"] else 0
        d["mrows"] = st.mean(d["rows"]) if d["rows"] else 0
        d["statspct"] = d["stats"] / d["n"] * 100 if d["n"] else 0
    return out


ARMS = {
    "A1 rich-source/lean-interior": [f"DataflowSystemGPT5MiniA1RolePolicyReplicate{r}" for r in (1,2,3,4,5,6,7,8)],
    "A2 anomaly-density stats": [f"DataflowSystemGPT5MiniA2AnomalyStatsReplicate{r}" for r in (1, 2, 3)],
    "A3 no source stats": [f"DataflowSystemGPT5MiniA3NoSourceStatsReplicate{r}" for r in (1, 2, 3)],
    "A4 provenance hint": [f"DataflowSystemGPT5MiniA4SourceProvReplicate{r}" for r in (1,2,3,4,5,6,7,8)],
    "A5 A1+B2 combo": [f"DataflowSystemGPT5MiniA5B2ComboReplicate{r}" for r in (1, 2, 3)],
    "A6 hints-only": [f"DataflowSystemGPT5MiniA6HintsOnlyReplicate{r}" for r in (1, 2, 3, 4)],
    "A0 fresh control (new sha)": [f"DataflowSystemGPT5MiniA0ControlReplicate{r}" for r in (9, 10, 11, 12)],
    "A0 control (uniform render)": [f"DataflowSystemGPT5MiniA0ControlReplicate{r}" for r in (1,2,3,4,5,6,7,8)],
}

print("20 hard tasks; accuracy = KramaBench answer-type metric, subset-scoped")
print("both arms: service :3002 @ 4af1e98da, src_dirty=False, single vintage\n")
print(f"{'arm':<30}{'acc mean±std':>16}{'cache%':>9}{'reason/task':>13}{'uncached-in':>12}{'$/task':>9}{'steps':>7}{'reps':>6}")
print("-" * 102)
rows = {}
for label, suts in ARMS.items():
    accs, sts = [], []
    for s in suts:
        a, n = arm_rep_score(s)
        if a is not None:
            accs.append(a)
        x = arm_rep_stats(s)
        if x:
            sts.append(x)
    if not accs:
        print(f"{label:<30} no data")
        continue
    m = st.mean(accs); sd = st.pstdev(accs) if len(accs) > 1 else 0
    rows[label] = dict(
        acc=m, sd=sd, accs=accs,
        cache=st.mean([x["cachepct"] for x in sts]),
        reas=st.mean([x["reas"] for x in sts]),
        unc=st.mean([x["tin"] - x["cached"] for x in sts]),
        tin=st.mean([x["tin"] for x in sts]),
        tout=st.mean([x["tout"] for x in sts]),
        cost=st.mean([x["cost"] for x in sts]),
        steps=st.mean([x["steps"] for x in sts]), reps=len(accs))
    r = rows[label]
    print(f"{label:<30}{m:>10.1f} ±{sd:4.1f}{r['cache']:>9.1f}{r['reas']:>13,.0f}"
          f"{r['unc']:>12,.0f}{r['cost']:>9.4f}{r['steps']:>7.1f}{r['reps']:>6}")

A = rows.get("A1 rich-source/lean-interior")
B = rows.get("A0 control (uniform render)")
if A and B:
    print(f"\n=== A1 - A0 ===")
    print(f"  accuracy    {A['acc']-B['acc']:+6.1f} pt   (A1 reps {['%.1f'%x for x in A['accs']]}, A0 reps {['%.1f'%x for x in B['accs']]})")
    print(f"  cache       {A['cache']-B['cache']:+6.1f} pp")
    print(f"  reasoning   {A['reas']-B['reas']:+7,.0f} tok/task")
    print(f"  uncached-in {A['unc']-B['unc']:+7,.0f} tok/task")
    print(f"  input tok   {A['tin']-B['tin']:+7,.0f} tok/task")
    print(f"  output tok  {A['tout']-B['tout']:+7,.0f} tok/task")
    print(f"  cost        {(A['cost']/B['cost']-1)*100:+6.1f} %")
    print(f"  steps       {A['steps']-B['steps']:+6.1f}")
    # variance gate: the measured randomness floor is ~+-4-5pt run-level
    d = A["acc"] - B["acc"]
    pooled = max(A["sd"], B["sd"])
    print(f"\n  variance gate: |delta| {abs(d):.1f} vs pooled rep std {pooled:.1f} "
          f"and the measured +-4-5pt run-level floor -> "
          f"{'INSIDE NOISE' if abs(d) <= max(pooled, 4.0) else 'outside noise'}")

    print(f"\n=== manipulation footprint (per rendered operator block) ===")
    print("  rows/stats are set directly by the policy = trustworthy;")
    print("  chars and n are NOT matched (arms build different pipelines).")
    fa, fb = footprint(ARMS["A1 rich-source/lean-interior"]), footprint(ARMS["A0 control (uniform render)"])
    print(f"\n{'arm':<22}{'src rows':>10}{'src stats':>11}{'src chars':>11}{'int rows':>10}{'int stats':>11}{'int chars':>11}{'n_src':>7}{'n_int':>7}")
    for lab, f in (("A1 policy", fa), ("A0 control", fb)):
        print(f"{lab:<22}{f['src']['mrows']:>10.1f}{f['src']['statspct']:>10.0f}%{f['src']['mchars']:>11.0f}"
              f"{f['int']['mrows']:>10.1f}{f['int']['statspct']:>10.0f}%{f['int']['mchars']:>11.0f}"
              f"{f['src']['n']:>7}{f['int']['n']:>7}")
    print(f"{'delta':<22}{fa['src']['mrows']-fb['src']['mrows']:>+10.1f}"
          f"{fa['src']['statspct']-fb['src']['statspct']:>+10.0f}%{fa['src']['mchars']-fb['src']['mchars']:>+11.0f}"
          f"{fa['int']['mrows']-fb['int']['mrows']:>+10.1f}"
          f"{fa['int']['statspct']-fb['int']['statspct']:>+10.0f}%{fa['int']['mchars']-fb['int']['mchars']:>+11.0f}")

    print(f"\n=== per-task flips (A1 mean vs A0 mean, >=0.5 swing) ===")
    flips = []
    for t in TASKS:
        a = [task_score(s, t) for s in ARMS["A1 rich-source/lean-interior"]]
        b = [task_score(s, t) for s in ARMS["A0 control (uniform render)"]]
        a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
        if not a or not b:
            continue
        d2 = st.mean(a) - st.mean(b)
        if abs(d2) >= 0.5:
            flips.append((t, st.mean(a), st.mean(b), d2))
    for t, x, y, d2 in sorted(flips, key=lambda z: -abs(z[3])):
        print(f"  {t:<24} A1 {x:.2f}  A0 {y:.2f}  {d2:+.2f}")
    won = sum(1 for f in flips if f[3] > 0); lost = len(flips) - won
    print(f"  {won} tasks better under policy, {lost} worse "
          f"(net {won-lost:+d} of {len(TASKS)} tasks)")
