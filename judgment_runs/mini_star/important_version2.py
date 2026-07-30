#!/usr/bin/env python3
"""S1 drill-down: was the rebuilt version actually LOST, or just ignored?

A rebuild only argues for retention if the earlier version was GONE from the
rendered context when the agent rewrote it. If it was still on screen, retention
is not the fix (same conclusion the error analysis reached).

For every rebuild event we therefore classify:
  VISIBLE  — a >=0.85-similar Code: block was still rendered at the rebuild step
  EVICTED  — it was not; the agent rewrote from memory / from scratch
and split by same-operator vs cross-operator, plus confounds (steps, difficulty).
"""
import json, glob, re, sys, difflib, statistics as st
from collections import defaultdict

ARMS = sys.argv[1:] or (
    [f"DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate{r}" for r in range(5)]
    + [f"DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate{r}" for r in range(5)]
)
OP_BLOCK = re.compile(r"### Operator `([^`]+)` \([^)]*\)(.*?)(?=\n### |\Z)", re.S)
CODE_BLOCK = re.compile(r"Code:\s*\n(.*?)(?=\n\s*(?:Summary|Output Table|Result|Status|###)|\Z)", re.S)
ATM = {"numeric_exact": "success", "string_exact": "success", "list_exact": "f1",
       "numeric_approximate": "rae_score", "list_approximate": "f1_approximate",
       "string_approximate": "llm_paraphrase"}


def norm(c):
    return re.sub(r"\s+", " ", re.sub(r"#.*", "", c or "")).strip()


def score(d):
    try:
        ev = json.load(open(f"{d}/evaluation.json")); gt = json.load(open(f"{d}/ground_truth.json"))
    except Exception:
        return None
    k = ATM.get(gt.get("answer_type") or "")
    v = ev.get(k) if isinstance(ev.get(k), (int, float)) else None
    if v is None:
        c = [float(ev[m]) for m in ATM.values() if isinstance(ev.get(m), (int, float))]
        v = max(c) if c else None
    return v


events, runs = [], []
for arm in ARMS:
    for f in glob.glob(f"system_scratch/{arm}/*/react_steps.json"):
        task = f.split("/")[-2]
        sc = score("/".join(f.split("/")[:-1]))
        if sc is None:
            continue
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        steps = doc.get("steps", [])
        subs, ctx_at = [], {}
        for i, s in enumerate(steps):
            if s.get("inputMessages"):
                ctx_at[i] = "\n".join(str(m.get("content") or "") for m in s["inputMessages"])
            if s.get("role") != "agent":
                continue
            for tc in s.get("toolCalls") or []:
                inp = tc.get("input") or {}
                if inp.get("operatorId") and inp.get("code"):
                    subs.append((i, inp["operatorId"], norm(inp["code"])))
        passed = 1 if sc >= 0.9 else 0
        run = dict(task=task, passed=passed, steps=len(steps), vis=0, evi=0,
                   vis_x=0, evi_x=0, hard="hard" in task)
        for j in range(len(subs)):
            sj, opj, cj = subs[j]
            if len(cj) < 60:
                continue
            for i in range(j):
                si, opi, ci = subs[i]
                if len(ci) < 60 or sj - si < 2:
                    continue
                superseded = any(o == opi and s > si for s, o, _ in subs[i + 1:])
                if not superseded:
                    continue
                if difflib.SequenceMatcher(None, ci, cj).ratio() < 0.85:
                    continue
                # was a similar Code: block still rendered at the rebuild step?
                ctx = ctx_at.get(sj) or max((ctx_at[k] for k in ctx_at if k <= sj), key=len, default="")
                rendered = [norm(m.group(1)) for b in OP_BLOCK.finditer(ctx)
                            for m in CODE_BLOCK.finditer(b.group(2))]
                visible = any(difflib.SequenceMatcher(None, ci, rc).ratio() >= 0.85 for rc in rendered)
                cross = opi != opj
                run["vis" if visible else "evi"] += 1
                if cross:
                    run["vis_x" if visible else "evi_x"] += 1
                events.append(dict(task=task, arm=arm, passed=passed, visible=visible,
                                   cross=cross, gap=sj - si, opi=opi, opj=opj))
                break
        runs.append(run)

n = len(runs)
print(f"runs={n}  rebuild events={len(events)}\n")


def bucket(pred, label):
    has = [r for r in runs if pred(r)]
    non = [r for r in runs if not pred(r)]
    if not has:
        print(f"{label:<38}{0:>6}      —        —")
        return
    ph = st.mean([r["passed"] for r in has]) * 100
    pn = st.mean([r["passed"] for r in non]) * 100
    sh = st.mean([r["steps"] for r in has]); sn = st.mean([r["steps"] for r in non])
    hd = st.mean([r["hard"] for r in has]) * 100
    print(f"{label:<38}{len(has):>6}{ph:>9.1f}%{pn:>9.1f}%{ph-pn:>+8.1f}{sh:>8.1f}{sn:>7.1f}{hd:>8.0f}%")


print(f"{'run bucket':<38}{'n':>6}{'pass':>9}{'pass|not':>10}{'delta':>8}{'steps':>8}{'vs':>7}{'hard':>8}")
print("-" * 94)
bucket(lambda r: r["evi"] > 0, "rebuilt an EVICTED version")
bucket(lambda r: r["vis"] > 0, "rebuilt a VISIBLE version")
bucket(lambda r: r["evi_x"] > 0, "  ...evicted AND cross-operator")
bucket(lambda r: r["vis_x"] > 0, "  ...visible AND cross-operator")

ev = [e for e in events if not e["visible"]]
vi = [e for e in events if e["visible"]]
print(f"\nevents: {len(ev)} evicted ({sum(e['cross'] for e in ev)} cross-op), "
      f"{len(vi)} visible ({sum(e['cross'] for e in vi)} cross-op)")
if ev:
    print(f"  evicted rebuild step-gap: median {st.median([e['gap'] for e in ev]):.0f}, max {max(e['gap'] for e in ev)}")
tt = defaultdict(int)
for e in ev:
    tt[e["task"]] += 1
print("  top tasks rebuilding evicted code:", ", ".join(f"{k}({v})" for k, v in sorted(tt.items(), key=lambda x: -x[1])[:8]))
