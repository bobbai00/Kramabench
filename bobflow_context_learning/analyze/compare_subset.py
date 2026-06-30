#!/usr/bin/env python
"""
Verification view for the hybrid e2e: compare latest / delta / hybrid arms on the
representative subset, per task and in aggregate. Answers the three questions the
selective-reinjection method must satisfy:
  1. Does the hybrid RECOVER delta's wins (tasks where latest failed, delta passed)?
  2. Does it RESCUE thrash both_fail tasks (fail under both baselines -> pass)?
  3. Does it REGRESS the both_pass controls, and at what COST?

Reads scores from each arm's evaluation.json (answer_type metric, >=TH) and cost
from stats.json. Arms passed on the CLI; the subset from repr_subset.json (or all
shared tasks if --subset omitted).

Usage:
  python bobflow_context_learning/analyze/compare_subset.py \
      --subset bobflow_context_learning/data/repr_subset.json \
      --arms LATEST DELTA ERRORREFLECT REINJECT
"""
import argparse
import json
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "system_scratch")
TH = 0.9
A2M = {"numeric_exact": "success", "numeric_approximate": "rae_score", "string_exact": "success",
       "string_approximate": "llm_paraphrase", "list_exact": "f1", "list_approximate": "f1_approximate"}
SK = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def score(ev, at):
    if ev is None:
        return None
    k = A2M.get(at)
    if k and isinstance(ev.get(k), (int, float)):
        return float(ev[k])
    vs = [float(ev[x]) for x in SK if isinstance(ev.get(x), (int, float))]
    return max(vs) if vs else None


def rec(arm, t):
    d = os.path.join(SCRATCH, arm, t)
    ev = _load(os.path.join(d, "evaluation.json"))
    at = (_load(os.path.join(d, "ground_truth.json")) or {}).get("answer_type")
    stt = _load(os.path.join(d, "stats.json")) or {}
    sc = score(ev, at)
    return {"score": sc, "pass": sc is not None and sc >= TH,
            "cost": float(stt.get("cost_usd", 0) or 0), "steps": int(stt.get("num_steps", 0) or 0),
            "present": ev is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None)
    ap.add_argument("--arms", nargs="+", required=True, help="order: latest delta [hybrids...]")
    ap.add_argument("--labels", nargs="+", default=None, help="short labels per arm")
    args = ap.parse_args()
    arms = args.arms
    labels = args.labels or [a.replace("DataflowSystemGPT54", "").replace("SchemaConverge", "")[:10] for a in arms]
    L, D = arms[0], arms[1]  # baselines: latest, delta

    if args.subset:
        tasks = json.load(open(args.subset))
    else:
        tasks = sorted(set.intersection(*[set(os.listdir(os.path.join(SCRATCH, a))) for a in arms]))

    data = {t: {a: rec(a, t) for a in arms} for t in tasks}
    # baseline case (latest vs delta)
    def basecase(t):
        l, d = data[t][L]["pass"], data[t][D]["pass"]
        return ("both_pass" if l and d else "latest_win" if l else "delta_win" if d else "both_fail")

    print(f"\n===== HYBRID VERIFICATION on {len(tasks)} subset tasks =====")
    print(f"arms: " + " | ".join(f"{lb}={a}" for lb, a in zip(labels, arms)))
    hdr = f"{'task':22s} {'base':10s} " + " ".join(f"{lb[:8]:>8s}" for lb in labels)
    print("\n" + hdr); print("-" * len(hdr))
    miss = []
    for t in tasks:
        bc = basecase(t)
        cells = []
        for a in arms:
            r = data[t][a]
            if not r["present"]:
                cells.append("  --  "); miss.append((a, t)); continue
            mark = "P" if r["pass"] else "."
            cells.append(f"{r['score']:.2f}{mark}")
        print(f"{t:22s} {bc:10s} " + " ".join(f"{c:>8s}" for c in cells))

    # aggregate accuracy + cost per arm
    print("\n----- AGGREGATE (subset) -----")
    print(f"{'arm':12s} {'acc':>6s} {'cost$':>8s} {'$/task':>8s} {'steps':>6s}")
    for lb, a in zip(labels, arms):
        rs = [data[t][a] for t in tasks if data[t][a]["present"]]
        if not rs:
            print(f"{lb:12s}  (no data)"); continue
        acc = sum(r["pass"] for r in rs) / len(rs)
        cost = sum(r["cost"] for r in rs)
        steps = st.mean(r["steps"] for r in rs)
        print(f"{lb:12s} {acc:6.3f} {cost:8.2f} {cost/len(rs):8.4f} {steps:6.1f}  (n={len(rs)})")

    # the three verification questions (use first hybrid = arms[2] if present)
    if len(arms) >= 3:
        for hidx in range(2, len(arms)):
            H = arms[hidx]; hl = labels[hidx]
            present = [t for t in tasks if data[t][H]["present"]]
            recov = [t for t in present if basecase(t) == "delta_win" and data[t][H]["pass"]]
            dwin = [t for t in present if basecase(t) == "delta_win"]
            kept = [t for t in present if basecase(t) == "latest_win" and data[t][H]["pass"]]
            lwin = [t for t in present if basecase(t) == "latest_win"]
            rescue = [t for t in present if basecase(t) == "both_fail" and data[t][H]["pass"]]
            bf = [t for t in present if basecase(t) == "both_fail"]
            regress = [t for t in present if basecase(t) == "both_pass" and not data[t][H]["pass"]]
            bp = [t for t in present if basecase(t) == "both_pass"]
            print(f"\n----- {hl} vs baselines -----")
            print(f"  recovered delta_wins : {len(recov)}/{len(dwin)}   {recov}")
            print(f"  kept latest_wins     : {len(kept)}/{len(lwin)}   (lost: {[t for t in lwin if t not in kept]})")
            print(f"  RESCUED both_fail    : {len(rescue)}/{len(bf)}   {rescue}")
            print(f"  regressed both_pass  : {len(regress)}/{len(bp)}   {regress}")

    if miss:
        print(f"\n  NOTE: {len(miss)} (arm,task) missing evals (incomplete/timeout): {miss[:8]}")


if __name__ == "__main__":
    main()
