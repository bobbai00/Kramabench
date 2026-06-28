#!/usr/bin/env python
"""
All-domains latest-vs-delta comparison: accuracy, TOKEN-COST BREAKDOWN, case-type
splits, and trajectory characteristics — the strengths/drawbacks report.

Reads the manifest written by collect/run_batch.py (which records the two arm
class names and the task sample), then for each (task, arm) pulls evaluate.py's
evaluation.json (accuracy metric by answer_type + token usage breakdown) and
react_steps.json (steps / operators / churn). Arm-generic: the two arms are taken
from the manifest; the one whose name contains "Latest"/"Delta" is labelled
accordingly.

Usage:
    python bobflow_context_learning/analyze/compare_all.py
"""
import json
import os
import statistics as st
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "system_scratch")
MANIFEST = os.path.join(ROOT, "bobflow_context_learning", "data", "gate0_manifest.json")
OUT = os.path.join(ROOT, "bobflow_context_learning", "data", "compare_all.json")

ANSWER_TYPE_TO_METRIC = {
    "numeric_exact": "success", "numeric_approximate": "rae_score",
    "string_exact": "success", "string_approximate": "llm_paraphrase",
    "list_exact": "f1", "list_approximate": "f1_approximate",
}
ACCURACY_KEYS = ["rae_score", "success", "f1", "f1_approximate", "llm_paraphrase"]
SUCCESS_THRESHOLD = 0.9
TOK_FIELDS = ["token_usage_sut_input", "token_usage_sut_output",
              "token_usage_sut_reasoning", "token_usage_sut_cached"]


def read_eval(arm, tid):
    p = os.path.join(SCRATCH, arm, tid, "evaluation.json")
    return json.load(open(p)) if os.path.exists(p) else None


def accuracy(ev, atype):
    if ev is None:
        return None
    key = ANSWER_TYPE_TO_METRIC.get(atype)
    if key and isinstance(ev.get(key), (int, float)):
        return float(ev[key])
    for k in ACCURACY_KEYS:
        if isinstance(ev.get(k), (int, float)):
            return float(ev[k])
    return None


def trace_stats(arm, tid):
    """steps / distinct ops / re-edits / hit-cap from react_steps.json."""
    p = os.path.join(SCRATCH, arm, tid, "react_steps.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    steps = d if isinstance(d, list) else d.get("steps", [])
    agent = [s for s in steps if s.get("role") == "agent"]
    edits = Counter()
    for s in agent:
        for tc in s.get("toolCalls") or []:
            if tc.get("toolName") == "createOrModifyOperator":
                op = (tc.get("input") or {}).get("operatorId")
                if op:
                    edits[op] += 1
    return {"agent_steps": len(agent), "distinct_ops": len(edits),
            "total_edits": sum(edits.values()), "reedits": sum(c - 1 for c in edits.values() if c > 1)}


def difficulty(tid):
    return "hard" if "hard" in tid else ("easy" if "easy" in tid else "other")


def fmt(x, w=7, p=0):
    return (f"{x:,.{p}f}" if isinstance(x, (int, float)) else str(x)).rjust(w)


def main():
    man = json.load(open(MANIFEST))
    arms = man["arms"]
    L = next((a for a in arms if "Latest" in a), arms[0])
    D = next((a for a in arms if "Delta" in a), arms[-1])
    rows = []
    for item in man["sample"]:
        tid, atype, dom = item["task_id"], item.get("answer_type"), item["workload"]
        el, ed = read_eval(L, tid), read_eval(D, tid)
        ql, qd = accuracy(el, atype), accuracy(ed, atype)
        r = {"task_id": tid, "workload": dom, "difficulty": difficulty(tid), "answer_type": atype,
             "q_latest": ql, "q_delta": qd,
             "ev_latest": el, "ev_delta": ed,
             "tr_latest": trace_stats(L, tid), "tr_delta": trace_stats(D, tid)}
        if ql is not None and qd is not None:
            sl, sd = ql >= SUCCESS_THRESHOLD, qd >= SUCCESS_THRESHOLD
            r["case"] = ("delta_win" if sd and not sl else "latest_win" if sl and not sd
                         else "both_success" if sl else "both_fail")
            r["delta_q"] = qd - ql
        else:
            r["case"] = "incomplete"
        rows.append(r)

    done = [r for r in rows if r["case"] != "incomplete"]
    print(f"\n===== COVERAGE: {len(done)}/{len(rows)} complete pairs "
          f"({len(rows)-len(done)} incomplete) | arms: {L} vs {D} =====")

    # ---- accuracy by domain ----
    print("\n----- ACCURACY by domain (complete pairs) -----")
    print(f"{'domain':12s} {'n':>3s} {'acc_lat':>8s} {'acc_del':>8s} {'Δacc':>7s}   cases")
    bydom = defaultdict(list)
    for r in done:
        bydom[r["workload"]].append(r)
    for dom in sorted(bydom):
        rs = bydom[dom]
        al, ad = st.mean(r["q_latest"] for r in rs), st.mean(r["q_delta"] for r in rs)
        cc = Counter(r["case"] for r in rs)
        sh = {"both_fail": "BF", "both_success": "BS", "latest_win": "L>D", "delta_win": "D>L"}
        cs = " ".join(f"{sh[k]}:{v}" for k, v in cc.items())
        print(f"{dom:12s} {len(rs):>3d} {al:8.3f} {ad:8.3f} {ad-al:+7.3f}   {cs}")

    # ---- accuracy by difficulty ----
    print("\n----- ACCURACY by difficulty -----")
    bydiff = defaultdict(list)
    for r in done:
        bydiff[r["difficulty"]].append(r)
    for diff in ["easy", "hard", "other"]:
        rs = bydiff.get(diff, [])
        if not rs:
            continue
        al, ad = st.mean(r["q_latest"] for r in rs), st.mean(r["q_delta"] for r in rs)
        print(f"  {diff:6s} n={len(rs):3d}  acc_latest={al:.3f}  acc_delta={ad:.3f}  Δ={ad-al:+.3f}")

    # ---- token-cost breakdown ----
    def tok_summary(arm, key_ev):
        evs = [r[key_ev] for r in done if r[key_ev]]
        agg = {f: st.mean(e.get(f, 0) for e in evs) for f in TOK_FIELDS}
        agg["total"] = st.mean(e.get("token_usage_sut", 0) for e in evs)
        agg["cost_usd"] = st.mean(e.get("cost_usd_sut", 0) for e in evs)
        agg["runtime_s"] = st.mean(e.get("runtime", 0) for e in evs)
        steps = [r[key_ev.replace("ev", "tr")].get("agent_steps", 0) for r in done]
        agg["agent_steps"] = st.mean(s for s in steps if s)
        return agg

    print("\n----- TOKEN-COST BREAKDOWN (mean per task, complete pairs) -----")
    sl, sd = tok_summary(L, "ev_latest"), tok_summary(D, "ev_delta")
    print(f"{'metric':18s} {'latest':>12s} {'delta':>12s} {'delta/latest':>13s}")
    for label, k in [("input tok", "token_usage_sut_input"), ("output tok", "token_usage_sut_output"),
                     ("reasoning tok", "token_usage_sut_reasoning"), ("cached tok", "token_usage_sut_cached"),
                     ("TOTAL tok", "total"), ("cost USD", "cost_usd"),
                     ("agent steps", "agent_steps"), ("runtime s", "runtime_s")]:
        vl, vd = sl.get(k, 0), sd.get(k, 0)
        ratio = (vd / vl) if vl else float("nan")
        p = 4 if k == "cost_usd" else (1 if k in ("agent_steps", "runtime_s") else 0)
        print(f"{label:18s} {fmt(vl,12,p)} {fmt(vd,12,p)} {ratio:12.2f}x")

    # accuracy efficiency
    al = st.mean(r["q_latest"] for r in done)
    ad = st.mean(r["q_delta"] for r in done)
    print(f"\n  OVERALL accuracy   latest={al:.3f}   delta={ad:.3f}   Δ={ad-al:+.3f}")
    print(f"  acc per 1k tok     latest={al/(sl['total']/1000):.4f}   delta={ad/(sd['total']/1000):.4f}")
    print(f"  tokens per step    latest={sl['total']/max(sl['agent_steps'],1):,.0f}   "
          f"delta={sd['total']/max(sd['agent_steps'],1):,.0f}")

    # ---- case types + divergent cases ----
    cc = Counter(r["case"] for r in done)
    print(f"\n----- CASE TYPES (n={len(done)}) -----\n  {dict(cc)}")
    print("\n  DELTA WINS (delta right, latest wrong):")
    for r in [r for r in done if r["case"] == "delta_win"]:
        print(f"    {r['task_id']:22s} tokL={r['ev_latest'].get('token_usage_sut')} tokD={r['ev_delta'].get('token_usage_sut')}")
    print("  LATEST WINS (latest right, delta wrong):")
    for r in [r for r in done if r["case"] == "latest_win"]:
        print(f"    {r['task_id']:22s} tokL={r['ev_latest'].get('token_usage_sut')} tokD={r['ev_delta'].get('token_usage_sut')}")

    # ---- trajectory characteristics ----
    print("\n----- TRAJECTORY CHARACTERISTICS (mean) -----")
    for label, k in [("agent_steps", "agent_steps"), ("distinct_ops", "distinct_ops"),
                     ("total_edits", "total_edits"), ("reedits(churn)", "reedits")]:
        vl = st.mean(r["tr_latest"].get(k.split("(")[0], 0) for r in done if r["tr_latest"])
        vd = st.mean(r["tr_delta"].get(k.split("(")[0], 0) for r in done if r["tr_delta"])
        print(f"  {label:16s} latest={vl:5.2f}   delta={vd:5.2f}")

    # strip heavy ev objects before writing
    for r in rows:
        for k in ("ev_latest", "ev_delta"):
            ev = r.get(k)
            if ev:
                r[k] = {f: ev.get(f) for f in TOK_FIELDS + ["token_usage_sut", "cost_usd_sut", "runtime", "success"]}
    json.dump({"arms": [L, D], "rows": rows,
               "token_summary": {"latest": sl, "delta": sd}}, open(OUT, "w"), indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
