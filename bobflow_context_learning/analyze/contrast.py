#!/usr/bin/env python
"""
Contrastive analysis: latest-core vs full-delta-trajectory, per task.

Reads the per-task evaluation.json that evaluate.py writes for each arm
(system_scratch/<arm>/<task_id>/evaluation.json), pulls the accuracy metric
(by answer_type) and the SUT token usage, and reports the Gate-0 headroom
signal: how often delta beats latest, the reverse, token cost, and ΔQ.

Usage:
    python bobflow_context_learning/analyze/contrast.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "system_scratch")
MANIFEST = os.path.join(ROOT, "bobflow_context_learning", "data", "gate0_manifest.json")
OUT = os.path.join(ROOT, "bobflow_context_learning", "data", "contrast.json")
LATEST = "DataflowSystemGPT54Gate0Latest"
DELTA = "DataflowSystemGPT54Gate0Delta"

# session_evaluator.ANSWER_TYPE_TO_METRIC (the key present in evaluation.json)
ANSWER_TYPE_TO_METRIC = {
    "numeric_exact": "success",
    "numeric_approximate": "rae_score",
    "string_exact": "success",
    "string_approximate": "llm_paraphrase",
    "list_exact": "f1",
    "list_approximate": "f1_approximate",
}
ACCURACY_KEYS = ["rae_score", "success", "f1", "f1_approximate", "llm_paraphrase"]
SUCCESS_THRESHOLD = 0.9  # SessionEvaluator default


def read_eval(arm, task_id):
    p = os.path.join(SCRATCH, arm, task_id, "evaluation.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def accuracy(ev, answer_type):
    if ev is None:
        return None
    key = ANSWER_TYPE_TO_METRIC.get(answer_type)
    if key and key in ev and isinstance(ev[key], (int, float)):
        return float(ev[key])
    for k in ACCURACY_KEYS:
        if k in ev and isinstance(ev[k], (int, float)):
            return float(ev[k])
    return None


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    rows = []
    for item in manifest["sample"]:
        tid, atype = item["task_id"], item.get("answer_type")
        el, ed = read_eval(LATEST, tid), read_eval(DELTA, tid)
        ql, qd = accuracy(el, atype), accuracy(ed, atype)
        tl = el.get("token_usage_sut") if el else None
        td = ed.get("token_usage_sut") if ed else None
        row = {
            "task_id": tid, "workload": item["workload"], "answer_type": atype,
            "q_latest": ql, "q_delta": qd,
            "delta_q": (qd - ql) if (ql is not None and qd is not None) else None,
            "tok_latest": tl, "tok_delta": td,
            "ans_latest": (el or {}).get("model_output", {}).get("answer") if el else None,
            "ans_delta": (ed or {}).get("model_output", {}).get("answer") if ed else None,
        }
        # case type using success threshold
        if ql is not None and qd is not None:
            sl, sd = ql >= SUCCESS_THRESHOLD, qd >= SUCCESS_THRESHOLD
            row["case_type"] = (
                "delta_win_latest_fail" if (sd and not sl) else
                "latest_win_delta_fail" if (sl and not sd) else
                "both_success" if (sl and sd) else "both_fail"
            )
        else:
            row["case_type"] = "incomplete"
        rows.append(row)

    done = [r for r in rows if r["case_type"] != "incomplete"]
    print(f"\n{'task':24s} {'type':20s} {'Qlat':>6s} {'Qdel':>6s} {'ΔQ':>7s} {'tokLat':>8s} {'tokDel':>8s}  case")
    print("-" * 110)
    for r in rows:
        ql = f"{r['q_latest']:.3f}" if r['q_latest'] is not None else "  -  "
        qd = f"{r['q_delta']:.3f}" if r['q_delta'] is not None else "  -  "
        dq = f"{r['delta_q']:+.3f}" if r['delta_q'] is not None else "   -  "
        tl = str(r['tok_latest'] or "-"); td = str(r['tok_delta'] or "-")
        print(f"{r['task_id']:24s} {str(r['answer_type']):20s} {ql:>6s} {qd:>6s} {dq:>7s} {tl:>8s} {td:>8s}  {r['case_type']}")

    if done:
        import statistics as st
        ml = st.mean(r["q_latest"] for r in done)
        md = st.mean(r["q_delta"] for r in done)
        mtl = st.mean(r["tok_latest"] for r in done if r["tok_latest"])
        mtd = st.mean(r["tok_delta"] for r in done if r["tok_delta"])
        from collections import Counter
        cases = Counter(r["case_type"] for r in done)
        print("\n===== HEADROOM SUMMARY (%d/%d tasks complete) =====" % (len(done), len(rows)))
        print(f"  mean accuracy   latest={ml:.3f}   delta={md:.3f}   Δ={md-ml:+.3f}")
        print(f"  mean SUT tokens latest={mtl:,.0f}  delta={mtd:,.0f}  (delta/latest = {mtd/mtl:.2f}x)")
        print(f"  accuracy / 1k tok  latest={ml/(mtl/1000):.4f}  delta={md/(mtd/1000):.4f}")
        print(f"  case types: {dict(cases)}")
        kill = cases.get("delta_win_latest_fail", 0)
        print(f"\n  GO/NO-GO signal — delta_win_latest_fail = {kill}/{len(done)}"
              f"  ({'headroom exists → worth building the selector' if kill > 0 else 'NO headroom → latest already sufficient'})")

    with open(OUT, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
