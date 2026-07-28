#!/usr/bin/env python3
"""Mini-star collector: restrict kb.answer_scores to the focused-10 task set,
build the C1/C2/C3 matrix. Also pulls per-task cost/runtime/answer/gold.
Usage: python collect.py [scores|matrix]  (default matrix)
"""
import json, sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/bobflow/Kramabench"))
import kb

ARMS = {
    "anchor": "DataflowSystemGPT5MiniDelta1kSchemaOnly",
    "C1_5k":  "DataflowSystemGPT5MiniDelta5kSchemaOnly",
    "C2_stats": "DataflowSystemGPT5MiniDeltaStats1kD2",
    "C3_latest_code": "DataflowSystemGPT5MiniLatest1kCodeInSnap",
}
TASKS = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "tasks10.txt")) if l.strip()]
PASS = 0.9

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}

def arm_scores(sut):
    s = kb.answer_scores(sut)
    return {t: s.get(t) for t in TASKS}

def task_detail(sut, t):
    d = kb.KB_ROOT / "system_scratch" / sut / t
    ev, gt = load(d / "evaluation.json"), load(d / "ground_truth.json")
    return {
        "answer": (ev.get("model_output") or {}).get("answer"),
        "gold": gt.get("answer"),
        "atype": gt.get("answer_type"),
        "cost": ev.get("cost_usd") or ev.get("cost_usd_sut"),
        "tok_in": ev.get("token_usage_sut_input"),
        "tok_out": ev.get("token_usage_sut_output"),
        "runtime": ev.get("runtime"),
    }

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    # `failed <armkey>`: print space-separated failed (score<0.9) ids for one arm
    if mode == "failed":
        armkey = sys.argv[2]
        s = arm_scores(ARMS[armkey])
        print(" ".join(t for t in TASKS if (s.get(t) or 0) < PASS))
        return
    sc = {k: arm_scores(v) for k, v in ARMS.items()}
    if mode == "scores":
        print(json.dumps({k: sc[k] for k in ARMS}, indent=1))
        return
    # `snapshot <label>`: dump current 10-task scores per arm to snap_<label>.json
    if mode == "snapshot":
        label = sys.argv[2]
        p = os.path.join(os.path.dirname(__file__), f"snap_{label}.json")
        json.dump({k: sc[k] for k in ARMS}, open(p, "w"), indent=1)
        print(f"wrote {p}")
        return
    # matrix
    cols = ["anchor", "C1_5k", "C2_stats", "C3_latest_code"]
    print(f"{'task':<22} " + " ".join(f"{c:>15}" for c in cols))
    passc = {c: 0 for c in cols}
    for t in TASKS:
        row = []
        for c in cols:
            v = sc[c][t]
            mark = "?" if v is None else ("P" if v >= PASS else ".")
            if v is not None and v >= PASS:
                passc[c] += 1
            row.append(f"{('' if v is None else f'{v:.2f}')}{mark:>2}")
        print(f"{t:<22} " + " ".join(f"{r:>15}" for r in row))
    print(f"{'PASS/10':<22} " + " ".join(f"{passc[c]:>14}P" for c in cols))
    # comparison deltas
    def flips(a, b):
        aw = [t for t in TASKS if (sc[a][t] or 0) >= PASS and (sc[b][t] or 0) < PASS]
        bw = [t for t in TASKS if (sc[b][t] or 0) >= PASS and (sc[a][t] or 0) < PASS]
        both = [t for t in TASKS if (sc[a][t] or 0) >= PASS and (sc[b][t] or 0) >= PASS]
        neither = [t for t in TASKS if (sc[a][t] or 0) < PASS and (sc[b][t] or 0) < PASS]
        return aw, bw, both, neither
    print("\n=== comparisons (A=anchor) ===")
    for name, b in [("C1 sampling 1k->5k", "C1_5k"),
                    ("C2 stats off->on", "C2_stats"),
                    ("C3 delta->latest+code", "C3_latest_code")]:
        aw, bw, both, neither = flips("anchor", b)
        print(f"\n{name}: anchor={passc['anchor']} {b}={passc[b]}")
        print(f"  both pass: {both}")
        print(f"  anchor-only (B lost): {aw}")
        print(f"  {b}-only (B won): {bw}")
        print(f"  both fail: {neither}")

if __name__ == "__main__":
    main()
