#!/usr/bin/env python
"""
Compare two SUT arms task-by-task DIRECTLY from system_scratch (no manifest, no
results CSV needed): accuracy by domain/difficulty, the latest-vs-delta case-type
cross, a token-cost + prompt-cache breakdown, and the divergent tasks.

Accuracy per task = the answer_type's metric from evaluation.json (success /
rae_score / f1 / llm_paraphrase / f1_approximate), thresholded at >= TH. Cost,
tokens, steps, and cached tokens come from stats.json (litellm cost_usd).

Usage:
    python bobflow_context_learning/analyze/compare_arms.py <LATEST_SUT> <DELTA_SUT>
"""
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "system_scratch")
TH = 0.9
ANSWER_TYPE_TO_METRIC = {
    "numeric_exact": "success", "numeric_approximate": "rae_score",
    "string_exact": "success", "string_approximate": "llm_paraphrase",
    "list_exact": "f1", "list_approximate": "f1_approximate",
}
SCORE_KEYS = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def score_of(ev, atype):
    if ev is None:
        return None
    k = ANSWER_TYPE_TO_METRIC.get(atype)
    if k and isinstance(ev.get(k), (int, float)):
        return float(ev[k])
    vals = [float(ev[x]) for x in SCORE_KEYS if isinstance(ev.get(x), (int, float))]
    return max(vals) if vals else None


def difficulty(tid):
    return "hard" if "hard" in tid else ("easy" if "easy" in tid else "?")


def domain(tid):
    return tid.rsplit("-", 2)[0]


def collect(arm):
    """task_id -> {score, pass, cost, tot, cached, steps, atype, answer}"""
    base = os.path.join(SCRATCH, arm)
    out = {}
    for tid in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, tid)
        if not os.path.isdir(d):
            continue
        ev = _load(os.path.join(d, "evaluation.json"))
        gt = _load(os.path.join(d, "ground_truth.json")) or {}
        stt = _load(os.path.join(d, "stats.json")) or {}
        atype = gt.get("answer_type")
        sc = score_of(ev, atype)
        out[tid] = {
            "score": sc, "pass": (sc is not None and sc >= TH),
            "atype": atype,
            "cost": float(stt.get("cost_usd", 0) or 0),
            "tot": int(stt.get("total_tokens", 0) or 0),
            "inp": int(stt.get("input_tokens", 0) or 0),
            "cached": int(stt.get("cached_tokens", 0) or 0),
            "steps": int(stt.get("num_steps", 0) or 0),
            "answer": ((ev or {}).get("model_output") or {}).get("answer"),
        }
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: compare_arms.py <LATEST_SUT> <DELTA_SUT>")
    LSUT, DSUT = sys.argv[1], sys.argv[2]
    L, D = collect(LSUT), collect(DSUT)
    tids = sorted(set(L) & set(D))
    print(f"\n===== {len(tids)} shared tasks | LATEST={LSUT}  DELTA={DSUT} (pass = score>={TH}) =====")

    rows = []
    for t in tids:
        l, d = L[t], D[t]
        if l["score"] is None or d["score"] is None:
            case = "incomplete"
        elif l["pass"] and d["pass"]:
            case = "both_pass"
        elif l["pass"]:
            case = "latest_win"
        elif d["pass"]:
            case = "delta_win"
        else:
            case = "both_fail"
        rows.append({"t": t, "dom": domain(t), "diff": difficulty(t), "case": case, "l": l, "d": d})

    done = [r for r in rows if r["case"] != "incomplete"]

    # accuracy + cost by domain
    print(f"\n{'domain':12s} {'n':>3s} {'accL':>5s} {'accD':>5s} {'Δacc':>6s} {'$L':>7s} {'$D':>7s}   cases")
    bydom = defaultdict(list)
    for r in done:
        bydom[r["dom"]].append(r)
    sh = {"both_fail": "BF", "both_pass": "BP", "latest_win": "L>D", "delta_win": "D>L"}
    for dom in sorted(bydom):
        rs = bydom[dom]
        aL = sum(r["l"]["pass"] for r in rs) / len(rs)
        aD = sum(r["d"]["pass"] for r in rs) / len(rs)
        cL = sum(r["l"]["cost"] for r in rs)
        cD = sum(r["d"]["cost"] for r in rs)
        cc = Counter(r["case"] for r in rs)
        cs = " ".join(f"{sh[k]}:{v}" for k, v in cc.items())
        print(f"{dom:12s} {len(rs):>3d} {aL:5.2f} {aD:5.2f} {aD-aL:+6.2f} {cL:7.2f} {cD:7.2f}   {cs}")

    # by difficulty
    print("\n----- by difficulty -----")
    bydiff = defaultdict(list)
    for r in done:
        bydiff[r["diff"]].append(r)
    for diff in ["easy", "hard", "?"]:
        rs = bydiff.get(diff, [])
        if not rs:
            continue
        aL = sum(r["l"]["pass"] for r in rs) / len(rs)
        aD = sum(r["d"]["pass"] for r in rs) / len(rs)
        print(f"  {diff:5s} n={len(rs):3d}  accL={aL:.3f} accD={aD:.3f} Δ={aD-aL:+.3f}  "
              f"$L={sum(r['l']['cost'] for r in rs):.2f} $D={sum(r['d']['cost'] for r in rs):.2f}")

    # overall + cache
    aL = sum(r["l"]["pass"] for r in done) / len(done)
    aD = sum(r["d"]["pass"] for r in done) / len(done)
    cL = sum(r["l"]["cost"] for r in done)
    cD = sum(r["d"]["cost"] for r in done)
    cacheL = sum(r["l"]["cached"] for r in done) / max(sum(r["l"]["inp"] for r in done), 1)
    cacheD = sum(r["d"]["cached"] for r in done) / max(sum(r["d"]["inp"] for r in done), 1)
    print(f"\n===== OVERALL ({len(done)} tasks) =====")
    print(f"  accuracy   latest={aL:.3f}   delta={aD:.3f}   Δ={aD-aL:+.3f}")
    print(f"  cost $     latest={cL:.2f}    delta={cD:.2f}    (delta/latest={cD/cL:.2f}x)")
    print(f"  $/task     latest={cL/len(done):.4f}  delta={cD/len(done):.4f}")
    print(f"  cache hit  latest={cacheL:.1%}   delta={cacheD:.1%}  (cached/input tokens)")
    print(f"  steps avg  latest={st.mean(r['l']['steps'] for r in done):.1f}   "
          f"delta={st.mean(r['d']['steps'] for r in done):.1f}")
    cc = Counter(r["case"] for r in done)
    print(f"  cases: {dict(cc)}")
    # oracle: per task, pass if EITHER arm passes; cost = cheaper passing arm (or cheaper arm if both fail)
    orc_acc = sum(1 for r in done if r["l"]["pass"] or r["d"]["pass"]) / len(done)
    orc_cost = 0.0
    for r in done:
        cands = [a for a in (r["l"], r["d"]) if a["pass"]] or [r["l"], r["d"]]
        orc_cost += min(a["cost"] for a in cands)
    print(f"  ORACLE (best arm/task): acc={orc_acc:.3f}  cost=${orc_cost:.2f}  "
          f"(headroom over latest: +{orc_acc-aL:.3f} acc)")

    print("\n----- DIVERGENT TASKS -----")
    for r in done:
        if r["case"] in ("delta_win", "latest_win"):
            print(f"  {r['case']:10s} {r['t']:22s} scL={r['l']['score']} scD={r['d']['score']} "
                  f"$L={r['l']['cost']:.3f} $D={r['d']['cost']:.3f} stepsL={r['l']['steps']} stepsD={r['d']['steps']}")


if __name__ == "__main__":
    main()
