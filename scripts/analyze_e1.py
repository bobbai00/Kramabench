#!/usr/bin/env python3
"""
Three-way analysis for the E1 demand-paging experiment:

  control : DataflowSystemGPT52LatestStats3kD2SmallTableControl (full push)
  lean    : DataflowSystemGPT52LatestStats3kD2Lean3      (rows=3, no pull)
  pull    : DataflowSystemGPT52LatestStats3kD2Lean3Pull  (rows=3 + inspectResult)

Sections: accuracy 3-way, paired cache-aware cost (lean vs control, pull vs
lean, pull vs control), pull-usage stats (calls, args, per-task), and the
pull-cohort accuracy cut (tasks where the agent actually pulled).

Usage: python scripts/analyze_e1.py [--out judgment_runs/<dir>/e1_analysis.md]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = KB_ROOT / "system_scratch"

CONTROL = "DataflowSystemGPT52LatestStats3kD2SmallTableControl"
LEAN = "DataflowSystemGPT52LatestStats3kD2Lean3"
PULL = "DataflowSystemGPT52LatestStats3kD2Lean3Pull"

TH = 0.9
PRIMARY_METRIC = {
    "numeric_exact": "success",
    "string_exact": "success",
    "list_exact": "f1",
    "numeric_approximate": "rae_score",
    "list_approximate": "f1_approximate",
    "string_approximate": "llm_paraphrase",
}
SCORE_KEYS = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def score_of(ev, atype):
    if not ev:
        return 0.0
    k = PRIMARY_METRIC.get(atype or "")
    if k and isinstance(ev.get(k), (int, float)):
        return float(ev[k])
    vals = [float(ev[x]) for x in SCORE_KEYS if isinstance(ev.get(x), (int, float))]
    return max(vals) if vals else 0.0


def collect(sut):
    out = {}
    base = SCRATCH / sut
    if not base.is_dir():
        return out
    for td in sorted(base.iterdir()):
        if not td.is_dir():
            continue
        ev = load(td / "evaluation.json")
        gt = load(td / "ground_truth.json") or {}
        stats = load(td / "stats.json")
        steps_doc = load(td / "react_steps.json")
        steps = steps_doc.get("steps", []) if isinstance(steps_doc, dict) else (steps_doc or [])
        pulls = []
        for s in steps:
            if s.get("role") != "agent":
                continue
            for tc in s.get("toolCalls") or []:
                if tc.get("toolName") == "inspectResult":
                    pulls.append(tc.get("input") or {})
        out[td.name] = {
            "score": score_of(ev, gt.get("answer_type")),
            "stats": stats,
            "pulls": pulls,
        }
    return out


def paired_cost(A, B, tasks):
    agg = {"a": Counter(), "b": Counter()}
    n = 0
    for t in tasks:
        sa, sb = A[t]["stats"], B[t]["stats"]
        if not sa or not sb:
            continue
        n += 1
        for key, side in (("a", sa), ("b", sb)):
            for k in ("cost_usd", "input_tokens", "cached_tokens", "output_tokens", "num_steps"):
                v = side.get(k)
                if isinstance(v, (int, float)):
                    agg[key][k] += v
    return n, agg


def cost_rows(label_a, label_b, n, agg):
    rows = [f"### {label_b} vs {label_a} ({n} paired tasks)", "",
            f"| Measure | {label_a} | {label_b} | Δ |", "| --- | ---: | ---: | ---: |"]
    for k in ("cost_usd", "input_tokens", "cached_tokens", "output_tokens", "num_steps"):
        a, b = agg["a"][k], agg["b"][k]
        d = b - a
        pct = f" ({100 * d / a:+.2f}%)" if a else ""
        if k == "cost_usd":
            rows.append(f"| Cache-aware cost | ${a:.4f} | ${b:.4f} | ${d:+.4f}{pct} |")
        else:
            rows.append(f"| {k} | {a:,.0f} | {b:,.0f} | {d:+,.0f}{pct} |")
    ua = agg["a"]["input_tokens"] - agg["a"]["cached_tokens"]
    ub = agg["b"]["input_tokens"] - agg["b"]["cached_tokens"]
    if ua:
        rows.append(f"| Uncached input | {ua:,.0f} | {ub:,.0f} | {ub - ua:+,.0f} ({100 * (ub - ua) / ua:+.2f}%) |")
    rows.append("")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    C, L, P = collect(CONTROL), collect(LEAN), collect(PULL)
    shared = sorted(set(C) & set(L) & set(P))
    lines = []
    w = lines.append

    w("# E1 demand-paging analysis: full-push vs lean-push vs lean-push+pull")
    w("")
    w(f"Shared tasks: {len(shared)} (control {len(C)}, lean {len(L)}, pull {len(P)})")
    w("")

    # accuracy 3-way
    cp = sum(1 for t in shared if C[t]["score"] >= TH)
    lp = sum(1 for t in shared if L[t]["score"] >= TH)
    pp = sum(1 for t in shared if P[t]["score"] >= TH)
    w("## Accuracy (pass = metric >= 0.9)")
    w("")
    w("| Arm | Passes | Rate |")
    w("| --- | ---: | ---: |")
    for name, v in (("control (full push)", cp), ("lean (rows=3)", lp), ("lean+pull", pp)):
        w(f"| {name} | {v}/{len(shared)} | {100 * v / max(1, len(shared)):.1f}% |")
    w("")
    # flips vs control
    for name, X in (("lean", L), ("pull", P)):
        co = [t for t in shared if C[t]["score"] >= TH and X[t]["score"] < TH]
        xo = [t for t in shared if C[t]["score"] < TH and X[t]["score"] >= TH]
        w(f"{name} vs control flips: control-only {len(co)} {co}; {name}-only {len(xo)} {xo}")
        w("")

    # pull usage
    w("## Pull usage (treatment arm)")
    w("")
    pull_tasks = {t: P[t]["pulls"] for t in P if P[t]["pulls"]}
    total_pulls = sum(len(v) for v in pull_tasks.values())
    w(f"Tasks with >=1 pull: {len(pull_tasks)} / {len(P)}; total pulls: {total_pulls}")
    w("")
    if pull_tasks:
        args_hist = Counter()
        for pulls in pull_tasks.values():
            for p in pulls:
                if p.get("where"):
                    args_hist["where:" + str(p["where"]).split(":")[0]] += 1
                if p.get("columns"):
                    args_hist["columns"] += 1
                if p.get("stats"):
                    args_hist["stats"] += 1
                if p.get("maxRows"):
                    args_hist[f"maxRows"] += 1
                if not any([p.get("where"), p.get("columns"), p.get("stats"), p.get("maxRows")]):
                    args_hist["(bare)"] += 1
        w(f"Argument usage: {dict(args_hist)}")
        w("")
        w("| Task | Pulls | Score lean | Score pull |")
        w("| --- | ---: | ---: | ---: |")
        for t in sorted(pull_tasks):
            w(f"| `{t}` | {len(pull_tasks[t])} | {L.get(t, {}).get('score', 0):.2f} | {P[t]['score']:.2f} |")
        w("")

    # costs
    w("## Paired cache-aware usage")
    w("")
    for la, lb, A, B in (("control", "lean", C, L), ("lean", "pull", L, P), ("control", "pull", C, P)):
        n, agg = paired_cost(A, B, shared)
        lines.extend(cost_rows(la, lb, n, agg))

    text = "\n".join(lines)
    if a.out:
        outp = KB_ROOT / a.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text)
        print(f"[analyze_e1] wrote {outp}")
    else:
        print(text)


if __name__ == "__main__":
    main()
