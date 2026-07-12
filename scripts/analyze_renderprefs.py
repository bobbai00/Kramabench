#!/usr/bin/env python3
"""
Three-way analysis for the write-time render-prefs experiment (DELTA 5k):

  treatment : DataflowSystemGPT52DeltaStats5kD2RenderPrefs (agent-declared
              outputSummary/showOutputStatistics + investigation-op cleanup)
  stats     : DataflowSystemGPT52DeltaStats5kD2   (stats always on)
  schema    : DataflowSystemGPT52Delta5kSchemaOnly (stats never on)

Sections: accuracy, paired cache-aware cost, knob-usage stats (per-call
declarations, minimal/stats-off rates, deleteOperator usage), and the label
harvest (operator role at declaration time → chosen level).

NOTE: the two comparator runs are July-6 vintage (pre small-table renderer
rule); treatment is current code. Interpret tight deltas with that caveat.

Usage: python scripts/analyze_renderprefs.py [--out judgment_runs/<dir>/x.md]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = KB_ROOT / "system_scratch"

TREAT = "DataflowSystemGPT52DeltaStats5kD2RenderPrefs"
STATS = "DataflowSystemGPT52DeltaStats5kD2"
SCHEMA = "DataflowSystemGPT52Delta5kSchemaOnly"

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
        doc = load(td / "react_steps.json")
        steps = doc.get("steps", []) if isinstance(doc, dict) else (doc or [])
        prefs = []
        deletes = 0
        creates = 0
        for s in steps:
            if s.get("role") != "agent":
                continue
            for tc in s.get("toolCalls") or []:
                inp = tc.get("input") or {}
                if tc.get("toolName") == "deleteOperator":
                    deletes += 1
                if tc.get("toolName") == "createOrModifyOperator":
                    creates += 1
                    if "outputSummary" in inp or "showOutputStatistics" in inp:
                        prefs.append(inp)
        out[td.name] = {
            "score": score_of(ev, gt.get("answer_type")),
            "stats": stats,
            "prefs": prefs,
            "creates": creates,
            "deletes": deletes,
        }
    return out


def cost_block(w, label_a, label_b, A, B, shared):
    agg = {"a": Counter(), "b": Counter()}
    n = 0
    for t in shared:
        sa, sb = A[t]["stats"], B[t]["stats"]
        if not sa or not sb:
            continue
        n += 1
        for key, side in (("a", sa), ("b", sb)):
            for k in ("cost_usd", "input_tokens", "cached_tokens", "output_tokens", "num_steps"):
                v = side.get(k)
                if isinstance(v, (int, float)):
                    agg[key][k] += v
    w(f"### {label_b} vs {label_a} ({n} paired tasks)")
    w("")
    w(f"| Measure | {label_a} | {label_b} | Δ |")
    w("| --- | ---: | ---: | ---: |")
    for k in ("cost_usd", "input_tokens", "cached_tokens", "output_tokens", "num_steps"):
        a, b = agg["a"][k], agg["b"][k]
        d = b - a
        pct = f" ({100 * d / a:+.2f}%)" if a else ""
        if k == "cost_usd":
            w(f"| Cache-aware cost | ${a:.4f} | ${b:.4f} | ${d:+.4f}{pct} |")
        else:
            w(f"| {k} | {a:,.0f} | {b:,.0f} | {d:+,.0f}{pct} |")
    ua = agg["a"]["input_tokens"] - agg["a"]["cached_tokens"]
    ub = agg["b"]["input_tokens"] - agg["b"]["cached_tokens"]
    if ua:
        w(f"| Uncached input | {ua:,.0f} | {ub:,.0f} | {ub - ua:+,.0f} ({100 * (ub - ua) / ua:+.2f}%) |")
    w("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    T, S, O = collect(TREAT), collect(STATS), collect(SCHEMA)
    shared = sorted(set(T) & set(S) & set(O))
    lines = []
    w = lines.append

    w("# Write-time render prefs (DELTA 5k): treatment vs stats-on vs schema-only")
    w("")
    w(f"Shared tasks: {len(shared)}. Comparators are July-6 vintage (code caveat).")
    w("")

    w("## Accuracy (pass = metric >= 0.9)")
    w("")
    w("| Arm | Passes | Rate |")
    w("| --- | ---: | ---: |")
    for name, X in (("stats-on (Delta5kD2)", S), ("schema-only", O), ("render-prefs", T)):
        p = sum(1 for t in shared if X[t]["score"] >= TH)
        w(f"| {name} | {p}/{len(shared)} | {100 * p / max(1, len(shared)):.1f}% |")
    w("")
    for name, X in (("stats-on", S), ("schema-only", O)):
        xo = [t for t in shared if X[t]["score"] >= TH and T[t]["score"] < TH]
        to = [t for t in shared if X[t]["score"] < TH and T[t]["score"] >= TH]
        w(f"vs {name}: {name}-only {len(xo)} {xo}; treatment-only {len(to)} {to}")
        w("")

    w("## Knob usage (treatment)")
    w("")
    used = {t: v for t, v in T.items() if v["prefs"]}
    total_creates = sum(v["creates"] for v in T.values())
    total_prefs = sum(len(v["prefs"]) for v in T.values())
    w(f"Tasks with >=1 declaration: {len(used)}/{len(T)}; "
      f"pref-bearing calls: {total_prefs}/{total_creates} create/modify calls "
      f"({100 * total_prefs / max(1, total_creates):.0f}%)")
    hist = Counter()
    for v in T.values():
        for p in v["prefs"]:
            os_ = p.get("outputSummary")
            st_ = p.get("showOutputStatistics")
            hist[f"outputSummary={os_}"] += os_ is not None
            hist[f"showOutputStatistics={st_}"] += st_ is not None
    w(f"Declaration histogram: {dict(hist)}")
    deletes = sum(v["deletes"] for v in T.values())
    del_stats = sum(v["deletes"] for v in S.values())
    w(f"deleteOperator calls: treatment {deletes} vs stats-on comparator {del_stats}")
    w("")

    w("## Paired cache-aware usage")
    w("")
    cost_block(w, "stats-on", "render-prefs", S, T, shared)
    cost_block(w, "schema-only", "render-prefs", O, T, shared)

    text = "\n".join(lines)
    if a.out:
        outp = KB_ROOT / a.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text)
        print(f"[analyze_renderprefs] wrote {outp}")
    else:
        print(text)


if __name__ == "__main__":
    main()
