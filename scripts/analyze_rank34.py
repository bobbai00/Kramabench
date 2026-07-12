#!/usr/bin/env python3
"""
Paired analysis for the rank-3 (fold resolved revisions, DELTA) and rank-4
(probe retirement, LATEST) static-rule experiments.

Per arm pair (control vs treatment, matched except the rule config):
  1. config check      — agent_settings diff across shared tasks
  2. rule activation   — rendered signatures in each agent step's inputMessages
  3. paired accuracy   — answer-type metric >= 0.9 (compare_arms.py convention)
  4. fair paired cost  — stats.json cost_usd + token fields over tasks with
                         usage artifacts in BOTH arms

Usage:
  python scripts/analyze_rank34.py --control <SUT> --treatment <SUT> \
      --rule fold|probe [--out judgment_runs/<dir>/analysis.md]
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = KB_ROOT / "system_scratch"

TH = 0.9
PRIMARY_METRIC = {
    "numeric_exact": "success",
    "string_exact": "success",
    "list_exact": "f1",
    "numeric_approximate": "rae_score",
    "list_approximate": "f1_approximate",
    "string_approximate": "llm_paraphrase",
}

SIGNATURES = {
    "fold": [
        "(folded — superseded revision",
        "(folded superseded attempt",
        "(folded — rejected submission",
    ],
    "probe": ["(probe retired — "],
}

RULE_CONFIG_KEY = {
    "fold": "fold_resolved_revisions_config",
    "probe": "probe_retirement_config",
}


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


SCORE_KEYS = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]


def score_of(ev, atype):
    """compare_arms.py convention: the answer type's metric, else best available."""
    if not ev:
        return 0.0
    k = PRIMARY_METRIC.get(atype or "")
    if k and isinstance(ev.get(k), (int, float)):
        return float(ev[k])
    vals = [float(ev[x]) for x in SCORE_KEYS if isinstance(ev.get(x), (int, float))]
    return max(vals) if vals else 0.0


def task_dirs(sut):
    d = SCRATCH / sut
    if not d.is_dir():
        return {}
    return {t.name: t for t in sorted(d.iterdir()) if t.is_dir()}


def collect(sut):
    out = {}
    for tid, td in task_dirs(sut).items():
        ev = load(td / "evaluation.json")
        stats = load(td / "stats.json")
        steps_doc = load(td / "react_steps.json")
        steps = steps_doc.get("steps", []) if isinstance(steps_doc, dict) else (steps_doc or [])
        cfg = load(td / "config.json")
        gt = load(td / "ground_truth.json") or {}
        out[tid] = {
            "eval": ev,
            "stats": stats,
            "steps": steps,
            "settings": (cfg or {}).get("agent_settings"),
            "atype": gt.get("answer_type"),
        }
    return out


def activation(steps, sigs):
    """(agent steps scanned, steps with >=1 signature, signature count in the
    final agent step's rendered context)."""
    agent_steps = [s for s in steps if s.get("role") == "agent" and s.get("inputMessages")]
    hit_steps = 0
    final_count = 0
    for s in agent_steps:
        text = "\n".join(str(m.get("content", "")) for m in s["inputMessages"])
        n = sum(text.count(sig) for sig in sigs)
        if n > 0:
            hit_steps += 1
        final_count = n  # last agent step wins
    return len(agent_steps), hit_steps, final_count


def fmt_usd(x):
    return f"${x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--rule", required=True, choices=["fold", "probe"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sigs = SIGNATURES[a.rule]
    ckey = RULE_CONFIG_KEY[a.rule]
    C = collect(a.control)
    T = collect(a.treatment)
    shared = sorted(set(C) & set(T))
    lines = []
    w = lines.append

    w(f"# Rank-{'3' if a.rule == 'fold' else '4'} paired analysis: {a.control} vs {a.treatment}")
    w("")
    w(f"Shared task directories: {len(shared)} (control {len(C)}, treatment {len(T)})")
    w("")

    # 1. config check ------------------------------------------------------
    diffs = Counter()
    checked = 0
    for tid in shared:
        cs, ts = C[tid]["settings"], T[tid]["settings"]
        if not cs or not ts:
            continue
        checked += 1
        for k in sorted(set(cs) | set(ts)):
            if cs.get(k) != ts.get(k):
                diffs[k] += 1
    w("## Configuration check")
    w("")
    w(f"Tasks with configs in both arms: {checked}")
    expected = {ckey}
    unexpected = {k: v for k, v in diffs.items() if k not in expected}
    w(f"Settings keys differing: {dict(diffs) or '(none)'}")
    w(f"Unexpected diffs (should be empty): {unexpected or '(none)'}")
    w("")

    # 2. activation ---------------------------------------------------------
    w("## Rule activation (treatment arm)")
    w("")
    act_rows = []
    tasks_hit, total_final = 0, 0
    for tid in sorted(T):
        n_steps, hit_steps, final_n = activation(T[tid]["steps"], sigs)
        if hit_steps > 0:
            tasks_hit += 1
            total_final += final_n
            act_rows.append((tid, n_steps, hit_steps, final_n))
    w(f"Tasks with >=1 rendered activation: {tasks_hit} / {len(T)}")
    w(f"Sum of final-step signature counts: {total_final}")
    w("")
    if act_rows:
        w("| Task | Agent steps | Steps w/ signature | Final-step signatures |")
        w("| --- | ---: | ---: | ---: |")
        for tid, n, h, fn in act_rows:
            w(f"| `{tid}` | {n} | {h} | {fn} |")
        w("")
    # Control must never show the signature.
    leak = [tid for tid in C if activation(C[tid]["steps"], sigs)[1] > 0]
    w(f"Control-arm signature leak (must be []): {leak or '[]'}")
    w("")

    # 3. accuracy -----------------------------------------------------------
    w("## Accuracy (pass = answer-type metric >= 0.9)")
    w("")
    both_pass = c_only = t_only = both_fail = 0
    flips = []
    for tid in shared:
        cp = score_of(C[tid]["eval"], C[tid]["atype"]) >= TH
        tp = score_of(T[tid]["eval"], T[tid]["atype"]) >= TH
        if cp and tp:
            both_pass += 1
        elif cp:
            c_only += 1
            flips.append((tid, "control-only"))
        elif tp:
            t_only += 1
            flips.append((tid, "treatment-only"))
        else:
            both_fail += 1
    w("| Outcome | Tasks |")
    w("| --- | ---: |")
    w(f"| Both pass | {both_pass} |")
    w(f"| Control only | {c_only} |")
    w(f"| Treatment only | {t_only} |")
    w(f"| Both fail | {both_fail} |")
    w("")
    cpass, tpass = both_pass + c_only, both_pass + t_only
    w(f"Passes: control {cpass}/{len(shared)} ({100 * cpass / max(1, len(shared)):.1f}%), "
      f"treatment {tpass}/{len(shared)} ({100 * tpass / max(1, len(shared)):.1f}%)")
    w("")
    if flips:
        w("Accuracy divergences (independent trajectories — attribution needs the trace):")
        w("")
        for tid, d in flips:
            w(f"- `{tid}`: {d}")
        w("")

    # 4. fair paired cost ----------------------------------------------------
    w("## Fair paired usage (tasks with stats.json in both arms)")
    w("")
    keys = ["cost_usd", "input_tokens", "cached_tokens", "output_tokens", "total_tokens", "num_steps"]
    agg = {arm: Counter() for arm in ("control", "treatment")}
    paired = 0
    for tid in shared:
        cs, ts = C[tid]["stats"], T[tid]["stats"]
        if not cs or not ts:
            continue
        paired += 1
        for arm, st in (("control", cs), ("treatment", ts)):
            for k in keys:
                v = st.get(k)
                if isinstance(v, (int, float)):
                    agg[arm][k] += v

    w(f"Paired usage tasks: {paired}")
    w("")
    w("| Measure | Control | Treatment | Treatment − Control |")
    w("| --- | ---: | ---: | ---: |")
    for k in keys:
        c, t = agg["control"][k], agg["treatment"][k]
        d = t - c
        if k == "cost_usd":
            pct = f" ({100 * d / c:+.2f}%)" if c else ""
            w(f"| Cache-aware cost | {fmt_usd(c)} | {fmt_usd(t)} | {fmt_usd(d)}{pct} |")
        else:
            pct = f" ({100 * d / c:+.2f}%)" if c else ""
            w(f"| {k} | {c:,.0f} | {t:,.0f} | {d:+,.0f}{pct} |")
    c_uncached = agg["control"]["input_tokens"] - agg["control"]["cached_tokens"]
    t_uncached = agg["treatment"]["input_tokens"] - agg["treatment"]["cached_tokens"]
    if c_uncached:
        w(f"| Uncached input | {c_uncached:,.0f} | {t_uncached:,.0f} | {t_uncached - c_uncached:+,.0f} "
          f"({100 * (t_uncached - c_uncached) / c_uncached:+.2f}%) |")
    for arm in ("control", "treatment"):
        i, ci = agg[arm]["input_tokens"], agg[arm]["cached_tokens"]
        if i:
            w(f"| Cache hit ({arm}) | | | {100 * ci / i:.1f}% |")
    w("")

    text = "\n".join(lines)
    if a.out:
        outp = KB_ROOT / a.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text)
        print(f"[analyze_rank34] wrote {outp}")
    else:
        print(text)


if __name__ == "__main__":
    main()
