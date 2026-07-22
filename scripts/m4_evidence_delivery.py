#!/usr/bin/env python3
"""
M4 — evidence-delivery coverage (what the KNOB controls, deterministic).

Knobs (rows/stats/history) change what the agent SEES, not what it writes.
So instead of checking the agent's code (M1) or outputs (M2), M4 checks the
agent's RENDERED OBSERVATIONS: for each gold subtask step, were its key facts
delivered on screen? Focus on VALUE signals (literals, numeric thresholds,
data-value identifiers) — a data value appears in context only if it was
rendered (the agent doesn't hardcode it), so its presence measures delivery.

This isolates the knob's actual job from the agent's stochastic use of it.

Run: .venv/bin/python scripts/m4_evidence_delivery.py --arms SUT... [--tasks ..]
"""
import argparse, json, re, glob, importlib.util
from pathlib import Path
KB = Path(__file__).resolve().parent.parent
m1 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("m1", KB / "scripts/m1_plan_coverage.py"))
importlib.util.spec_from_file_location("m1", KB / "scripts/m1_plan_coverage.py").loader.exec_module(m1)

SOURCE_ONLY = False  # set by --source-only: keep only loader-op renders (raw
# source-data the knob actually controls), excluding the agent's derived outputs
# — avoids the circularity where a correct arm's own computed answer/intermediates
# get counted as "delivered".

def rendered_context(sut, task):
    """Union of the observation blocks the agent SAW. With SOURCE_ONLY, restrict
    to blocks from loader operators (code reads a file) — the raw input renders."""
    p = KB / "system_scratch" / sut / task / "react_steps.json"
    if not p.exists():
        return None
    steps = json.load(open(p)).get("steps", [])
    loader = {}
    if SOURCE_ONLY:
        for s in steps:
            for tc in (s.get("toolCalls") or []):
                if tc.get("toolName") == "createOrModifyOperator":
                    inp = tc.get("input") or {}
                    loader[inp.get("operatorId")] = bool(
                        re.search(r"read_csv|read_excel|read_fwf|read_html|ExcelFile|open\(|glob\.",
                                  inp.get("code", "") or ""))
    seen = set(); chunks = []
    for s in steps:
        for m in (s.get("inputMessages") or []):
            c = m.get("content", "")
            for block in re.split(r"## Agent Event \d+", c):
                if not ("result:" in block or "Output Table" in block or "Column" in block):
                    continue
                if SOURCE_ONLY:
                    mm = re.search(r"operator\s+(\S+)\s+(added|updated)", block)
                    if not (mm and loader.get(mm.group(1))):
                        continue
                h = hash(block)
                if h not in seen:
                    seen.add(h); chunks.append(block)
    return "\n".join(chunks)

def score_task(sut, task_id, task_def):
    ctx = rendered_context(sut, task_id)
    if ctx is None:
        return None
    ctx_l = ctx.lower()
    per = []
    for st in task_def.get("subtasks", []):
        sigs = m1.gold_signals(st.get("step", ""))
        # value-only signals: literals, numeric thresholds, and the ANSWER values
        vsigs = {s for s in sigs if s[0] in ("lit", "num")}
        # also fold in the gold intermediate ANSWER values (delivered iff rendered)
        ans = st.get("answer")
        for v in (ans if isinstance(ans, list) else [ans]):
            if isinstance(v, (int, float)):
                vsigs.add(("num", str(v)))
            elif isinstance(v, str) and 2 < len(v) <= 40:
                vsigs.add(("lit", v))
        if not vsigs:
            continue
        hit = sum(1 for s in vsigs if m1.present(s, ctx, ctx_l))
        per.append((st.get("id"), round(hit / len(vsigs), 3), len(vsigs)))
    cov = sum(x[1] for x in per) / len(per) if per else None
    return dict(coverage=cov, per=per)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--source-only", action="store_true")
    a = ap.parse_args()
    global SOURCE_ONLY
    SOURCE_ONLY = a.source_only
    W = m1.load_workload()
    tasks = a.tasks or sorted(t for t in W if t.startswith("biomedical"))
    spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
    kb = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)
    scores = {arm: kb.answer_scores(arm) for arm in a.arms}
    short = lambda s: s.replace("DataflowSystemGPT52", "52.").replace("ProbePrompt", "·P")
    print(f"{'task':22s} " + " ".join(f"{short(arm):>24s}" for arm in a.arms))
    print("-" * (22 + 25 * len(a.arms)))
    agg = {arm: [] for arm in a.arms}
    for t in tasks:
        if t not in W: continue
        row = f"{t:22s} "
        for arm in a.arms:
            r = score_task(arm, t, W[t]); ans = scores[arm].get(t)
            if r is None or r["coverage"] is None:
                row += f"{'—':>24s} "
            else:
                agg[arm].append(r["coverage"])
                anss = "P" if (ans is not None and ans >= 0.9) else "f"
                row += f"{anss+' '+format(r['coverage'],'.2f'):>24s} "
        print(row)
    print("-" * (22 + 25 * len(a.arms)))
    mrow = f"{'MEAN M4 delivery':22s} "
    for arm in a.arms:
        m = sum(agg[arm]) / len(agg[arm]) if agg[arm] else 0
        mrow += f"{m:>24.3f} "
    print(mrow)

if __name__ == "__main__":
    main()
