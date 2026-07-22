#!/usr/bin/env python3
"""
M1-v0 — plan/step coverage from CODE (deterministic, no LLM, no re-exec).

Each gold subtask `step` describes one operation of the correct pipeline. This
metric asks, per step: does the agent's pipeline CODE reference the distinctive
tokens of that step — the column names, string/numeric literals, and files that
a correct implementation of the step must touch? It survives fusion (a fused
op's code still contains `Case_excluded`) and checks correctness at the token
level (the specific column/value, not just "a filter exists" — that's what made
the built-in judge saturate).

score(subtask) = |distinctive tokens present in agent code| / |distinctive tokens|
coverage(task) = mean over subtasks that have >=1 distinctive token

Run: .venv/bin/python scripts/m1_plan_coverage.py --arms SUT... [--tasks ID...]
"""
import argparse, json, re, glob, importlib.util
from pathlib import Path

KB = Path(__file__).resolve().parent.parent

STOP = set("""the a an and or of to for in on at is are be as by with from into that this these those
data file files row rows column columns value values table tables step steps result results answer
compute calculate find get read load use using number count list set each all any per over between
where which what study contains contain include included has have with total based only then them their
type types name names version convert converted linearized linear scale sheet subset filtered create""".split())

OP_KEYWORDS = ["median", "mean", "sum", "dropna", "drop_duplicates", "duplicated",
               "merge", "join", "interpolate", "groupby", "round", "cumsum",
               "nunique", "unique", "pivot", "melt", "concat", "sort_values",
               "argmax", "idxmax", "idxmin", "value_counts", "std", "var", "corr"]

def gold_signals(step):
    sigs = set()
    for a, b in re.findall(r"'([^']+)'|\"([^\"]+)\"", step):
        v = (a or b).strip()
        if v and len(v) <= 40:
            sigs.add(("lit", v))
    for m in re.findall(r"[\w.\-]+\.(?:csv|xlsx|xls|txt|json|html|dat|tsv|parquet)", step):
        sigs.add(("file", m))
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", step):
        if ("_" in tok or any(c.isupper() for c in tok[1:])) and tok.lower() not in STOP and len(tok) > 2:
            sigs.add(("id", tok))
    for num in re.findall(r"(?<![\w.])\d{2,}(?:\.\d+)?", step):  # 2+ digit numbers (thresholds)
        sigs.add(("num", num))
    for kw in OP_KEYWORDS:
        if re.search(rf"\b{kw}\b", step.lower()):
            sigs.add(("op", kw))
    return sigs

def present(sig, code, code_l):
    kind, v = sig
    if kind in ("id", "file"):
        return v in code  # case-sensitive: column/file names must match exactly
    if kind == "lit":
        return v.lower() in code_l  # values may be compared lower/upper-cased in code
    if kind == "op":
        return v in code_l
    if kind == "num":
        return v in code
    return False

def agent_code(sut, task):
    p = KB / "system_scratch" / sut / task / "react_steps.json"
    if not p.exists():
        return None
    order, code = [], {}
    for s in json.load(open(p)).get("steps", []):
        for tc in (s.get("toolCalls") or []):
            nm = tc.get("toolName", ""); inp = tc.get("input") or {}; oid = inp.get("operatorId")
            if nm == "createOrModifyOperator" and oid:
                if oid not in code: order.append(oid)
                code[oid] = inp.get("code", "")
            elif nm == "deleteOperator" and oid:
                code.pop(oid, None); order = [o for o in order if o != oid]
    return "\n".join(code[o] for o in order if o in code)

def score_task(sut, task_id, task_def):
    code = agent_code(sut, task_id)
    if code is None:
        return None
    code_l = code.lower()
    per = []
    for st in task_def.get("subtasks", []):
        sigs = gold_signals(st.get("step", ""))
        if not sigs:
            continue
        hit = sum(1 for s in sigs if present(s, code, code_l))
        per.append((st.get("id"), round(hit / len(sigs), 3), len(sigs)))
    cov = sum(x[1] for x in per) / len(per) if per else None
    return dict(coverage=cov, per=per)

def load_workload():
    tasks = {}
    for f in glob.glob(str(KB / "workload" / "*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")): continue
        try: ts = json.load(open(f))
        except Exception: continue
        if isinstance(ts, list):
            for t in ts:
                if isinstance(t, dict) and t.get("id"): tasks[t["id"]] = t
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    W = load_workload()
    tasks = a.tasks or sorted(t for t in W if t.startswith("biomedical"))
    spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
    kb = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)
    scores = {arm: (kb.answer_scores(arm) if True else {}) for arm in a.arms}
    short = lambda s: s.replace("DataflowSystemGPT52", "52.").replace("DataflowSystemGPT5Mini", "mini.").replace("ProbePrompt", "·P")
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
                anss = "P" if (ans is not None and ans >= 0.9) else ("f" if ans is not None else "?")
                row += f"{anss+' '+format(r['coverage'],'.2f'):>24s} "
            if a.verbose and r: print(f"   [{short(arm)}] {t}: {r['per']}")
        print(row)
    print("-" * (22 + 25 * len(a.arms)))
    mrow = f"{'MEAN M1 coverage':22s} "
    for arm in a.arms:
        m = sum(agg[arm]) / len(agg[arm]) if agg[arm] else 0
        mrow += f"{m:>24.3f} "
    print(mrow)

if __name__ == "__main__":
    main()
