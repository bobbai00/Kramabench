#!/usr/bin/env python3
"""
M2 — intermediate-answer coverage (process-aware, deterministic, retroactive).

For each task, KramaBench ships gold *subtask* answers = the correct
intermediate results (e.g. the 12 kept serous case IDs, the linearized value
list, the final scalar). This metric re-executes the agent's final DAG in a
sandbox and asks, per gold intermediate: did ANY operator output actually
compute it? Coverage = mean best-match over the task's intermediates.

Match by subtask answer_type:
  numeric_exact       -> 1.0 if any single cell within 1e-4 rel tol
  numeric_approximate -> best 1/(1+RAE) over cells
  string_exact        -> 1.0 if any cell/col value equals (normalized)
  string_approximate  -> normalized containment (no LLM in the loop; v0)
  list_exact          -> best set-F1 of gold list vs any op column
  list_approximate    -> best set-F1 (fuzzy numeric/normalized string)

Deterministic (no LLM). Ops that error on re-exec -> that op unavailable
(downstream may still run); logged. Run from repo root:
  .venv/bin/python scripts/m2_intermediate_coverage.py --arms SUT... --tasks ID...
"""
import argparse, json, os, re, sys, glob, io, contextlib
import multiprocessing as mp
from pathlib import Path

KB = Path(__file__).resolve().parent.parent

# ---------- load gold subtasks ----------
def load_workload():
    tasks = {}
    for f in glob.glob(str(KB / "workload" / "*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")):
            continue
        try:
            ts = json.load(open(f))
        except Exception:
            continue
        if isinstance(ts, list):
            for t in ts:
                if isinstance(t, dict) and t.get("id"):
                    tasks[t["id"]] = t
    return tasks

# ---------- reconstruct the final operator DAG from a trace ----------
def final_ops(sut, task):
    p = KB / "system_scratch" / sut / task / "react_steps.json"
    if not p.exists():
        return None
    order, code = [], {}
    for s in json.load(open(p)).get("steps", []):
        for tc in (s.get("toolCalls") or []):
            nm = tc.get("toolName", "")
            inp = tc.get("input") or {}
            oid = inp.get("operatorId")
            if nm == "createOrModifyOperator" and oid:
                if oid not in code:
                    order.append(oid)
                code[oid] = inp.get("code", "")
            elif nm == "deleteOperator" and oid:
                code.pop(oid, None)
                order = [o for o in order if o != oid]
    return [(o, code[o]) for o in order if o in code]

# ---------- execute the DAG, return {op_id: DataFrame} ----------
def exec_dag(ops):
    import pandas as pd, numpy as np
    outputs, failed = {}, {}
    idset = {o for o, _ in ops}
    # resolve dependencies: process(argnames) -> upstream op ids
    def deps(src):
        m = re.search(r"def\s+process\s*\(([^)]*)\)", src)
        if not m:
            return []  # load(): no upstream
        args = [a.split("=")[0].split(":")[0].strip() for a in m.group(1).split(",") if a.strip()]
        return [a for a in args if a in idset]
    remaining = list(ops)
    guard = 0
    while remaining and guard < 500:
        guard += 1
        progressed = False
        for o, src in list(remaining):
            d = deps(src)
            if any(x not in outputs for x in d):
                if any(x in failed for x in d):  # upstream dead -> can't run
                    failed[o] = f"upstream failed: {d}"
                    remaining.remove((o, src))
                    progressed = True
                continue
            g = {"pd": pd, "np": np, "__builtins__": __builtins__}
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(src, g)
                    if "process" in g:
                        res = g["process"](*[outputs[x] for x in d])
                    elif "load" in g:
                        res = g["load"]()
                    else:
                        raise RuntimeError("no load()/process()")
                if not isinstance(res, pd.DataFrame):
                    res = pd.DataFrame(res if hasattr(res, "__len__") and not isinstance(res, str) else [res])
                outputs[o] = res
            except Exception as e:
                failed[o] = f"{type(e).__name__}: {str(e)[:80]}"
            remaining.remove((o, src))
            progressed = True
        if not progressed:
            for o, src in remaining:
                failed[o] = "unresolved dep cycle/missing"
            break
    return outputs, failed

# ---------- matching ----------
def as_float(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").replace("%", "").strip()
        return float(x)
    except Exception:
        return None

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def num_match(a, b, tol=1e-4):
    fa, fb = as_float(a), as_float(b)
    if fa is None or fb is None:
        return False
    if fb == 0:
        return abs(fa) < tol
    return abs(fa - fb) / abs(fb) < tol

def col_value_lists(df):
    import pandas as pd
    out = []
    for c in df.columns:
        vals = df[c].dropna().tolist()
        if vals:
            out.append(vals)
    return out

def best_scalar(gold, outputs, approx=False):
    best = 0.0
    for df in outputs.values():
        for c in df.columns:
            for v in df[c].dropna().tolist():
                if approx:
                    fg, fv = as_float(gold), as_float(v)
                    if fg is not None and fv is not None and fg != 0:
                        best = max(best, 1.0 / (1.0 + abs(fv - fg) / abs(fg)))
                else:
                    if num_match(v, gold) or norm(v) == norm(gold):
                        return 1.0
    return best

def set_f1(gold_list, col, approx=False):
    gold = list(gold_list)
    # numeric branch
    gnum = [as_float(x) for x in gold]
    if all(x is not None for x in gnum):
        cnum = [as_float(x) for x in col if as_float(x) is not None]
        if not cnum:
            return 0.0
        matched = 0
        used = [False] * len(cnum)
        for gv in gnum:
            for i, cv in enumerate(cnum):
                if not used[i] and (abs(cv - gv) / (abs(gv) if gv else 1) < (1e-3 if approx else 1e-4)):
                    used[i] = True; matched += 1; break
        prec = matched / len(cnum); rec = matched / len(gnum)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # string branch
    gset = {norm(x) for x in gold}
    cset = {norm(x) for x in col}
    if not cset:
        return 0.0
    inter = len(gset & cset)
    prec = inter / len(cset); rec = inter / len(gset)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def best_list(gold, outputs, approx=False):
    best = 0.0
    for df in outputs.values():
        for col in col_value_lists(df):
            best = max(best, set_f1(gold, col, approx))
    return best

def match_intermediate(answer, atype, outputs):
    if not outputs:
        return 0.0
    if atype in ("numeric_exact",):
        return best_scalar(answer, outputs, approx=False)
    if atype in ("numeric_approximate",):
        return best_scalar(answer, outputs, approx=True)
    if atype in ("string_exact", "string_approximate"):
        gold = answer if isinstance(answer, list) else [answer]
        return max((best_scalar(g, outputs, approx=False) for g in gold), default=0.0)
    if atype in ("list_exact",):
        gold = answer if isinstance(answer, list) else [answer]
        return best_list(gold, outputs, approx=False)
    if atype in ("list_approximate",):
        gold = answer if isinstance(answer, list) else [answer]
        return best_list(gold, outputs, approx=True)
    return 0.0

# ---------- per (arm,task) ----------
def score_task(sut, task_id, task_def):
    ops = final_ops(sut, task_id)
    if ops is None:
        return None
    outputs, failed = exec_dag(ops)
    subs = task_def.get("subtasks", [])
    per = []
    for st in subs:
        sc = match_intermediate(st.get("answer"), st.get("answer_type", "string_exact"), outputs)
        per.append((st.get("id"), st.get("answer_type"), round(sc, 3)))
    cov = sum(x[2] for x in per) / len(per) if per else None
    return dict(coverage=cov, n_sub=len(subs), n_ops=len(ops),
                n_ok=len(outputs), n_failed=len(failed), per=per)

# ---------- per-task timeout guard (re-executing agent code can hang) ----------
def _worker(q, sut, task_id, task_def):
    try:
        q.put(score_task(sut, task_id, task_def))
    except Exception as e:
        q.put({"error": f"{type(e).__name__}: {str(e)[:80]}"})

def score_task_timeout(sut, task_id, task_def, timeout=25):
    q = mp.Queue()
    p = mp.Process(target=_worker, args=(q, sut, task_id, task_def), daemon=True)
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return {"timeout": True, "coverage": None}
    try:
        return q.get_nowait()
    except Exception:
        return {"error": "no result", "coverage": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", help="task ids (default: all biomedical)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    W = load_workload()
    tasks = a.tasks or sorted(t for t in W if t.startswith("biomedical"))
    # load answer scores for the pass/fail column
    spec_scores = {}
    import importlib.util
    spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
    kb = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)
    for arm in a.arms:
        try: spec_scores[arm] = kb.answer_scores(arm)
        except Exception: spec_scores[arm] = {}

    short = lambda s: s.replace("DataflowSystemGPT52", "52.").replace("DataflowSystemGPT5Mini", "mini.").replace("ProbePrompt", "·P")
    print(f"{'task':22s} " + " ".join(f"{short(arm):>26s}" for arm in a.arms))
    print(f"{'':22s} " + " ".join(f"{'ans / M2-coverage':>26s}" for _ in a.arms))
    print("-" * (22 + 27 * len(a.arms)))
    agg = {arm: [] for arm in a.arms}
    for t in tasks:
        if t not in W: continue
        row = f"{t:22s} "
        for arm in a.arms:
            print(f"  running {t} / {short(arm)} ...", file=sys.stderr, flush=True)
            r = score_task_timeout(arm, t, W[t])
            ans = spec_scores[arm].get(t)
            if r is None or r.get("coverage") is None:
                tag = "TIMEOUT" if r and r.get("timeout") else ("ERR" if r and r.get("error") else "—")
                row += f"{tag:>26s} "
            else:
                agg[arm].append(r["coverage"])
                anss = "P" if (ans is not None and ans >= 0.9) else ("f" if ans is not None else "?")
                cell = f"{anss} {r['coverage']:.2f} ({r['n_ok']}/{r['n_ops']}op)"
                row += f"{cell:>26s} "
            if a.verbose and r:
                print(f"    [{short(arm)}] {t}: cov={r['coverage']} per={r['per']} failed={r['n_failed']}")
        print(row)
    print("-" * (22 + 27 * len(a.arms)))
    def passrate(arm):
        v = [spec_scores[arm].get(t) for t in tasks if t in W]
        v = [x for x in v if x is not None]
        return sum(1 for x in v if x >= 0.9) / len(v) if v else 0
    mrow = f"{'MEAN M2 coverage':22s} "
    for arm in a.arms:
        m = sum(agg[arm]) / len(agg[arm]) if agg[arm] else 0
        mrow += f"{m:>26.3f} "
    print(mrow)
    prow = f"{'answer pass-rate':22s} "
    for arm in a.arms:
        prow += f"{passrate(arm):>26.1%} "
    print(prow)

if __name__ == "__main__":
    main()
