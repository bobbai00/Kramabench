#!/usr/bin/env python
"""
Behavioral-insight metrics from agent traces (react_steps.json) — quantify HOW the
agent works, not just whether it succeeds. For each arm, aggregated over tasks:

  operators        create / update(re-edit) / delete counts, churn (edits per op)
  debugging        errored edits ("debug attempts"), distinct ops that errored,
                   debug steps (agent steps containing an error)
  recovery         trial-error-FIXED ops (errored then re-edited clean, in place)
                   vs errored-ABANDONED ops (errored, never recovered in place ->
                   typically deleted or replaced by a fresh op = the thrash signature)

Errors are read from toolResults.isError (the agent's [ERROR] outputs). Split by
task outcome (passed/failed) so you can see e.g. "failed tasks debug 3x more".

Usage:
  python bobflow_context_learning/analyze/behavior.py --arms SUT1 [SUT2 ...] \
      [--subset path.json] [--labels a b ...] [--by-task]
"""
import argparse
import json
import os
import statistics as st
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "system_scratch")
TH = 0.9
A2M = {"numeric_exact": "success", "numeric_approximate": "rae_score", "string_exact": "success",
       "string_approximate": "llm_paraphrase", "list_exact": "f1", "list_approximate": "f1_approximate"}
SK = ["success", "rae_score", "f1", "f1_approximate", "llm_paraphrase"]
EDIT = "createOrModifyOperator"
DELETE = "deleteOperator"


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def _passed(ev, at):
    if ev is None:
        return None
    k = A2M.get(at)
    v = ev.get(k) if k else None
    if not isinstance(v, (int, float)):
        cands = [ev[x] for x in SK if isinstance(ev.get(x), (int, float))]
        v = max(cands) if cands else None
    return None if v is None else (float(v) >= TH)


def trace_behavior(task_dir):
    """Per-op edit/error timeline -> behavioral counts for one trace."""
    rs = _load(os.path.join(task_dir, "react_steps.json")) or {}
    steps = rs.get("steps", []) if isinstance(rs, dict) else (rs if isinstance(rs, list) else [])
    edits = OrderedDict()   # op -> [isError per createOrModify edit, in order]
    deleted = set()
    agent_steps = 0
    debug_steps = 0         # agent steps containing >=1 errored toolResult
    for s in steps:
        if s.get("role") != "agent":
            continue
        agent_steps += 1
        step_has_error = False
        for tc, tr in zip(s.get("toolCalls") or [], s.get("toolResults") or []):
            name = tc.get("toolName")
            op = (tc.get("input") or {}).get("operatorId")
            err = bool(tr.get("isError"))
            if name == EDIT and op:
                edits.setdefault(op, []).append(1 if err else 0)
            elif name == DELETE and op:
                deleted.add(op)
            if err:
                step_has_error = True
        if step_has_error:
            debug_steps += 1

    n_create = len(edits)                                   # distinct ops created/modified
    total_edits = sum(len(v) for v in edits.values())
    n_update = total_edits - n_create                       # re-edits (updates beyond first)
    n_errored_edits = sum(sum(v) for v in edits.values())   # debug attempts (failed edits)
    ops_errored = [op for op, v in edits.items() if any(v)]

    def fixed_in_place(v):  # an errored edit (1) followed later by a clean edit (0)
        seen_err = False
        for x in v:
            if x:
                seen_err = True
            elif seen_err:
                return True
        return False

    n_trial_error_fixed = sum(1 for op in ops_errored if fixed_in_place(edits[op]))
    n_errored_abandoned = len(ops_errored) - n_trial_error_fixed
    churn = {op: len(v) for op, v in edits.items()}
    return {
        "agent_steps": agent_steps,
        "ops_created": n_create,
        "ops_updated": n_update,
        "ops_deleted": len(deleted),
        "total_edits": total_edits,
        "max_edits_per_op": max(churn.values()) if churn else 0,
        "mean_edits_per_op": (total_edits / n_create) if n_create else 0.0,
        "errored_edits": n_errored_edits,          # debugging attempts
        "ops_errored": len(ops_errored),
        "debug_steps": debug_steps,
        "trial_error_fixed": n_trial_error_fixed,  # recovered in place
        "errored_abandoned": n_errored_abandoned,  # thrash: errored, never fixed in place
    }


KEYS = ["agent_steps", "ops_created", "ops_updated", "ops_deleted", "max_edits_per_op",
        "mean_edits_per_op", "errored_edits", "ops_errored", "trial_error_fixed",
        "errored_abandoned", "debug_steps"]


def collect(arm, tasks):
    base = os.path.join(SCRATCH, arm)
    rows = []
    for t in tasks:
        d = os.path.join(base, t)
        if not os.path.isdir(d):
            continue
        ev = _load(os.path.join(d, "evaluation.json"))
        at = (_load(os.path.join(d, "ground_truth.json")) or {}).get("answer_type")
        b = trace_behavior(d)
        b["task"] = t
        b["passed"] = _passed(ev, at)
        rows.append(b)
    return rows


def agg(rows, keys=KEYS):
    return {k: (sum(r[k] for r in rows) / len(rows) if rows else 0.0) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--by-task", action="store_true")
    args = ap.parse_args()
    labels = args.labels or [a.replace("DataflowSystemGPT54", "").replace("SchemaConverge", "")[:12] for a in args.arms]

    for lb, arm in zip(labels, args.arms):
        base = os.path.join(SCRATCH, arm)
        if args.subset:
            tasks = json.load(open(args.subset))
        else:
            tasks = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith("_")) if os.path.isdir(base) else []
        rows = collect(arm, tasks)
        if not rows:
            print(f"\n### {lb}: no traces"); continue
        a = agg(rows)
        n = len(rows)
        tot = {k: sum(r[k] for r in rows) for k in ["ops_created", "ops_updated", "ops_deleted",
                                                     "errored_edits", "ops_errored", "trial_error_fixed",
                                                     "errored_abandoned"]}
        print(f"\n### {lb}  (n={n} traces) ###")
        print(f"  operators/trace : created {a['ops_created']:.1f}  updated(re-edit) {a['ops_updated']:.1f}  "
              f"deleted {a['ops_deleted']:.2f}   churn(max/mean edits per op) {a['max_edits_per_op']:.1f}/{a['mean_edits_per_op']:.2f}")
        print(f"  debugging/trace : errored-edits {a['errored_edits']:.2f} (attempts)  ops-errored {a['ops_errored']:.2f}  "
              f"debug-steps {a['debug_steps']:.2f}  agent-steps {a['agent_steps']:.1f}")
        print(f"  recovery/trace  : trial-error-FIXED {a['trial_error_fixed']:.2f}  errored-ABANDONED {a['errored_abandoned']:.2f}")
        print(f"  totals          : created {tot['ops_created']}  updated {tot['ops_updated']}  deleted {tot['ops_deleted']}  | "
              f"errored-edits {tot['errored_edits']}  ops-errored {tot['ops_errored']}  fixed {tot['trial_error_fixed']}  abandoned {tot['errored_abandoned']}")
        # split by outcome
        for label, sub in [("passed", [r for r in rows if r["passed"]]),
                           ("failed", [r for r in rows if r["passed"] is False])]:
            if sub:
                s = agg(sub)
                print(f"  [{label} n={len(sub)}] errored-edits {s['errored_edits']:.2f}  ops-errored {s['ops_errored']:.2f}  "
                      f"abandoned {s['errored_abandoned']:.2f}  steps {s['agent_steps']:.1f}  deleted {s['ops_deleted']:.2f}")
        if args.by_task:
            print(f"  {'task':24s} {'pass':5s} {'cre':>4s}{'upd':>4s}{'del':>4s}{'errE':>5s}{'opsE':>5s}{'fix':>4s}{'aband':>6s}")
            for r in sorted(rows, key=lambda r: r["task"]):
                p = "?" if r["passed"] is None else ("P" if r["passed"] else ".")
                print(f"  {r['task']:24s} {p:5s} {r['ops_created']:>4d}{r['ops_updated']:>4d}{r['ops_deleted']:>4d}"
                      f"{r['errored_edits']:>5d}{r['ops_errored']:>5d}{r['trial_error_fixed']:>4d}{r['errored_abandoned']:>6d}")


if __name__ == "__main__":
    main()
