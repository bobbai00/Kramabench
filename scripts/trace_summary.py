#!/usr/bin/env python
"""Print a compact, human-readable summary of one agent trace.

    python scripts/trace_summary.py --sut ARM --task TASK_ID

Emits:
  Data:  the files the task declares
  Task:  the original question
  Agent trace: one line per operator, in authoring order, with its input operators.

Operator lines are drawn from workflow.json (final DAG: code + links) joined
with react_steps.json (authoring order, edit count, execution errors).
"""
import argparse
import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def task_def(task_id):
    wl = task_id.rsplit("-", 2)[0]
    for t in load(os.path.join(ROOT, f"workload/{wl}.json")) or []:
        if t.get("id") == task_id:
            return t
    return {}


def authoring_order(sut, task_id):
    """[(operatorId, n_writes, had_error)] in the order the agent first wrote them."""
    doc = load(os.path.join(ROOT, "system_scratch", sut, task_id, "react_steps.json"))
    order, writes, errs = [], {}, set()
    for s in (doc or {}).get("steps", []):
        if s.get("role") != "agent":
            continue
        tc = str(s.get("toolCalls") or "")
        tr = str(s.get("toolResults") or "")
        ids = re.findall(r"'operatorId': '([^']+)'", tc)
        for oid in ids:
            if oid not in writes:
                order.append(oid)
            writes[oid] = writes.get(oid, 0) + 1
        if "'isError': True" in tr:
            errs.update(ids)
    return [(o, writes[o], o in errs) for o in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--no-header", action="store_true")
    a = ap.parse_args()

    d = os.path.join(ROOT, "system_scratch", a.sut, a.task)
    t = task_def(a.task)
    wf = (load(os.path.join(d, "workflow.json")) or {}).get("workflow") or {}
    ops = {o["operatorID"]: o for o in wf.get("operators", [])}
    deps = {}
    for l in wf.get("links", []):
        deps.setdefault(l["target"]["operatorID"], []).append(l["source"]["operatorID"])

    ans = (load(os.path.join(d, "answer.json")) or {}).get("answer")
    ev = load(os.path.join(d, "evaluation.json")) or {}
    score = next((ev[m] for m in ("success", "llm_paraphrase", "rae_score", "f1", "f1_approximate")
                  if isinstance(ev.get(m), (int, float))), None)

    if not a.no_header:
        print(f"Data: {t.get('data_sources')}")
        print(f"Task: {t.get('query')}")
        print(f"Gold: {t.get('answer')}   ({t.get('answer_type')})")
        print()
    print(f"### {a.sut}  ->  answer {ans!r}   score {score}")
    print("Agent trace:")
    for oid, n, err in authoring_order(a.sut, a.task):
        op = ops.get(oid)
        if op is None:
            print(f"* {oid}: [written then deleted]"
                  f"{' (edited %dx)' % n if n > 1 else ''}")
            continue
        name = op.get("customDisplayName") or ""
        flags = []
        if n > 1:
            flags.append(f"edited {n}x")
        if err:
            flags.append("had exec error")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"* {oid}: {name}, dependent operators: {deps.get(oid, [])}{suffix}")


if __name__ == "__main__":
    main()
