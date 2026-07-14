#!/usr/bin/env python3
"""Compact one-shot dump for a common-core failure dive: gold solution, the
question, and every star-arm's answer/score/steps for one task. Keeps the
main-loop dive cheap. Usage: python scripts/gather_failure.py <task-id>"""
import json
import sys
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
ARMS = ["DataflowSystemGPT52DeltaStats3kD2", "DataflowSystemGPT52Delta3kSchemaOnly",
        "DataflowSystemGPT52Delta5kSchemaOnly", "DataflowSystemGPT52Latest3kSchemaOnly"]
task = sys.argv[1]
domain = task.rsplit("-", 2)[0]

gt = json.load(open(KB / "system_scratch" / ARMS[0] / task / "ground_truth.json"))
print(f"### {task}\nQ: {gt.get('query')}\nGOLD: {gt.get('answer')}  (type {gt.get('answer_type')})")
sol = KB / "solutions" / domain / f"{task}.py"
print(f"\n--- GOLD SOLUTION ({sol.relative_to(KB)}):\n{sol.read_text() if sol.exists() else 'MISSING'}")
print("\n--- ARM OUTCOMES:")
for a in ARMS:
    d = KB / "system_scratch" / a / task
    ans = json.load(open(d / "answer.json")).get("answer") if (d / "answer.json").exists() else "?"
    ev = json.load(open(d / "evaluation.json")) if (d / "evaluation.json").exists() else {}
    st = json.load(open(d / "stats.json")) if (d / "stats.json").exists() else {}
    sc = ev.get("success") or ev.get("f1") or ev.get("rae_score") or 0
    print(f"  {a.replace('DataflowSystemGPT52',''):20s} steps={st.get('num_steps'):>3} "
          f"score={sc} ans={str(ans)[:70]!r}")
print(f"\n--- data files referenced by gold:")
import re
for f in sorted(set(re.findall(r"data/[\w\-./ ]+\.\w+", sol.read_text() if sol.exists() else ""))):
    p = KB / f
    print(f"  {f}  ({'exists' if p.exists() else 'MISSING'}, {p.stat().st_size if p.exists() else 0} bytes)")
