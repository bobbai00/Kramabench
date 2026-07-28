#!/usr/bin/env python3
"""Print "ARM WORKLOAD TASK" lines for tasks scoring <0.9 (answer-type metric),
across the given arms. Round-0 mode (--all) prints every (arm,task) pair.
Used to feed the global 4-wide evaluate.py pool in orchestrator4b."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Desktop/bobflow/Kramabench"))
import kb
D = os.path.dirname(__file__)
TASKS = open(os.path.join(D, "all104.txt")).read().split()
PASS = 0.9

args = sys.argv[1:]
allmode = "--all" in args
arms = [a for a in args if a != "--all"]

for arm in arms:
    sc = {} if allmode else kb.answer_scores(arm)
    for t in TASKS:
        if allmode or (sc.get(t) or 0) < PASS:
            wl = t.rsplit("-", 2)[0]
            print(f"{arm} {wl} {t}")
