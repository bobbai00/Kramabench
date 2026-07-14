#!/usr/bin/env python3
"""
Uniform trace extraction for semantic walks (compare-traces skill, accuracy
cases). Prints, for one (SUT, task):

  - the task question + gold answer (ground_truth.json)
  - per agent step: every tool call (operator id + FULL code / other input),
    and per rendered execution: each op's Output Table shape + observation
    head (from the NEXT step's rendered context delta, both grammars)
  - the final answer + response tail

Usage: python scripts/extract_walk.py --sut <ARM> --task <task-id> [--obs-chars 500]
"""

import argparse
import json
import re
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
TBL = re.compile(r"Output Table: (\d+) rows?, (\d+) cols")
DELTA_OP = re.compile(r"^- operator (\S+) (?:added|updated)\s*$", re.M)
LATEST_OP = re.compile(r"^#{2,4} (?:Operator )?`(\S+)` \(\w+\)\s*$", re.M)
SECTION = re.compile(r"^#{1,6} ", re.M)


def op_blocks(ctx):
    fences, pos = [], 0
    while True:
        i = ctx.find("```", pos)
        if i < 0:
            break
        j = ctx.find("```", i + 3)
        fences.append((i, len(ctx) if j < 0 else j + 3))
        pos = len(ctx) if j < 0 else j + 3
    outside = lambda i: all(not (a <= i < b) for a, b in fences)
    heads = sorted([(m.start(), m.group(1)) for p in (DELTA_OP, LATEST_OP)
                    for m in p.finditer(ctx) if outside(m.start())])
    stops = sorted([m.start() for m in SECTION.finditer(ctx) if outside(m.start())]
                   + [h[0] for h in heads])
    out = {}
    for start, op in heads:
        nl = ctx.find("\n", start)
        begin = len(ctx) if nl < 0 else nl + 1
        end = min((s for s in stops if s > start), default=len(ctx))
        out[op] = ctx[begin:min(end, begin + 12000)]
    return out


def step_ctx(step):
    return "\n".join(str(m.get("content", "")) for m in (step.get("inputMessages") or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--obs-chars", type=int, default=500)
    a = ap.parse_args()
    d = KB / "system_scratch" / a.sut / a.task

    gt = json.load(open(d / "ground_truth.json")) if (d / "ground_truth.json").exists() else {}
    print(f"##### {a.sut} / {a.task}")
    print(f"QUESTION: {gt.get('query') or gt.get('question') or '(see prompt.txt)'}")
    print(f"GOLD ANSWER: {gt.get('answer')}\n")

    doc = json.load(open(d / "react_steps.json"))
    steps = doc.get("steps", [])
    agent_steps = [s for s in steps if s.get("role") == "agent"]

    prev_blocks = {}
    for i, s in enumerate(agent_steps):
        ctx = step_ctx(s)
        blocks = op_blocks(ctx) if ctx else {}
        # observations that are NEW/CHANGED versus the previous step's render
        changed = {op: b for op, b in blocks.items()
                   if prev_blocks.get(op, "")[:200] != b[:200]}
        if changed and i > 0:
            print(f"--- rendered before step {i} (new/changed observations) ---")
            for op, b in changed.items():
                m = TBL.search(b)
                shape = f"{m.group(1)}x{m.group(2)}" if m else "?"
                head = " | ".join(ln.strip() for ln in b.strip().splitlines()[:6])
                print(f"  [{op}] Output {shape}: {head[:a.obs_chars]}")
        prev_blocks = blocks or prev_blocks

        tcs = s.get("toolCalls") or []
        if tcs:
            print(f"=== STEP {i}: {len(tcs)} tool call(s)")
            for tc in tcs:
                inp = tc.get("input") or {}
                if inp.get("code"):
                    print(f"  TOOL op={inp.get('operatorId')}"
                          + (f" upstream={inp.get('upstreamOperatorIds')}" if inp.get("upstreamOperatorIds") else ""))
                    for ln in str(inp["code"]).splitlines():
                        print(f"    | {ln}")
                else:
                    print(f"  TOOL {json.dumps(inp)[:400]}")
        txt = s.get("content") or s.get("text") or ""
        if txt and not tcs:
            print(f"=== STEP {i}: TEXT: {str(txt)[:600]}")

    ans = json.load(open(d / "answer.json")) if (d / "answer.json").exists() else {}
    print(f"\nFINAL PARSED ANSWER: {ans.get('answer')}")
    resp = (d / "response.txt").read_text() if (d / "response.txt").exists() else ""
    print(f"RESPONSE TAIL: ...{resp[-400:]}")


if __name__ == "__main__":
    main()
