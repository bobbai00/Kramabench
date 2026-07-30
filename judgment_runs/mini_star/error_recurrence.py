#!/usr/bin/env python3
"""Would keeping the ERRORING version of each operator help?

Proposed rule: in a snapshot core, retain an operator's erroring version (code +
error) so the agent cannot repeat a mistake whose evidence has been overwritten.

This measures, from existing LATEST+code traces (no new runs):
  A. how often an operator errors at all;
  B. SAME-OP REPEAT — an operator errors with signature S, is re-submitted, and
     errors again with the SAME S (the agent did not learn from what was still
     on screen -> retention would NOT have helped, the error was visible);
  C. EVICTED-THEN-REPEATED — signature S errored on some operator, S later
     disappeared from the rendered context (fixed/replaced), and S then appears
     again. This is exactly the case the rule targets: the evidence was gone.

Ground truth for "what the agent could see" is the rendered context itself
(inputMessages), parsed per step, so eviction is measured, not assumed.
"""
import json, glob, re, sys
from collections import defaultdict

ARMS = sys.argv[1:] or (
    [f"DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate{r}" for r in range(5)]
    + [f"DataflowSystemGPT5MiniLatest1kCodeInSnapReplicate{r}" for r in range(5)]
)

OP_BLOCK = re.compile(r"### Operator `([^`]+)` \([^)]*\)(.*?)(?=\n### |\Z)", re.S)
# the rendered error line inside a Result: block
ERR_LINE = re.compile(r"\[ERROR\][^\n]*?([A-Za-z_]*(?:Error|Exception))\s*:\s*([^\n]{0,80})")
ERR_ANY = re.compile(r"\[ERROR\]")


def sig(err_cls, msg):
    msg = re.sub(r"\d+", "N", msg)
    msg = re.sub(r"'[^']*'", "'X'", msg)  # normalise quoted paths/keys
    return f"{err_cls}|{msg.strip()[:50]}"


def errored_ops(ctx):
    """{opId: signature} for operators whose CURRENT result is an error."""
    out = {}
    for m in OP_BLOCK.finditer(ctx):
        op, body = m.group(1), m.group(2)
        # only the Result: region, so `except Exception` in code doesn't count
        res = body.split("Operator code:")[0]
        if not ERR_ANY.search(res):
            continue
        em = ERR_LINE.search(res)
        out[op] = sig(em.group(1), em.group(2)) if em else "unlabelled|"
    return out


tot = defaultdict(int)
tasks = defaultdict(int)
examples = []
n_runs = 0

for arm in ARMS:
    for f in glob.glob(f"system_scratch/{arm}/*/react_steps.json"):
        task = f.split("/")[-2]
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        ctxs = []
        for s in doc.get("steps", []):
            ims = s.get("inputMessages")
            if ims:
                ctxs.append("\n".join(str(m.get("content") or "") for m in ims))
        if not ctxs:
            continue
        n_runs += 1
        per_step = [errored_ops(c) for c in ctxs]
        if not any(per_step):
            continue
        tasks["any_error"] += 1
        tot["error_ops"] += sum(len(d) for d in per_step)

        # B: same op, same signature, seen in two NON-CONSECUTIVE renders with a
        # clean render in between => it was fixed then broke the same way again
        same_op_repeat = 0
        for op in {o for d in per_step for o in d}:
            states = [d.get(op) for d in per_step]  # None = absent/healthy
            prev = None
            gap = False
            for stt in states:
                if stt is None:
                    if prev is not None:
                        gap = True
                elif prev is not None and stt == prev and gap:
                    same_op_repeat += 1
                    gap = False
                if stt is not None:
                    prev = stt
        tot["same_op_repeat_after_clean"] += same_op_repeat

        # C: signature visible, later fully absent from context, then returns
        sig_steps = [set(d.values()) for d in per_step]
        evicted_returned = 0
        for s_ in set().union(*sig_steps) if sig_steps else set():
            seen = [s_ in ss for ss in sig_steps]
            # pattern True ... False ... True
            if True in seen:
                first = seen.index(True)
                after = seen[first:]
                if False in after:
                    off = after.index(False)
                    if True in after[off:]:
                        evicted_returned += 1
        tot["evicted_then_returned"] += evicted_returned
        if same_op_repeat:
            tasks["same_op_repeat"] += 1
        if evicted_returned:
            tasks["evicted_then_returned"] += 1
            if len(examples) < 6:
                examples.append((arm.replace("DataflowSystemGPT5Mini", ""), task, evicted_returned))

print(f"arms={len(ARMS)}  task-runs with rendered context={n_runs}")
print(f"  task-runs with >=1 errored operator : {tasks['any_error']} ({tasks['any_error']/max(n_runs,1)*100:.1f}%)")
print(f"  errored-operator renders (total)    : {tot['error_ops']:,}")
print()
print("Case B — operator errored, went clean, then SAME error again")
print(f"  occurrences {tot['same_op_repeat_after_clean']:,} in {tasks['same_op_repeat']} task-runs "
      f"({tasks['same_op_repeat']/max(n_runs,1)*100:.1f}%)")
print()
print("Case C — error signature LEFT the context entirely, then returned")
print("  (the case 'retain the erroring version' is designed to prevent)")
print(f"  occurrences {tot['evicted_then_returned']:,} in {tasks['evicted_then_returned']} task-runs "
      f"({tasks['evicted_then_returned']/max(n_runs,1)*100:.1f}%)")
if examples:
    print("\n  examples (arm, task, count):")
    for a, t, c in examples:
        print(f"    {a:<28} {t:<22} {c}")
