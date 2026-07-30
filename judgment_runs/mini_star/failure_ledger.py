#!/usr/bin/env python3
"""Prototype: parse a cumulative FAILURE-FACT LEDGER out of rendered errors.

The service already emits a structured error line per failing operator:

    [ERROR] (line N) `<failing expression>` - <ExceptionClass>: <message>

so parsing is one regex + a per-class rule that names the SUBJECT (usually the
file the expression touched) and a normalised PREDICATE (what is wrong with it).
Facts are deduped on (class, subject, predicate) and counted, so the ledger is
one line per distinct failure no matter how many times it recurs.

Design points that matter for the context:
  * the subject often lives in the EXPRESSION, not the message — UnicodeDecodeError
    never names the file, so a message-only parse would produce useless facts;
  * numbers/byte-offsets are normalised away, otherwise "position 1489" and
    "position 2013" look like two different problems;
  * append-only: the ledger sits at the END of a stable prefix, so unlike
    thought replay it does not invalidate the prompt cache.

Usage: failure_ledger.py [<sut> <task>]   (defaults to a known thrashing run)
"""
import json, re, sys, glob
from collections import OrderedDict

ERR = re.compile(r"\[ERROR\]\s*(?:\(line (\d+)\)\s*)?`(.*?)`\s*-\s*([A-Za-z_]*(?:Error|Exception))\s*:\s*(.*)")
OP_BLOCK = re.compile(r"### Operator `([^`]+)` \([^)]*\)(.*?)(?=\n### |\Z)", re.S)
STR_LIT = re.compile(r"""['"]([^'"]{3,200})['"]""")


def subject_from(expr: str, msg: str):
    """Name the thing that failed. Prefer a path/filename in the message, else
    the first string literal of the failing expression, else the expression."""
    m = re.search(r"No such file or directory:\s*'([^']+)'", msg)
    if m:
        return m.group(1)
    lit = STR_LIT.search(msg)
    if lit and ("/" in lit.group(1) or "." in lit.group(1)):
        return lit.group(1)
    lit = STR_LIT.search(expr)
    if lit:
        return lit.group(1)
    return expr.strip()[:60]


def predicate_from(cls: str, msg: str):
    """Normalised, actionable statement of what is wrong."""
    m = re.sub(r"\d+", "N", msg)
    m = re.sub(r"0x[0-9a-fA-F]+", "0xNN", m)
    if cls == "FileNotFoundError":
        return "does not exist"
    if cls == "UnicodeDecodeError":
        enc = re.search(r"'([\w-]+)' codec", msg)
        return f"is not {enc.group(1) if enc else 'utf-8'} — needs another encoding (try latin-1)"
    if cls == "EmptyDataError":
        return "has no parseable columns at this header/skiprows setting"
    if cls == "ParserError":
        return f"failed CSV tokenisation ({m.strip()[:60]})"
    if cls == "ModuleNotFoundError":
        return "module is unavailable in the sandbox"
    if cls == "KeyError":
        return f"key absent: {m.strip()[:50]}"
    return m.strip()[:70]


def parse_ledger(contexts):
    """contexts: rendered context strings in step order -> OrderedDict of facts."""
    facts = OrderedDict()
    for step_no, ctx in enumerate(contexts, 1):
        for m in OP_BLOCK.finditer(ctx):
            op, body = m.group(1), m.group(2)
            result_region = body.split("Operator code:")[0]
            for line in result_region.splitlines():
                em = ERR.search(line)
                if not em:
                    continue
                _, expr, cls, msg = em.groups()
                subj = subject_from(expr, msg)
                pred = predicate_from(cls, msg)
                key = (cls, subj, pred)
                if key in facts:
                    facts[key]["hits"] += 1
                    facts[key]["ops"].add(op)
                    facts[key]["last_step"] = step_no
                else:
                    facts[key] = dict(hits=1, ops={op}, first_step=step_no, last_step=step_no)
    return facts


def render(facts, max_lines=8):
    """The block that would be appended to the context."""
    if not facts:
        return ""
    out = ["Known failures (do not repeat these):"]
    for (cls, subj, pred), meta in list(facts.items())[:max_lines]:
        n = meta["hits"]
        times = f" ×{n}" if n > 1 else ""
        out.append(f"- `{subj}` {pred} [{cls}{times}]")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        sut, task = sys.argv[1], sys.argv[2]
    else:
        sut, task = "DataflowSystemGPT5MiniLatest5kCodeInSnapReplicate0", "wildfire-hard-19"
    path = f"system_scratch/{sut}/{task}/react_steps.json"
    doc = json.load(open(path))
    ctxs = [
        "\n".join(str(m.get("content") or "") for m in s["inputMessages"])
        for s in doc.get("steps", [])
        if s.get("inputMessages")
    ]
    facts = parse_ledger(ctxs)
    print(f"=== {task} ({sut.replace('DataflowSystemGPT5Mini','')}) — {len(ctxs)} steps ===")
    print(f"distinct failure facts parsed: {len(facts)}\n")
    for (cls, subj, pred), meta in facts.items():
        print(f"  [{cls}] hits={meta['hits']} ops={sorted(meta['ops'])} steps {meta['first_step']}->{meta['last_step']}")
        print(f"     subject : {subj}")
        print(f"     predicate: {pred}")
    print("\n--- rendered ledger block (what the agent would see) ---")
    print(render(facts))
    print(f"\nledger size: {len(render(facts))} chars")
