#!/usr/bin/env python3
"""
Programmatic per-operator footprint analysis for the three context levers.

For every operator in the star arms, extract structural features and label
whether each lever's render actually TOUCHES it:

  C1 sampling  : cap-bound  — its observation hits the 3k render cap
                 (block length near cap, or renderer elision markers present)
  C2 profiling : informative-stats — its stats block carries non-redundant
                 facts (duplicate_values>0 / material nulls / dirty profile)
  C3 history   : multi-edit — >=2 landed code versions (history renders only
                 for these)

Features per operator: role (source/interior/sink), topological depth, fan-in/
out, code lines, files read; per source file: extension, size, engine-computed
dirtiness (duplicate-row %, empty rows/cols, unnamed headers — parsed from the
stats arm's rendered Output Table profile, i.e. measured on FULL data).

Outputs judgment_runs/levers_report/footprints.json + association tables.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
SC = KB / "system_scratch"

ANCHOR = "DataflowSystemGPT52Delta3kSchemaOnly"
C1RAY = "DataflowSystemGPT52Delta5kSchemaOnly"
C2RAY = "DataflowSystemGPT52DeltaStats3kD2"

ELISION_ROW = re.compile(r"\n\s*\.\.\.(\t\.\.\.)+")   # jsonToTableFormat row gap
TRUNC_MARKS = ("…(truncated", "...[truncated]...")

OP_BLOCK = re.compile(r"^- operator (\S+) (?:added|updated)\s*$", re.M)


def last_context(sut, task):
    try:
        doc = json.load(open(SC / sut / task / "react_steps.json"))
    except Exception:
        return ""
    steps = [s for s in doc.get("steps", []) if s.get("role") == "agent" and s.get("inputMessages")]
    if not steps:
        return ""
    return "\n".join(str(m.get("content", "")) for m in steps[-1]["inputMessages"])


def op_blocks(context):
    """opId -> LAST rendered observation block for that op (DELTA grammar)."""
    out = {}
    matches = list(OP_BLOCK.finditer(context))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(context), start + 8000)
        # a block also ends at the next event header
        nxt = context.find("## Agent Event", start)
        if 0 <= nxt < end:
            end = nxt
        out[m.group(1)] = context[start:end]
    return out


def dag_features(sut, task):
    """opId -> role/depth/fanin/fanout/loc/files/edits from workflow + calls."""
    try:
        wf = json.load(open(SC / sut / task / "workflow.json"))["workflow"]
        doc = json.load(open(SC / sut / task / "react_steps.json"))
    except Exception:
        return {}
    edits = Counter()
    code_by = {}
    for s in doc.get("steps", []):
        if s.get("role") != "agent":
            continue
        for tc in s.get("toolCalls") or []:
            inp = tc.get("input") or {}
            if inp.get("operatorId") and inp.get("code"):
                edits[inp["operatorId"]] += 1
                code_by[inp["operatorId"]] = inp["code"]
    ops = {o["operatorID"]: o for o in wf.get("operators", [])}
    fin, fout = Counter(), Counter()
    parents = defaultdict(list)
    for l in wf.get("links", []):
        s_, t_ = l["source"]["operatorID"], l["target"]["operatorID"]
        fout[s_] += 1
        fin[t_] += 1
        parents[t_].append(s_)
    depth = {}

    def d_of(o, seen=()):
        if o in depth:
            return depth[o]
        if o in seen or not parents[o]:
            depth[o] = 0
            return 0
        depth[o] = 1 + max(d_of(p, seen + (o,)) for p in parents[o])
        return depth[o]

    feats = {}
    for op in ops:
        code = code_by.get(op, str(ops[op].get("operatorProperties", {}).get("code", "")))
        feats[op] = dict(
            role="source" if fin[op] == 0 else ("sink" if fout[op] == 0 else "interior"),
            depth=d_of(op),
            fanin=fin[op],
            fanout=fout[op],
            loc=code.count("\n") + 1,
            files=sorted(set(re.findall(r"data/[\w\-./]+\.\w+", code))),
            edits=edits.get(op, 1),
        )
    return feats


def file_dirtiness():
    """file -> engine-measured dirtiness, parsed from the stats arm's rendered
    Output Table profile of the op(s) that load it (full-data facts)."""
    dirt = {}
    for task in sorted(os.listdir(SC / C2RAY)):
        ctx = last_context(C2RAY, task)
        if not ctx:
            continue
        feats = dag_features(C2RAY, task)
        blocks = op_blocks(ctx)
        for op, f in feats.items():
            if f["role"] != "source" or not f["files"]:
                continue
            b = blocks.get(op, "")
            m_dup = re.search(r"duplicate rows: (\d+) of (\d+)", b)
            m_empty = re.search(r"empty rows: (\d+) of (\d+)", b)
            unnamed = len(re.findall(r"Unnamed", b))
            for file in f["files"]:
                d = dirt.setdefault(file, dict(dup_pct=0.0, empty_pct=0.0, unnamed=0, seen=0))
                d["seen"] += 1
                if m_dup:
                    d["dup_pct"] = max(d["dup_pct"], 100 * int(m_dup.group(1)) / max(1, int(m_dup.group(2))))
                if m_empty:
                    d["empty_pct"] = max(d["empty_pct"], 100 * int(m_empty.group(1)) / max(1, int(m_empty.group(2))))
                d["unnamed"] = max(d["unnamed"], unnamed)
    return dirt


def main():
    dirt = file_dirtiness()

    rows = []
    for task in sorted(os.listdir(SC / ANCHOR)):
        feats = dag_features(ANCHOR, task)
        if not feats:
            continue
        ctx = last_context(ANCHOR, task)
        blocks = op_blocks(ctx)
        # C2 informative-stats needs the stats arm's blocks for the SAME files;
        # ops don't match across arms, so C2 footprint is computed on C2's own
        # op table below. Here: C1 + C3 on the anchor.
        for op, f in feats.items():
            b = blocks.get(op, "")
            capped = bool(ELISION_ROW.search(b)) or any(m in b for m in TRUNC_MARKS) or len(b) >= 2700
            file = f["files"][0] if f["files"] else None
            fd = dirt.get(file, {}) if file else {}
            rows.append(dict(
                arm="anchor", task=task, op=op, **{k: f[k] for k in ("role", "depth", "fanin", "fanout", "loc", "edits")},
                file=file,
                ext=(os.path.splitext(file)[1] if file else None),
                fsize_kb=(os.path.getsize(KB / file) // 1024 if file and (KB / file).exists() else None),
                dup_pct=fd.get("dup_pct"), empty_pct=fd.get("empty_pct"), unnamed=fd.get("unnamed"),
                capped=capped,
            ))

    # C2 footprint on the stats arm's own ops
    c2rows = []
    for task in sorted(os.listdir(SC / C2RAY)):
        feats = dag_features(C2RAY, task)
        if not feats:
            continue
        blocks = op_blocks(last_context(C2RAY, task))
        for op, f in feats.items():
            b = blocks.get(op, "")
            stats_block = "Column Schema and stats" in b
            dup = re.search(r"duplicate_values=(\d+)", b)
            nulls = [int(x) for x in re.findall(r"null=(\d+)", b)]
            rowsN = re.search(r"Output Table: (\d+) rows", b)
            n = int(rowsN.group(1)) if rowsN else 0
            informative = stats_block and (
                (dup and int(dup.group(1)) > 0)
                or (n > 0 and nulls and max(nulls) > 0.05 * n)
                or ("duplicate rows:" in b)
                or ("Unnamed" in b)
            )
            file = f["files"][0] if f["files"] else None
            c2rows.append(dict(
                arm="c2", task=task, op=op, **{k: f[k] for k in ("role", "depth", "fanin", "fanout", "loc", "edits")},
                file=file, ext=(os.path.splitext(file)[1] if file else None),
                fsize_kb=(os.path.getsize(KB / file) // 1024 if file and (KB / file).exists() else None),
                stats_block=stats_block, informative=informative,
            ))

    json.dump({"anchor": rows, "c2": c2rows, "file_dirtiness": dirt},
              open(KB / "judgment_runs/levers_report/footprints.json", "w"), indent=1)

    def assoc(pop, base, label):
        def share(p, fn):
            xs = [fn(r) for r in p]
            xs = [x for x in xs if x is not None]
            return xs
        def rate(fn):
            a = share(pop, fn); b = share(base, fn)
            return (sum(a) / max(1, len(a)), sum(b) / max(1, len(b)))
        def med(fn):
            a = sorted(share(pop, fn)); b = sorted(share(base, fn))
            m = lambda x: x[len(x) // 2] if x else 0
            return m(a), m(b)
        print(f"\n== {label}: n={len(pop)} vs complement n={len(base)}")
        for name, fn, kind in [
            ("source-role", lambda r: 1 if r["role"] == "source" else 0, "rate"),
            ("sink-role", lambda r: 1 if r["role"] == "sink" else 0, "rate"),
            ("depth", lambda r: r["depth"], "med"),
            ("code lines", lambda r: r["loc"], "med"),
            ("edits", lambda r: r["edits"], "med"),
            ("file size KB", lambda r: r.get("fsize_kb"), "med"),
            ("file dup%", lambda r: r.get("dup_pct"), "med"),
            ("file empty%", lambda r: r.get("empty_pct"), "med"),
        ]:
            va, vb = (rate(fn) if kind == "rate" else med(fn))
            mark = " <<<" if (vb and va and (va / max(vb, 1e-9) > 1.5 or va / max(vb, 1e-9) < 0.67)) else ""
            print(f"   {name:14s} {va:>8.2f} vs {vb:>8.2f}{mark}")
        ex_p = Counter(r["ext"] for r in pop if r.get("ext"))
        ex_b = Counter(r["ext"] for r in base if r.get("ext"))
        print(f"   exts: pop={dict(ex_p.most_common(5))} base={dict(ex_b.most_common(5))}")

    capped = [r for r in rows if r["capped"]]
    assoc(capped, [r for r in rows if not r["capped"]], "C1 CAP-BOUND (sampling touches)")
    multi = [r for r in rows if r["edits"] >= 2]
    assoc(multi, [r for r in rows if r["edits"] < 2], "C3 MULTI-EDIT (history renders)")
    inf = [r for r in c2rows if r["informative"]]
    assoc(inf, [r for r in c2rows if r["stats_block"] and not r["informative"]], "C2 INFORMATIVE-STATS")


if __name__ == "__main__":
    sys.exit(main())
