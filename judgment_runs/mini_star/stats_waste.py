#!/usr/bin/env python3
"""Audit the rendered `Column Schema and stats:` block for bytes that carry no
fact the agent could not read off the sample directly.

Waste classes measured (all render-side decidable — no worker change needed):

  W1 SAMPLE-COVERED   total_rows <= sampled_rows. Every value is already visible;
                      null=/distinct= are arithmetic on what is on screen.
  W2 ALL-UNIQUE       distinct == total_rows. Says only "no duplicates".
  W3 COMPLETE         null=0. Says only "no missing values".
  W4 CONSTANT         distinct == 1. One value, and the sample shows it.
  W5 SCHEMA ECHO      the `Schema (N cols): col (type), ...` line restates the
                      types that the stats block repeats per column.

W3 is the interesting one: it is the *default* state of clean data, so paying
bytes to announce it inverts the signal — an anomaly should cost bytes, not its
absence. Same argument as W2/W4.
"""
import json, glob, re, sys, statistics as st
from collections import Counter

ARMS = sys.argv[1:] or [
    f"DataflowSystemGPT5MiniA1RolePolicyReplicate{r}" for r in (1, 2, 3)
]
OPB = re.compile(r"### Operator `([^`]+)` \(([^)]+)\)(.*?)(?=\n### |\Z)", re.S)
SHAPE = re.compile(r"Output Table:\s*([\d,]+)\s*rows?,\s*(\d+)\s*cols?")
STATS_HDR = re.compile(r"Column Schema and stats:\s*\n((?:\s*-\s*\".*\n?)+)")
SCHEMA_ECHO = re.compile(r"^\s*Schema \(\d+ cols?\):.*$", re.M)
COL_LINE = re.compile(r'-\s*"([^"]+)"\s*\(([^)]+)\):\s*(.*)')
KV = re.compile(r"(\w+)=([^,]+)")

tot = Counter()
chars = Counter()
blocks = 0
seen_ops = set()

for arm in ARMS:
    for f in glob.glob(f"system_scratch/{arm}/*/react_steps.json"):
        try:
            doc = json.load(open(f))
        except Exception:
            continue
        S = [s for s in doc.get("steps", []) if s.get("inputMessages")]
        if not S:
            continue
        ctx = "\n".join(str(m.get("content", "")) for m in S[-1]["inputMessages"])
        for m in OPB.finditer(ctx):
            op, typ, body = m.group(1), m.group(2), m.group(3)
            key = (f, op)
            if key in seen_ops:
                continue
            seen_ops.add(key)
            sm = SHAPE.search(body)
            if not sm:
                continue
            tot["op_blocks"] += 1
            nrows = int(sm.group(1).replace(",", ""))
            shown = len(re.findall(r"\n\s*\d+\t", body))
            hdr = STATS_HDR.search(body)
            echo = SCHEMA_ECHO.search(body)
            if echo:
                tot["schema_echo_lines"] += 1
                chars["W5_schema_echo"] += len(echo.group(0))
            if not hdr:
                continue
            blocks += 1
            block_txt = hdr.group(0)
            chars["stats_block_total"] += len(block_txt)
            covered = shown >= nrows and nrows > 0
            if covered:
                tot["W1_blocks_sample_covered"] += 1
                chars["W1_sample_covered"] += len(block_txt)
            for cl in COL_LINE.finditer(block_txt):
                cname, ctype, rest = cl.group(1), cl.group(2), cl.group(3)
                tot["col_entries"] += 1
                kv = dict(KV.findall(rest))
                d = kv.get("distinct", "").strip()
                nl = kv.get("null", "").strip()
                dv = int(d) if d.isdigit() else None
                # length of just the metric text, excluding name/type
                mlen = len(rest)
                if nl == "0":
                    tot["W3_null_zero"] += 1
                    chars["W3_null_zero"] += len("null=0, ")
                if dv is not None and nrows and dv == nrows:
                    tot["W2_all_unique"] += 1
                    chars["W2_all_unique"] += len(f"distinct={d}, ")
                if dv == 1:
                    tot["W4_constant"] += 1
                if covered:
                    chars["W1_col_entries_covered"] += mlen

print(f"arms={len(ARMS)}  operator blocks with a stats block={blocks}  "
      f"column entries={tot['col_entries']}  (of {tot['op_blocks']} operator blocks total)\n")
print("waste class                          count    share      chars")
print("-" * 66)


def row(label, cnt, denom, ch):
    print(f"{label:<34}{cnt:>7}{cnt/max(denom,1)*100:>8.1f}%{ch:>11,}")


row("W1 block fully covered by sample", tot["W1_blocks_sample_covered"], blocks, chars["W1_sample_covered"])
row("W2 column all-unique (distinct=n)", tot["W2_all_unique"], tot["col_entries"], chars["W2_all_unique"])
row("W3 column complete (null=0)", tot["W3_null_zero"], tot["col_entries"], chars["W3_null_zero"])
row("W4 column constant (distinct=1)", tot["W4_constant"], tot["col_entries"], 0)
row("W5 Schema(...) echo line", tot["schema_echo_lines"], tot["op_blocks"], chars["W5_schema_echo"])

print(f"\ntotal stats-block bytes rendered      {chars['stats_block_total']:>12,}")
recoverable = chars["W1_sample_covered"] + chars["W5_schema_echo"]
print(f"  W1+W5 removable outright            {recoverable:>12,}  "
      f"({recoverable/max(chars['stats_block_total']+chars['W5_schema_echo'],1)*100:.0f}% of stats+echo bytes)")
partial = chars["W2_all_unique"] + chars["W3_null_zero"]
print(f"  W2+W3 degenerate metric text        {partial:>12,}")
print(f"  ---> upper-bound trim               {recoverable+partial:>12,} bytes "
      f"(~{(recoverable+partial)/4:,.0f} tokens across {len(ARMS)} reps x 20 tasks)")
