#!/usr/bin/env python3
"""Where does the render budget actually GO? Per-operator, per-component bytes.

Motivation: the source-rich/downstream-lean family (C9 67.5, C10 67.1, N3 68.4,
N5 67.9) all lost to uniform-5k (70.1-71.3). The stated reason was "derived
tables are already under 1k, so leaning downstream saves nothing" — but that was
inferred from a handful of traces, never measured across the corpus. Any
fine-grained per-operator rule is guesswork until we know the real distribution.

Method: `inputMessages` on the LAST step of react_steps.json is the full prompt
the model saw. Parse its `# Current Dataflow` section into operator blocks and
attribute every byte to a component (summary / inputs / files-read / code /
table-rows / schema / stats). Join to KramaBench's own per-task score.

No self-invented accuracy anywhere: correctness comes from the measures CSVs.
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

import pandas as pd

KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
SM = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]
WL_N = {"archeology": 12, "astronomy": 12, "biomedical": 9,
        "environment": 20, "legal": 30, "wildfire": 21}

OP_HDR = re.compile(r"^### Operator `([^`]+)` \(([^)]+)\)\s*$", re.M)


def task_scores(sut):
    """task_id -> mean metric value (KramaBench's own numbers)."""
    dfs = []
    for wl, n in WL_N.items():
        for f in sorted(glob.glob(f"{KB}/results/{sut}/{wl}_measures_*.csv"), reverse=True):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if d[d["metric"].isin(SM)]["task_id"].nunique() >= n:
                dfs.append(d)
                break
    if not dfs:
        return {}
    d = pd.concat(dfs, ignore_index=True)
    d = d[d["metric"].isin(SM)]
    return d.groupby("task_id")["value"].mean().to_dict()


def last_context(trace_path):
    """The largest rendered context in the trace (the step with most input)."""
    try:
        d = json.load(open(trace_path))
    except Exception:
        return None
    best = None
    for stp in d.get("steps", []):
        im = stp.get("inputMessages")
        if not im:
            continue
        try:
            msgs = im if isinstance(im, list) else json.loads(im)
        except Exception:
            continue
        for m in msgs:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str) and "# Current Dataflow" in c:
                if best is None or len(c) > len(best):
                    best = c
    return best


def split_blocks(ctx):
    """[(op_name, op_type, block_text)] from the Current Dataflow section."""
    i = ctx.find("# Current Dataflow")
    if i < 0:
        return []
    body = ctx[i:]
    hits = list(OP_HDR.finditer(body))
    out = []
    for k, m in enumerate(hits):
        end = hits[k + 1].start() if k + 1 < len(hits) else len(body)
        out.append((m.group(1), m.group(2), body[m.end():end]))
    return out


def attribute(block):
    """Bytes per component within one operator block."""
    comp = defaultdict(int)
    section = None          # None | code | result
    for line in block.splitlines():
        n = len(line) + 1
        s = line.strip()
        if s.startswith("Summary:"):
            comp["summary"] += n; section = None
        elif s.startswith("Inputs:"):
            comp["inputs"] += n; section = None
        elif s.startswith("Files read:"):
            comp["files_read"] += n; section = None
        elif s.startswith("Code:"):
            comp["code"] += n; section = "code"
        elif s.startswith("Result:"):
            comp["result_hdr"] += n; section = "result"
        elif s.startswith("Output Table:") or s.startswith("Output:"):
            comp["result_hdr"] += n
        elif s.startswith("Schema ("):
            comp["schema"] += n
        elif section == "code":
            comp["code"] += n
        elif section == "result":
            # A per-column stats line looks like:
            #   - "lng" (numeric): null=0, mean=14.53, min=-179.6, max=179.4
            # The leading `- "` is what matters; an earlier version of this
            # regex omitted the quote and misfiled ~26% of every stats-on arm's
            # bytes as "other", which understated stats as 2-3% of the render
            # when the true figure is ~25-30%.
            if s.startswith('- "') or s.startswith("- '"):
                comp["stats"] += n
            elif "\t" in line:
                comp["table_rows"] += n
            elif re.match(r"^\s*[\w .%()/-]+:", s):
                comp["stats"] += n
            elif not s:
                comp["blank"] += n
            else:
                comp["other_result"] += n
        else:
            comp["other"] += n
    return comp


def run(arms):
    print("BYTE ACCOUNTING — per-operator render components vs KramaBench score")
    print("Context = largest `# Current Dataflow` prompt in each trace.\n")
    for label, sut in arms:
        scores = task_scores(sut)
        if not scores:
            print(f"{label}: no scored CSVs, skipping")
            continue
        by_type = defaultdict(lambda: defaultdict(list))
        tot_ctx, per_task = [], {}
        n_ops = []
        for tdir in sorted(glob.glob(f"{KB}/system_scratch/{sut}/*/")):
            task = os.path.basename(tdir.rstrip("/"))
            ctx = last_context(os.path.join(tdir, "react_steps.json"))
            if not ctx:
                continue
            blocks = split_blocks(ctx)
            if not blocks:
                continue
            tot_ctx.append(len(ctx))
            n_ops.append(len(blocks))
            tsum = defaultdict(int)
            for name, otype, blk in blocks:
                comp = attribute(blk)
                for k, v in comp.items():
                    by_type[otype][k].append(v)
                    tsum[k] += v
                by_type[otype]["_block"].append(sum(comp.values()))
            per_task[task] = (sum(tsum.values()), tsum, scores.get(task))

        print(f"=== {label}  ({sut.split('Mini')[1]}) ===")
        print(f"  traces={len(tot_ctx)}  mean full prompt={st.mean(tot_ctx):,.0f} B"
              f"  mean operators/task={st.mean(n_ops):.1f}")
        print(f"  {'op type':<18}{'n':>5}{'block B':>10}{'rows':>9}{'stats':>8}"
              f"{'schema':>8}{'code':>8}{'summary':>8}{'files':>7}")
        for otype, comps in sorted(by_type.items(),
                                   key=lambda kv: -sum(kv[1]["_block"])):
            g = lambda k: st.mean(comps[k]) if comps.get(k) else 0
            print(f"  {otype:<18}{len(comps['_block']):>5}{g('_block'):>10,.0f}"
                  f"{g('table_rows'):>9,.0f}{g('stats'):>8,.0f}{g('schema'):>8,.0f}"
                  f"{g('code'):>8,.0f}{g('summary'):>8,.0f}{g('files_read'):>7,.0f}")

        # share of total render bytes by component
        agg = defaultdict(int)
        for _, tsum, _ in per_task.values():
            for k, v in tsum.items():
                agg[k] += v
        tot = sum(agg.values()) or 1
        share = "  ".join(f"{k}={v/tot*100:.1f}%" for k, v in
                          sorted(agg.items(), key=lambda kv: -kv[1])[:6])
        print(f"  share of dataflow bytes: {share}")

        # does a bigger render go with a better score?
        pairs = [(v[0], v[2]) for v in per_task.values() if v[2] is not None]
        if len(pairs) > 8:
            bs = sorted(pairs)
            half = len(bs) // 2
            lo = st.mean([s for _, s in bs[:half]])
            hi = st.mean([s for _, s in bs[half:]])
            print(f"  score by render size: smallest-half {lo*100:.1f}  "
                  f"largest-half {hi*100:.1f}   (confounded by task difficulty — "
                  f"big renders are big tasks)")
        print()


if __name__ == "__main__":
    ARMS = [
        ("D8   5k no-stats", "DataflowSystemGPT5MiniD8Latest5kCodeReplicate1"),
        ("N1   5k +stats",   "DataflowSystemGPT5MiniN1Latest5kStatsReplicate1"),
        ("D12  1k +stats",   "DataflowSystemGPT5MiniD12LatestStats1kCodeReplicate1"),
        ("N4   2k +stats",   "DataflowSystemGPT5MiniN4Latest2kStatsReplicate1"),
        ("C9   5k/1k split", "DataflowSystemGPT5MiniC9SourceRichLatestReplicate1"),
    ]
    if len(sys.argv) > 1:
        ARMS = [(a, a) for a in sys.argv[1:]]
    run(ARMS)
