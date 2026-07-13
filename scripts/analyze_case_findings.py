#!/usr/bin/env python3
"""
Full-population computations behind judgment_runs/levers_report/FINDINGS.md.

Three tables that generalize the per-case numbers of `kb.py case-metrics`
from the Venn categories to ALL tasks:

  F1  render coverage by output-table-size band x cap (C1 arms)
  F3  sink-share of the final DAG: pass-vs-fail medians + the churn tail flag
      (sink-share >= 50% AND ops >= 8), Latest arm with Delta anchor control
  F4  output cardinality by operator depth (anchor arm)

Run from repo root: .venv/bin/python scripts/analyze_case_findings.py
"""

import importlib.util
import os
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)

ANCHOR = "DataflowSystemGPT52Delta3kSchemaOnly"
C1RAY = "DataflowSystemGPT52Delta5kSchemaOnly"
C3RAY = "DataflowSystemGPT52Latest3kSchemaOnly"


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def f1_coverage_bands():
    """Rendered/actual row coverage by out-rows band, per cap. The 1-row band
    is an artifact (narrow 1-row tables render without >=2 tabs, so shown_rows
    undercounts) — printed but not interpretable."""
    print("== F1: render coverage by output-table-size band x cap ==")
    for sut in (ANCHOR, C1RAY):
        bands = {"1": [], "2-40": [], "41-100": [], "101-1k": [], ">1k": []}
        for task in sorted(os.listdir(KB / "system_scratch" / sut)):
            for o in kb.task_op_metrics(sut, task):
                r = o["out_rows"]
                if not r:
                    continue
                cov = min(100.0, 100.0 * o["shown_rows"] / r)
                b = ("1" if r == 1 else "2-40" if r <= 40 else "41-100" if r <= 100
                     else "101-1k" if r <= 1000 else ">1k")
                bands[b].append(cov)
        nm = sut.replace("DataflowSystemGPT52", "")
        for b, xs in bands.items():
            full = sum(1 for x in xs if x >= 90)
            print(f"  {nm:22s} rows {b:7s} n={len(xs):4d}  cov med={med(xs):6.1f}%  "
                  f"fully-visible(>=90%)={100 * full / max(1, len(xs)):5.1f}%")


def f3_sink_share():
    """Sink-share (unconsumed leaves / ops) of the final DAG. Medians do NOT
    separate pass from fail; the signal is a tail flag."""
    print("\n== F3: sink share of final DAG — medians + churn tail flag ==")
    for sut in (C3RAY, ANCHOR):
        sc = kb.answer_scores(sut)
        cost = {r["task_id"]: r for r in kb.load_cost_stats(sut)}
        rows = []
        for task, s in sc.items():
            feats = kb.task_op_features(sut, task)
            if len(feats) < 2:
                continue
            share = sum(1 for f in feats if f["role"] == "sink") / len(feats)
            rows.append((task, share, len(feats), s,
                         cost.get(task, {}).get("num_steps", 0),
                         cost.get(task, {}).get("cost", 0)))
        nm = sut.replace("DataflowSystemGPT52", "")
        for g, xs in (("pass", [r for r in rows if r[3] >= 0.9]),
                      ("fail", [r for r in rows if r[3] < 0.9])):
            print(f"  {nm:22s} {g:4s} n={len(xs):3d}  sink-share med={100 * med([x[1] for x in xs]):5.1f}%  "
                  f"ops/task med={med([x[2] for x in xs])}")
        flag = [r for r in rows if r[1] >= 0.5 and r[2] >= 8]
        rest = [r for r in rows if not (r[1] >= 0.5 and r[2] >= 8)]
        print(f"  {nm} CHURN FLAG (sink-share>=50% AND ops>=8): {len(flag)}/{len(rows)}")
        for lbl, xs in (("flagged", flag), ("rest", rest)):
            if xs:
                print(f"    {lbl:8s} pass-rate {100 * sum(1 for r in xs if r[3] >= 0.9) / len(xs):3.0f}%  "
                      f"steps med {med([r[4] for r in xs]):2d}  cost med ${med([r[5] for r in xs]):.2f}")
        for r in sorted(flag, key=lambda r: -r[1]):
            print(f"      {r[0]:26s} sink-share {100 * r[1]:4.0f}%  ops {r[2]:3d}  steps {r[4]:3d}  "
                  f"${r[5]:.2f}  {'PASS' if r[3] >= 0.9 else 'FAIL'}")


def f4_rows_by_depth():
    print("\n== F4: output cardinality by operator depth (anchor, all tasks) ==")
    by_d = {}
    for task in sorted(os.listdir(KB / "system_scratch" / ANCHOR)):
        for o in kb.task_op_metrics(ANCHOR, task):
            if o["out_rows"] is None:
                continue
            by_d.setdefault(min(o["depth"], 3), []).append((o["out_rows"], o["cells"] or 0))
    for d in sorted(by_d):
        rows = sorted(x[0] for x in by_d[d])
        print(f"  depth {'3+' if d == 3 else d}: n={len(rows):4d}  out-rows med={rows[len(rows) // 2]:8,d}  "
              f"p90={rows[int(0.9 * len(rows))]:9,d}  cells med={med([x[1] for x in by_d[d]]):10,.0f}")


if __name__ == "__main__":
    f1_coverage_bands()
    f3_sink_share()
    f4_rows_by_depth()
