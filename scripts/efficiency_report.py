#!/usr/bin/env python3
"""
Pipeline-efficiency report over judge_m5m6.json + stats.json.

(A) Productive fraction  — reuse M5/M6 per-subtask operator citations.
    productive units = distinct units cited by ANY subtask verdict.
    efficiency = productive / total_units ; redundancy = 1 - efficiency.
    work_ratio = total_units / n_subtasks (1.0 = one unit per gold step).
    NOTE: granularity-dependent — compare WITHIN a system family only.
(C) Cost per achieved coverage — granularity-invariant, cross-system fair.
    covered = M7 * n_subtasks (gold steps materialized-or-fused).
    tokens_per_covered, steps_per_covered, sec_per_covered.
churn = exec_error units / total_units (wasted work).

Run: .venv/bin/python scripts/efficiency_report.py --arms A B ...
"""
import argparse, json, glob, statistics as st
from pathlib import Path

KB = Path(__file__).resolve().parent.parent


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def unit_count(arm, task):
    """Total pipeline units: dataflow operators (workflow.json) or code steps."""
    rt = KB / "system_scratch" / arm / task / "reasoning_trace.json"
    if rt.exists():
        d = jload(rt) or []
        return len(d), "codeagent"
    w = jload(KB / "system_scratch" / arm / task / "workflow.json") or {}
    ops = (w.get("workflow") or {}).get("operators", []) or []
    return len(ops), "dataflow"


def per_task(arm, task):
    j = jload(KB / "system_scratch" / arm / task / "judge_m5m6.json")
    if not j:
        return None
    n_units, kind = unit_count(arm, task)
    if not n_units:
        n_units = j.get("n_operators", 0)
    n_sub = j.get("n_subtasks", 0) or 0
    if not n_sub:
        return None
    cited = set()
    for v in j["per_subtask"].values():
        for k in ("m5_operator", "m6_operator"):
            o = v.get(k)
            if o:
                for part in str(o).split(","):
                    cited.add(part.strip())
    cited.discard("")
    cited.discard("null")
    productive = len([1 for c in cited])  # distinct cited units
    # churn from op_flags in the cache
    flags = j.get("op_flags", {}) or {}
    err = sum(1 for f in flags.values() if "exec_error" in f)

    stats = jload(KB / "system_scratch" / arm / task / "stats.json") or {}
    toks = stats.get("total_tokens") or stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
    steps = stats.get("num_steps", n_units)
    sec = stats.get("elapsed_seconds", 0)

    covered = j.get("m7", 0) * n_sub
    return dict(
        kind=kind, n_units=n_units, n_sub=n_sub, productive=min(productive, n_units),
        m5=j.get("m5", 0), m6=j.get("m6", 0), m7=j.get("m7", 0),
        err=err, toks=toks, steps=steps, sec=sec, covered=covered,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()

    W = set()
    for f in glob.glob(str(KB / "workload/*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")):
            continue
        for t in json.load(open(f)):
            if t.get("id"):
                W.add(t["id"])

    print(f"{'arm':45s} {'n':>4s} {'units':>6s} {'wrk/sub':>7s} {'prod%':>6s} "
          f"{'M7':>5s} {'tok/cov':>8s} {'stp/cov':>7s} {'churn':>6s}")
    print("-" * 104)
    for arm in a.arms:
        base = KB / "system_scratch" / arm
        if not base.is_dir():
            print(f"{arm:45s}  (missing)")
            continue
        rows = [per_task(arm, d.name) for d in base.iterdir()
                if d.is_dir() and d.name in W]
        rows = [r for r in rows if r]
        if not rows:
            print(f"{arm:45s}    0")
            continue
        n = len(rows)
        units = st.mean(r["n_units"] for r in rows)
        wrk = st.mean(r["n_units"] / r["n_sub"] for r in rows if r["n_sub"])
        prod = st.mean(r["productive"] / r["n_units"] for r in rows if r["n_units"])
        m7 = st.mean(r["m7"] for r in rows)
        cov_rows = [r for r in rows if r["covered"] > 0 and r["toks"]]
        tpc = st.mean(r["toks"] / r["covered"] for r in cov_rows) if cov_rows else float("nan")
        spc = st.mean(r["steps"] / r["covered"] for r in cov_rows if r["steps"]) if cov_rows else float("nan")
        churn = st.mean((r["err"] / r["n_units"]) for r in rows if r["n_units"])
        short = arm.replace("DataflowSystem", "DF.").replace("CodeAgentSystem", "CA.")
        print(f"{short:45s} {n:4d} {units:6.1f} {wrk:7.2f} {prod*100:5.0f}% "
              f"{m7:5.2f} {tpc:8.0f} {spc:7.2f} {churn*100:5.0f}%")


if __name__ == "__main__":
    main()
