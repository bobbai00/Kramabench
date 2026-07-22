#!/usr/bin/env python3
"""
Part 2 — "pipeline design / functionality identification" metric, using
KramaBench's BUILTIN judge (GPTInterface.evaluate_data_pipeline + the paper's
PIPELINE_EVALUATION_PROMPT). For each task, feed the judge the agent's generated
pipeline (its final DAG code, serialized) + the task's key functionalities
(subtask steps); the judge returns Yes/No per functionality. Score = fraction Yes
= the paper's "% of important data tasks the system identified".

Run: .venv/bin/python scripts/pipeline_design_metric.py --arms SUT... --tasks ID...
"""
import argparse, os, re, sys, importlib.util
from pathlib import Path
KB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KB))  # so `import benchmark` resolves when run from scripts/
for line in open(KB / ".env"):
    line = re.sub(r"^export\s+", "", line.strip())
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
m1 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("m1", KB / "scripts/m1_plan_coverage.py"))
importlib.util.spec_from_file_location("m1", KB / "scripts/m1_plan_coverage.py").loader.exec_module(m1)
from benchmark.llm_tools.gpt_interface import GPTInterface

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    a = ap.parse_args()
    W = m1.load_workload()
    judge = GPTInterface(model=a.model)
    spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
    kb = importlib.util.module_from_spec(spec); spec.loader.exec_module(kb)
    scores = {arm: kb.answer_scores(arm) for arm in a.arms}
    short = lambda s: s.replace("DataflowSystemGPT52", "52.").replace("ProbePrompt", "·P")
    print(f"{'task':22s} " + " ".join(f"{short(arm):>22s}" for arm in a.arms))
    print("-" * (22 + 23 * len(a.arms)))
    agg = {arm: [] for arm in a.arms}
    for t in a.tasks:
        if t not in W: continue
        row = f"{t:22s} "
        for arm in a.arms:
            code = m1.agent_code(arm, t)
            if code is None:
                row += f"{'—':>22s} "; continue
            res = judge.evaluate_data_pipeline(sut_generated_pipeline=code, task=W[t])
            lst = res[0]
            if not lst:
                row += f"{'ERR':>22s} "; continue
            cov = sum(1 for b in lst if b) / len(lst)
            agg[arm].append(cov)
            ans = scores[arm].get(t)
            anss = "P" if (ans is not None and ans >= 0.9) else ("f" if ans is not None else "?")
            row += f"{anss+' '+format(cov,'.2f')+' ('+str(len(lst))+')':>22s} "
        print(row, flush=True)
    print("-" * (22 + 23 * len(a.arms)))
    mrow = f"{'MEAN design-coverage':22s} "
    for arm in a.arms:
        m = sum(agg[arm]) / len(agg[arm]) if agg[arm] else 0
        mrow += f"{m:>22.3f} "
    print(mrow)
    # spectrum: how many tasks are NOT 0 or 1 (i.e. partial => richer)?
    for arm in a.arms:
        vals = agg[arm]
        partial = sum(1 for x in vals if 0 < x < 1)
        print(f"  {short(arm)}: n={len(vals)}, partial(0<cov<1)={partial}, ==1.0: {sum(1 for x in vals if x>=0.999)}, distinct values: {sorted(set(round(x,2) for x in vals))}")

if __name__ == "__main__":
    main()
