#!/usr/bin/env python3
"""
M3 — LLM correctness-anchored per-step judge.

The decisive test: can a semantic judge catch the interpretation coins
(L2-vs-L box, denominator choice) that token/value metrics miss? For each gold
subtask we give the judge the step description, its CORRECT result, and the
agent's pipeline code, and ask whether the pipeline would produce that result
for that step (wrong column/filter/metric/grain -> No). Correctness-anchored
and per-step, so stricter than the built-in presence judge that saturated.

score(task) = fraction of steps judged correctly implemented.

Run: .venv/bin/python scripts/m3_semantic_step_judge.py [--model gpt-4o-mini]
"""
import argparse, os, re, json, importlib.util
from pathlib import Path
KB = Path(__file__).resolve().parent.parent
for line in open(KB / ".env"):
    line = re.sub(r"^export\s+", "", line.strip())
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from openai import OpenAI
m1 = importlib.util.module_from_spec(importlib.util.spec_from_file_location("m1", KB / "scripts/m1_plan_coverage.py"))
importlib.util.spec_from_file_location("m1", KB / "scripts/m1_plan_coverage.py").loader.exec_module(m1)
v = importlib.util.module_from_spec(importlib.util.spec_from_file_location("v", KB / "scripts/validate_metrics.py"))
importlib.util.spec_from_file_location("v", KB / "scripts/validate_metrics.py").loader.exec_module(v)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
SYS = ("You evaluate whether a data pipeline correctly implements ONE step of a data-science task. "
       "You are given the step, the CORRECT result of that step, and the pipeline code. "
       "Answer 'Yes' only if the pipeline implements this step so it would produce the correct result — "
       "the right columns, filters, method/metric, and grain. If the pipeline makes a different choice "
       "(wrong column, missing or different filter, wrong distance/aggregation/denominator, wrong grain) that "
       "would change this step's result, answer 'No'. Reply with exactly one word: Yes or No.")

def judge(step, gold, code, model):
    ans_s = str(gold)
    if len(ans_s) > 300: ans_s = ans_s[:300] + "..."
    if len(code) > 8000: code = code[:8000]
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": f"Step: {step}\nCorrect result of this step: {ans_s}\n\nPipeline code:\n{code}\n\nDoes the pipeline correctly implement THIS step? Yes or No."}]
    try:
        r = client.chat.completions.create(model=model, messages=msg, temperature=0)
        t = r.choices[0].message.content.strip().lower()
        return 1 if t.startswith("y") else 0
    except Exception as e:
        return None

def score_task(sut, task_id, task_def, model):
    code = m1.agent_code(sut, task_id)
    if code is None: return None
    hits = tot = 0
    for st in task_def.get("subtasks", []):
        r = judge(st.get("step", ""), st.get("answer"), code, model)
        if r is not None: hits += r; tot += 1
    return hits / tot if tot else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt-4o-mini"); a = ap.parse_args()
    W = m1.load_workload()
    cache = {}
    def cov(arm, task):
        k = (arm, task)
        if k not in cache: cache[k] = score_task(v.ARM[arm], task, W[task], a.model)
        return cache[k]
    print(f"judge model = {a.model}")
    print(f"{'flip':28s} {'verdict':11s} {'M3 w':>6s} {'M3 l':>6s} {'gap':>7s}")
    att, chr_ = [], []
    for task, comp, win, lose, verdict in v.GT:
        if task not in W: continue
        w, l = cov(win, task), cov(lose, task)
        if w is None or l is None: continue
        g = w - l; (att if verdict == "ATTRIBUTED" else chr_).append(g)
        mark = " <--" if verdict == "ATTRIBUTED" else ""
        print(f"{comp+' '+task:28s} {verdict:11s} {w:6.2f} {l:6.2f} {g:+7.2f}{mark}")
    import statistics as st
    print(f"\nM3 attribution: ATTRIBUTED gaps={[round(x,2) for x in att]} mean {st.mean(att):+.3f}")
    print(f"  chronic: mean {st.mean(chr_):+.3f}, mean|gap| {st.mean([abs(x) for x in chr_]):.3f}, max|gap| {max(abs(x) for x in chr_):.3f}")
    print(f"  signal/noise ratio = {st.mean(att)/st.mean([abs(x) for x in chr_]):.2f}")

if __name__ == "__main__":
    main()
