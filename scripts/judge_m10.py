#!/usr/bin/env python3
"""
M10 v3 — per-step judge, 5-way taxonomy, gold-anchored, decision-time faithful.

For-loop over react steps. Judge input per step (Bob's 3-part design):
  1. CONTEXT — what the agent could see at this step (its own inputMessages,
     UNCAPPED). For latest-mode this is the snapshot, so forgetting is visible.
  2. PRIOR ACTIONS — the sequence of previous actions (code only, no results).
  3. THIS STEP — its action + its execution_result (parsed from the trace by
     scripts/extract_step_results.py; delta = next input's new-event
     observation, latest = next snapshot's operator block).
The judge also knows the task, gold subtasks and ground-truth values
(outcome-SIGHTED). Verdicts, exactly one per step:

  useful      right subtask, execution advances (incl. schema/data-dictionary
              checks and verification — due diligence counts)
  wrong_param right subtask, wrong parameter/method -> wrong intermediate
              (bad join key, wrong column, > vs >=, wrong encoding choice)
  thwarted    right subtask, sound approach, result still wrong/failed for
              reasons outside the choice (malformed file surprise, engine
              error, truncation)
  off_task    works on data/columns/computations the solution does not need
              and that do not inform later steps
  redundant   re-attempts something already done or already failed unchanged

Scores per task: fraction per verdict; summary prints task-mean AND
step-pooled. Cache: judge_m10.json (version 3).
Run: .venv/bin/python scripts/judge_m10.py --arms A B [--tasks ...] [--model gpt-4o]
"""
import argparse, json, os, sys, importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
_s = importlib.util.spec_from_file_location("jm", KB / "scripts/judge_m5m6.py")
jm = importlib.util.module_from_spec(_s)
_s.loader.exec_module(jm)

VERDICTS = ["useful", "wrong_param", "thwarted", "off_task", "redundant"]
# NO truncation anywhere (Bob 2026-07-26): full context, full actions, full results.


M10_SYS = (
    "You evaluate ONE step of a data-analysis agent's execution, in sequence. "
    "You know the full task, its reference decomposition (subtasks), and the "
    "ground-truth value of each subtask — the agent does NOT know these.\n\n"
    "You are given: (1) the CONTEXT the agent could see when it acted (this may "
    "omit earlier work — the agent may have forgotten it), (2) the sequence of "
    "its PREVIOUS actions, (3) THIS step's action and the execution result it "
    "produced. Classify the step as exactly one of:\n"
    "  useful      — right subtask, and the execution advances it: loads/locates "
    "needed data, computes or verifies a needed intermediate, or gathers "
    "information that reasonably informs later steps (schema checks, data-"
    "dictionary reads, verification are useful due diligence).\n"
    "  wrong_param — right subtask, but a wrong parameter or method choice "
    "corrupts the result: wrong column, wrong join key, wrong filter boundary, "
    "wrong encoding/format choice the visible evidence argued against.\n"
    "  thwarted    — right subtask and a sound approach given what was visible, "
    "but the result is wrong or failed for reasons outside the choice: "
    "malformed data surprise, engine error, truncated output.\n"
    "  off_task    — works on data, columns or computations the reference "
    "solution does not need and that do not inform later steps.\n"
    "  redundant   — re-attempts something a previous action already did, or "
    "retries an already-failed approach without a meaningful change.\n"
    "Report target_subtask: the id of the reference subtask this step serves "
    "(null if none), and a one-sentence rationale. Judge only THIS step, on the "
    "evidence available to the agent at that point."
)


def m10_schema():
    return {"type": "json_schema", "json_schema": {
        "name": "m10_verdict", "strict": True, "schema": {
            "type": "object", "properties": {
                "verdict": {"type": "string", "enum": VERDICTS},
                "target_subtask": {"type": ["string", "null"]},
                "rationale": {"type": "string"}},
            "required": ["verdict", "target_subtask", "rationale"],
            "additionalProperties": False}}}


def gold_block(task_def):
    lines = [f"Task: {task_def.get('query','')}"]
    lines.append(f"Ground-truth final answer: {json.dumps(task_def.get('answer'), default=str)[:200]}")
    lines.append("Reference subtasks (id: step -> ground-truth value):")
    for s in task_def.get("subtasks", []) or []:
        a = json.dumps(s.get("answer"), ensure_ascii=False, default=str)
        lines.append(f"- {s.get('id')}: {s.get('step','')[:180]}  ->  {a[:200]}")
    return "\n".join(lines)


# --------------------------------------------------------------- step loading

def dataflow_steps(arm, task):
    """[(id, context, action, result)] from step_results.json + react_steps.json."""
    sr = jm.jload(KB / "system_scratch" / arm / task / "step_results.json")
    tr = jm.jload(KB / "system_scratch" / arm / task / "react_steps.json")
    if not sr or not tr:
        return None
    steps = tr.get("steps", [])
    out = []
    for rec in sr["records"]:
        i = rec["step"]
        ctx = None
        if 0 <= i < len(steps) and steps[i].get("inputMessages"):
            ctx = "\n".join(str(m.get("content", "")) for m in steps[i]["inputMessages"])
        out.append((f"step-{i}", ctx, rec["action"], rec["execution_result"]))
    return out


def codeagent_steps(arm, task):
    """Code agent: context = accumulated prior observations (its visible history)."""
    d = jm.jload(KB / "system_scratch" / arm / task / "reasoning_trace.json")
    if not d:
        return None
    out, hist = [], []
    for e in d:
        ctx = "\n\n".join(hist) if hist else None
        code = e.get("code") or ""
        obs = (e.get("observations") or "")
        outp = e.get("output") or ""
        res = obs if not outp else (obs + ("\n" + outp if outp not in obs else ""))
        out.append((f"step-{e.get('step')}", ctx, code, res or None))
        hist.append(f"[step {e.get('step')} observation]\n{obs}")
    return out


# --------------------------------------------------------------------- judge

def judge_step(client, model, gold, ctx, prior_actions, action, result, sid):
    prior = "\n".join(f"[{i+1}] {a}" for i, a in enumerate(prior_actions)) or "(none)"
    user = (
        gold + "\n\n"
        f"CONTEXT visible to the agent at this step:\n"
        f"{ctx if ctx else '[no context recorded]'}\n\n"
        f"PREVIOUS actions (most recent last):\n{prior}\n\n"
        f"THIS step ({sid}):\nAction:\n{action or '[none]'}\n\n"
        f"Execution result:\n{result if result else '[none recorded]'}\n\n"
        "Verdict for THIS step only."
    )
    r = client.chat.completions.create(
        model=model, temperature=0,
        response_format=m10_schema(),
        messages=[{"role": "system", "content": M10_SYS},
                  {"role": "user", "content": user}],
    )
    v = json.loads(r.choices[0].message.content)
    if v.get("verdict") not in VERDICTS:
        v["verdict"] = "off_task"
    return v, (r.usage.total_tokens if r.usage else 0)


def run_task(client, model, W, arm, task, force=False):
    outp = KB / "system_scratch" / arm / task / "judge_m10.json"
    prev = jm.jload(outp)
    if prev and prev.get("version") == 3 and not force:
        return prev
    task_def = W.get(task)
    if not task_def or not task_def.get("subtasks"):
        return None
    if (KB / "system_scratch" / arm / task / "reasoning_trace.json").exists():
        steps = codeagent_steps(arm, task)
    else:
        steps = dataflow_steps(arm, task)
    if not steps:
        return None
    gold = gold_block(task_def)
    per, prior, tokens = {}, [], 0
    for sid, ctx, action, result in steps:
        try:
            v, tok = judge_step(client, model, gold, ctx, prior, action, result, sid)
        except Exception as e:
            print(f"  [M10 ERR {arm}/{task}/{sid}] {e}", file=sys.stderr)
            v, tok = {"verdict": "off_task", "target_subtask": None,
                      "rationale": f"judge error: {e}"}, 0
        tokens += tok
        per[sid] = v
        prior.append(action or "")
    n = len(per)
    cnt = {k: sum(1 for v in per.values() if v["verdict"] == k) for k in VERDICTS}
    stats = jm.jload(KB / "system_scratch" / arm / task / "stats.json") or {}
    res = dict(arm=arm, task=task, judge_model=model, version=3, tokens=tokens,
               n_steps=n, counts=cnt,
               **{f"{k}_frac": cnt[k] / n for k in VERDICTS},
               cost_usd=float(stats.get("cost_usd") or 0), per_step=per)
    tmp = outp.with_suffix(".tmp")
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, outp)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    jm.load_env()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    W = jm.load_workload()

    jobs = []
    for arm in a.arms:
        tasks = a.tasks or sorted(
            d.name for d in (KB / "system_scratch" / arm).iterdir()
            if d.is_dir() and d.name in W
            and ((d / "step_results.json").exists() or (d / "reasoning_trace.json").exists()))
        jobs += [(arm, t) for t in tasks]

    done = {}
    def work(j):
        arm, t = j
        try:
            return (arm, run_task(client, a.model, W, arm, t, force=a.force))
        except Exception as e:
            print(f"  [ERR {arm}/{t}] {e}", file=sys.stderr)
            return (arm, None)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (arm, r) in enumerate(ex.map(work, jobs)):
            if r:
                done.setdefault(arm, []).append(r)
            if (i + 1) % 25 == 0:
                spent = sum(x["tokens"] for rs in done.values() for x in rs)
                print(f"  ...{i+1}/{len(jobs)}  judge-tokens so far: {spent/1e6:.1f}M", flush=True)

    print(f"\n{'arm':52s} {'n':>4s}  " + " ".join(f"{k:>13s}" for k in VERDICTS))
    print(f"{'':52s} {'':4s}  " + " ".join(f"{'task| pooled':>13s}" for _ in VERDICTS))
    print("-" * 130)
    for arm in a.arms:
        rs = done.get(arm, [])
        if not rs:
            print(f"{arm:52s}    0")
            continue
        tot = sum(r["n_steps"] for r in rs)
        row = f"{arm:52s} {len(rs):4d} "
        for k in VERDICTS:
            tmean = sum(r[f"{k}_frac"] for r in rs) / len(rs)
            pooled = sum(r["counts"][k] for r in rs) / tot
            row += f"  {tmean:5.3f}|{pooled:5.3f}"
        print(row)


if __name__ == "__main__":
    main()
