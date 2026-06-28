#!/usr/bin/env python
"""
Contrastive data generation (batch / parallel).

Picks HARD tasks balanced across all six KramaBench domains and runs each under
BOTH context arms (latest-core vs full-delta trajectory) via evaluate.py, reusing
Benchmark / SessionEvaluator for scoring.

Differences from run_pair.py:
  * domain-balanced HARD-only selection (per_domain tasks from EACH workload)
    instead of a flat random sample that ignores difficulty and one domain;
  * PARALLEL: a thread pool runs up to --workers (task, arm) evaluate.py
    subprocesses at once (the wall-clock cost is dominated by gpt-5.4 latency,
    which overlaps cleanly across processes), instead of one-at-a-time.

Shares run_pair.py's robustness: ONE evaluate.py per (task, arm) so each gets a
fresh login and one slow task can't block the rest; RESUMABLE (skips a (task,arm)
whose evaluation.json already exists); per-run TIMEOUT backstop.

Usage:
    python bobflow_context_learning/collect/run_batch.py \
        [--per-domain 5] [--seed 7] [--workers 6] [--timeout 1800]
"""
import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKLOAD_DIR = os.path.join(ROOT, "workload")
SCRATCH = os.path.join(ROOT, "system_scratch")
DOMAINS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]
DEFAULT_ARMS = ["DataflowSystemGPT54Gate0Latest", "DataflowSystemGPT54Gate0Delta"]


def _resolve_python():
    """evaluate.py needs the Kramabench venv (pandas etc.); the bare `python3`
    that launches this script may not have it. Prefer the repo venv."""
    cand = os.path.join(ROOT, ".venv", "bin", "python")
    return cand if os.path.exists(cand) else sys.executable


PY = _resolve_python()
MANIFEST = os.path.join(ROOT, "bobflow_context_learning", "data", "gate0_manifest.json")

_print_lock = Lock()


def say(*a):
    with _print_lock:
        print(*a, flush=True)


def load_tasks(domain, hard_only):
    d = json.load(open(os.path.join(WORKLOAD_DIR, f"{domain}.json")))
    tasks = d if isinstance(d, list) else d.get("tasks", [])
    return [(domain, t["id"], t.get("answer_type")) for t in tasks
            if (not hard_only) or "hard" in t.get("id", "")]


def done(arm, task_id):
    return os.path.exists(os.path.join(SCRATCH, arm, task_id, "evaluation.json"))


def run_one(domain, task_id, arm, timeout):
    if done(arm, task_id):
        say(f"[run_batch] SKIP (done) {arm} / {task_id}")
        return (task_id, arm, "skip")
    cmd = [PY, "evaluate.py", "--sut", arm, "--workload", domain,
           "--task_id", task_id, "--use_truth_subset", "--no_pipeline_eval", "--verbose"]
    say(f"[run_batch] START {arm} / {task_id}")
    try:
        r = subprocess.run(cmd, cwd=ROOT, timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = done(arm, task_id)
        say(f"[run_batch] END   {arm} / {task_id}  exit={r.returncode} produced_eval={ok}")
        return (task_id, arm, "ok" if ok else "noeval")
    except subprocess.TimeoutExpired:
        say(f"[run_batch] TIMEOUT after {timeout}s ({arm}/{task_id})")
        return (task_id, arm, "timeout")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6, help="concurrent (task,arm) subprocesses")
    ap.add_argument("--timeout", type=int, default=1800, help="per (task,arm) subprocess timeout sec")
    ap.add_argument("--all-tasks", action="store_true",
                    help="run EVERY task in every domain (ignore --per-domain and the hard filter)")
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                    help="SUT class names to run per task (default: Gate-0 pair)")
    args = ap.parse_args()
    arms = args.arms

    rng = random.Random(args.seed)
    sample = []
    for dom in DOMAINS:
        try:
            pool_tasks = load_tasks(dom, hard_only=not args.all_tasks)
        except Exception as e:
            say(f"[run_batch] skip {dom}: {e}")
            continue
        pool_tasks.sort(key=lambda x: x[1])  # deterministic before sampling
        if args.all_tasks:
            pick = pool_tasks
        else:
            pick = rng.sample(pool_tasks, min(args.per_domain, len(pool_tasks)))
        sample.extend(pick)
        say(f"[run_batch] {dom:12s} {len(pick)} tasks picked (of {len(pool_tasks)})")

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump({"seed": args.seed, "per_domain": args.per_domain, "all_tasks": args.all_tasks, "arms": arms,
               "sample": [{"workload": w, "task_id": t, "answer_type": a} for w, t, a in sample]},
              open(MANIFEST, "w"), indent=2)
    say(f"\n[run_batch] {len(sample)} tasks x {len(arms)} arms = {len(sample)*len(arms)} runs; "
        f"arms={arms}; workers={args.workers}; timeout={args.timeout}s/run")
    for w, t, _ in sample:
        say(f"    {w:12s} {t}")

    # Build the full (task, arm) job list. Latest and delta are independent jobs.
    jobs = [(w, t, arm) for (w, t, _) in sample for arm in arms]
    say(f"\n[run_batch] dispatching {len(jobs)} jobs ...\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, w, t, arm, args.timeout): (t, arm) for (w, t, arm) in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
            n_done = sum(1 for _, _, s in results if s in ("ok", "skip"))
            say(f"[run_batch] progress {len(results)}/{len(jobs)} finished, {n_done} have eval")

    from collections import Counter
    say(f"\n[run_batch] DONE. outcomes: {dict(Counter(s for _, _, s in results))}")


if __name__ == "__main__":
    main()
