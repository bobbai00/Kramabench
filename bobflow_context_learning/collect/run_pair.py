#!/usr/bin/env python
"""
Gate-0/contrastive data generation: randomly sample N tasks and run each under
BOTH context arms (latest-core vs full-delta trajectory), reusing evaluate.py /
Benchmark / SessionEvaluator.

Robust runner: ONE evaluate.py per (task, arm) so each gets a fresh login and
one slow task can't block the rest; RESUMABLE (skips a (task,arm) whose
evaluation.json already exists); and a per-run TIMEOUT backstop. Tasks are
processed pair-first (latest then delta for each task) so complete contrast
pairs appear as early as possible.

Usage:
    python bobflow_context_learning/collect/run_pair.py [--n 10] [--seed 7] [--timeout 1500]
"""
import argparse
import json
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKLOAD_DIR = os.path.join(ROOT, "workload")
SCRATCH = os.path.join(ROOT, "system_scratch")
WORKLOADS = ["legal", "environment", "wildfire", "astronomy", "biomedical"]
ARMS = ["DataflowSystemGPT54Gate0Latest", "DataflowSystemGPT54Gate0Delta"]
PY = sys.executable
MANIFEST = os.path.join(ROOT, "bobflow_context_learning", "data", "gate0_manifest.json")


def load_tasks(workload):
    d = json.load(open(os.path.join(WORKLOAD_DIR, f"{workload}.json")))
    tasks = d if isinstance(d, list) else d.get("tasks", [])
    return [(workload, t["id"], t.get("answer_type")) for t in tasks]


def done(arm, task_id):
    return os.path.exists(os.path.join(SCRATCH, arm, task_id, "evaluation.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=1500, help="per (task,arm) subprocess timeout sec")
    args = ap.parse_args()

    pool = []
    for w in WORKLOADS:
        try:
            pool.extend(load_tasks(w))
        except Exception as e:
            print(f"[run_pair] skip {w}: {e}", flush=True)
    random.seed(args.seed)
    sample = random.sample(pool, min(args.n, len(pool)))
    # Run easy tasks first so complete contrast pairs land sooner (incremental analysis).
    sample.sort(key=lambda x: ("hard" in (x[1] or ""), x[0]))

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump({"seed": args.seed, "n": args.n, "arms": ARMS,
               "sample": [{"workload": w, "task_id": t, "answer_type": a} for w, t, a in sample]},
              open(MANIFEST, "w"), indent=2)
    print(f"[run_pair] sampled {len(sample)} tasks (seed={args.seed}); timeout={args.timeout}s/run", flush=True)
    for w, t, _ in sample:
        print(f"    {w:12s} {t}", flush=True)

    for w, tid, _ in sample:
        for arm in ARMS:
            if done(arm, tid):
                print(f"[run_pair] SKIP (done) {arm} / {tid}", flush=True)
                continue
            cmd = [PY, "evaluate.py", "--sut", arm, "--workload", w,
                   "--task_id", tid, "--use_truth_subset", "--no_pipeline_eval", "--verbose"]
            print(f"\n[run_pair] ===== {arm} / {tid} =====", flush=True)
            try:
                r = subprocess.run(cmd, cwd=ROOT, timeout=args.timeout)
                print(f"[run_pair] exit={r.returncode} ({arm}/{tid}) done={done(arm, tid)}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[run_pair] TIMEOUT after {args.timeout}s ({arm}/{tid}) — skipping", flush=True)

    print("\n[run_pair] DONE.", flush=True)


if __name__ == "__main__":
    main()
