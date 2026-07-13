#!/usr/bin/env python3
"""
LakeQA Phase A runner: drive the existing DataflowSystem SUTs over the
stratified 24-task subset (oracle-style: each task is handed its own local
files), judge with LakeQA's own LLM judge, and persist the standard
system_scratch artifact set so all kb.py analysis tooling works unchanged.

Per task the scratch dir (system_scratch/LakeQA_<SUT>/<qid>/) receives the
usual prompt.txt / config.json / react_steps.json / workflow.json /
stats.json / answer.json / response.txt from serve_query, plus
ground_truth.json (via workload/lakeqa.json which this script generates) and
evaluation.json ({"success": 0|1} from the imported LakeQA judge → kb.py
answer_scores reads it through the string_exact -> success mapping).

Usage (combined venv; Texera stack + agent-service must be running):
  ~/Desktop/bobflow/dataflow-agent/.venv/bin/python lakeqa/run_lakeqa.py \
      --sut DataflowSystemGPT52DeltaStats5kD2 --parallel 3
  ... --ids "EQA000229 EQA000758"          # subset of the subset
  ... --skip-done                          # resume: skip tasks with evaluation.json
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
LAKEQA_REPO = Path(os.environ.get("LAKEQA_REPO", Path.home() / "Desktop/bobflow/lakeqa"))
SUBSET_PATH = Path(__file__).resolve().parent / "subset_24.json"
DATASET_DIR = "data/lakeqa/input"

sys.path.insert(0, str(KB))
sys.path.insert(0, str(LAKEQA_REPO))


def load_env():
    """Export OPENAI_API_KEY etc. from ./.env (kb.py convention) for the judge."""
    env_path = KB / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def write_workload(subset):
    """workload/lakeqa.json lets DataflowSystem._load_workload persist
    ground_truth.json through the existing machinery."""
    wl = [{"id": t["question_id"], "query": t["question"], "answer": t["answer"],
           "answer_type": "string_exact", "data_sources": t["datasets_used"]}
          for t in subset]
    out = KB / "workload" / "lakeqa.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(wl, open(out, "w"), indent=1)
    return out


def run_one(sut_name, task, scratch_root):
    """Fresh System instance per task (mirrors the isolate-mode convention)."""
    import systems  # deferred: heavy import, and must happen inside workers too
    from evaluation.llm_judge import judge_task_result  # LakeQA's own judge

    qid = task["question_id"]
    t0 = time.time()
    cls = getattr(systems, sut_name)
    inst = cls(verbose=False, output_dir=str(scratch_root))
    inst.output_dir = str(scratch_root)  # belt-and-suspenders across preset ctors
    os.makedirs(inst.output_dir, exist_ok=True)
    inst.process_dataset(DATASET_DIR)

    inst.serve_query(task["question"], query_id=qid, subset_files=task["datasets_used"])
    # serve_query returns the Benchmark-API dict (explanation/token usage); the
    # parsed answer is persisted to answer.json by the same call.
    answer = ""
    ans_path = scratch_root / qid / "answer.json"
    if ans_path.exists():
        answer = str((json.load(open(ans_path)) or {}).get("answer") or "")

    judge = judge_task_result(
        task={"question": task["question"], "answer": task["answer"]},
        result={"predicted_answer": answer},
    )
    ev = {"success": 1.0 if judge.get("llm_judge_pass") else 0.0, **judge}
    with open(scratch_root / qid / "evaluation.json", "w") as f:
        json.dump(ev, f, indent=2, default=str)
    return qid, ev["success"], answer, task["answer"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", default="DataflowSystemGPT52DeltaStats5kD2")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--ids", default="", help="space-separated question_ids (default: whole subset)")
    ap.add_argument("--skip-done", action="store_true", help="skip tasks that already have evaluation.json")
    ap.add_argument("--task-timeout-min", type=float, default=25.0)
    a = ap.parse_args()

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY missing (judge needs it) — put it in ./.env")

    subset = json.load(open(SUBSET_PATH))
    if a.ids:
        want = set(a.ids.split())
        subset = [t for t in subset if t["question_id"] in want]
    scratch_root = KB / "system_scratch" / f"LakeQA_{a.sut}"
    scratch_root.mkdir(parents=True, exist_ok=True)
    if a.skip_done:
        subset = [t for t in subset if not (scratch_root / t["question_id"] / "evaluation.json").exists()]
    write_workload(json.load(open(SUBSET_PATH)))
    print(f"[lakeqa] {len(subset)} task(s) -> {scratch_root.relative_to(KB)}  (parallel={a.parallel})")

    os.chdir(KB)  # data/... paths in prompts are relative to repo root
    results, t0 = [], time.time()
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        futs = {ex.submit(run_one, a.sut, t, scratch_root): t["question_id"] for t in subset}
        for fut in cf.as_completed(futs):
            qid = futs[fut]
            try:
                qid, ok, ans, gold, dt = fut.result(timeout=a.task_timeout_min * 60)
                results.append((qid, ok))
                print(f"[lakeqa] {qid}  {'PASS' if ok else 'fail'}  {dt/60:.1f}min  "
                      f"got={ans[:80]!r}  want={str(gold)[:60]!r}", flush=True)
            except Exception as e:
                results.append((qid, 0.0))
                print(f"[lakeqa] {qid}  ERROR {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

    n = len(results)
    p = sum(1 for _, ok in results if ok)
    print(f"\n[lakeqa] done in {(time.time()-t0)/60:.0f}min: {p}/{n} passed "
          f"({100*p/max(1,n):.0f}%)  scratch: {scratch_root.relative_to(KB)}")


if __name__ == "__main__":
    main()
