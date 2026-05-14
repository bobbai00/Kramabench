#!/bin/bash
# Re-evaluate all workloads from whatever currently lives in system_scratch/<SUT>/
#
# Workflow this supports:
#   1. ./run_dataflow_tasks.sh <task_ids>     -> re-runs the SUT for those tasks,
#                                                overwriting per-task scratch
#                                                (prompt/response/answer/evaluation.json/...)
#   2. ./evaluate_dataflow_system.sh          -> rebuilds the bulk response_cache from
#                                                per-task scratch, then runs the harness
#                                                with --use_system_cache so it picks up
#                                                the freshly merged data
#
# Why the cache rebuild step is needed:
#   evaluate.py's --task_id mode never writes the bulk response_cache file
#   (benchmark.py:177: `if cache_system_output and not self.task_id_filter`),
#   so partial reruns leave the cache stale.  Reading per-task evaluation.json
#   and rewriting the bulk cache before re-running the harness keeps every
#   downstream score (results/aggregated_results.csv, compute_scores.py) in
#   sync with whatever was last written to system_scratch.

set -euo pipefail

SUT=${SUT:-DataflowSystemHaiku45}
WORKLOADS=("archeology" "astronomy" "biomedical" "environment" "legal" "wildfire")

# Use the project venv if it exists so the script works even when the caller
# hasn't activated it (e.g. background watchers).  Fall back to whatever
# `python` resolves to on PATH otherwise.
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python"
fi

SCRATCH_DIR="system_scratch/$SUT"
CACHE_DIR="results/$SUT/response_cache"

if [ ! -d "$SCRATCH_DIR" ]; then
    echo "ERROR: $SCRATCH_DIR not found. Run ./run_dataflow_tasks.sh first." >&2
    exit 1
fi

echo "=========================================="
echo "Step 1/2: rebuild bulk response_cache from per-task scratch"
echo "=========================================="
mkdir -p "$CACHE_DIR"
"$PYTHON" - "$SUT" "${WORKLOADS[@]}" <<'PY'
# Build the bulk response_cache from the freshest data on disk for each task.
#
# A partial rerun (./run_dataflow_tasks.sh some_task_id) writes
#   answer.json, response.txt, react_steps.json, workflow.json, stats.json
# IMMEDIATELY after the agent finishes each task, but it does NOT rewrite
# evaluation.json — that file is only updated by evaluate.py after the
# full workload's metric pass finishes.  So if a rerun is killed mid-flight
# (or is just one task short of the workload's task count), evaluation.json
# is stale and no longer reflects the agent's latest answer.
#
# Fix: for each task, if answer.json is newer than evaluation.json, treat
# answer.json as the source of truth for model_output.  Other fields
# (runtime, token usage) come from stats.json when present, else
# evaluation.json.  This makes "rerun a few tasks, then re-evaluate all"
# work end-to-end without any benchmark.py changes.
import os, sys, json, datetime
sut = sys.argv[1]
workloads = sys.argv[2:]
scratch = f"system_scratch/{sut}"
cache_dir = f"results/{sut}/response_cache"
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def build_entry(task_dir, tid):
    """Return a cache entry for one task, preferring fresh agent output (answer.json/stats.json) over stale evaluation.json."""
    ev_path = os.path.join(task_dir, "evaluation.json")
    ans_path = os.path.join(task_dir, "answer.json")
    stats_path = os.path.join(task_dir, "stats.json")

    ev = json.load(open(ev_path)) if os.path.exists(ev_path) else {}
    entry = {
        "task_id": tid,
        "model_output": ev.get("model_output", {"id": "main-task", "answer": ""}),
        "code": ev.get("code", ""),
        "token_usage_sut": ev.get("token_usage_sut", 0),
        "token_usage_sut_input": ev.get("token_usage_sut_input", 0),
        "token_usage_sut_output": ev.get("token_usage_sut_output", 0),
        "runtime": ev.get("runtime", 0.0),
    }

    # ALWAYS prefer answer.json when it exists, regardless of mtime.
    # answer.json is only ever written by the SUT (dataflow_system.serve_query
    # via parse_answer) when the agent actually runs.  --use_system_cache
    # flows ONLY into evaluation.json::model_output, never into answer.json,
    # so answer.json is always the canonical record of what the agent
    # actually parsed-and-emitted for that task.  Conversely, repeated
    # evaluate_dataflow_system.sh runs can poison evaluation.json::model_output
    # with values from a stale bulk cache (from before the most recent
    # rerun), even when those overwrites are themselves recent.
    used_answer_json = False
    if os.path.exists(ans_path):
        ans = json.load(open(ans_path))
        # answer.json schema: {"id": "main-task", "answer": ...}
        entry["model_output"] = {"id": ans.get("id", "main-task"), "answer": ans.get("answer", "")}
        used_answer_json = True
        # Also pick up fresh runtime/token usage from stats.json when present
        if os.path.exists(stats_path):
            stats = json.load(open(stats_path))
            for k in ("token_usage_sut","token_usage_sut_input","token_usage_sut_output","runtime"):
                if k in stats:
                    entry[k] = stats[k]
                elif k.replace("_sut","") in stats:
                    entry[k] = stats[k.replace("_sut","")]
    return entry, used_answer_json

for w in workloads:
    workload_path = f"workload/{w}.json"
    if not os.path.exists(workload_path):
        print(f"  [skip] {w}: no workload file at {workload_path}")
        continue
    task_order = [t["id"] for t in json.load(open(workload_path))]

    fresh, missing, refreshed_from_answer = [], [], []
    for tid in task_order:
        td = f"{scratch}/{tid}"
        if not os.path.isdir(td):
            missing.append(tid); continue
        entry, used_answer_json = build_entry(td, tid)
        fresh.append(entry)
        if used_answer_json:
            refreshed_from_answer.append(tid)

    out = f"{cache_dir}/{w}_{ts}.json"
    with open(out, "w") as f:
        json.dump(fresh, f, indent=2)
    note = f"  (took {len(refreshed_from_answer)} answers from answer.json)"
    if missing:
        note += f"  MISSING {len(missing)}: {missing}"
    print(f"  {w}: {len(fresh)}/{len(task_order)} tasks -> {out}{note}")
PY

echo
echo "=========================================="
echo "Step 2/2: run harness with --use_system_cache (reads the cache we just wrote)"
echo "=========================================="
for workload in "${WORKLOADS[@]}"; do
    echo "------------------------------------------"
    echo "Evaluating: $SUT | $workload"
    echo "------------------------------------------"
    "$PYTHON" evaluate.py --sut "$SUT" --workload "$workload" --no_pipeline_eval --verbose --use_system_cache
    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "results/aggregated_results.csv now reflects the latest per-task scratch."
echo "=========================================="
