# KramaBench operating notes

This file is the operator's quick reference: which script does what, where the
artifacts land, how a score is computed, and how to find tasks that didn't
pass. It does **not** re-explain what's already in `README.md`,
`systems/DATAFLOW_README.md`, or `benchmark/metrics.py` — read those first
for concepts; come here for "how do I run it / where do I look."

## Prerequisites

- **Python venv** at `.venv/` (Python 3.11; later versions lack wheels for
  pinned `pyarrow==19.0.1`). Create via `uv venv --python 3.11 && uv pip install -r requirements.txt`.
- **`OPENAI_API_KEY`** must be exported in any shell that runs the harness.
  The scoring metrics `llm_paraphrase` (string_approximate) and
  `f1_approximate` (list_approximate) call `gpt-4o-mini` even when
  `--no_pipeline_eval` is passed; `benchmark/benchmark.py:224` instantiates
  the OpenAI client unconditionally.
- **Texera services on localhost** (only for DataflowSystem variants):
  - `8080` — backend
  - `8888` — computing unit
  - `3001` — agent service (which itself routes LLM calls through the
    LiteLLM gateway configured in `~/Desktop/dataflow-agent/agent-service/.env`)
- **Wildfire Kaggle files** (`WeatherEvents_Jan2016-Dec2022.csv`, `ZHVI.csv`)
  are not in the repo. Fetch with `python data/wildfire/input/load_data.py`
  (needs `kagglehub` installed and a Kaggle token in
  `~/.kaggle/access_token`).

## Script index

| Script | Purpose | Typical invocation |
|---|---|---|
| `run_dataflow_system.sh` | Run a DataflowSystem variant over full workloads (creates fresh scratch + bulk cache). | `SUT=DataflowSystemGPT5Mini PARALLEL=true ./run_dataflow_system.sh` |
| `run_dataflow_tasks.sh` | Re-run specific tasks (writes per-task scratch but **not** the bulk cache). | `SUT=DataflowSystemGPT5Mini ./run_dataflow_tasks.sh legal-hard-2 environment-hard-20` |
| `evaluate_dataflow_system.sh` | Rebuild bulk cache from per-task scratch, then re-score via `evaluate.py --use_system_cache`. Use this after any partial rerun. | `WORKLOADS_OVERRIDE="legal environment" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh` |
| `evaluate_dataflow_tasks.sh` | Re-score specific tasks only (hardcoded to a single SUT — edit per use). | `./evaluate_dataflow_tasks.sh legal legal-hard-1 legal-hard-3` |
| `evaluate.py` | The benchmark harness. Every script above ultimately calls this. | `python evaluate.py --sut <SUT> --workload <domain> --no_pipeline_eval --verbose [--use_truth_subset] [--use_system_cache] [--task_id ID …]` |
| `scripts/list_failed_tasks.py` | Group every non-passing task by workload, with a reason (execution_error / empty_answer / wrong_answer / partial / missing_eval). | `python scripts/list_failed_tasks.py --sut <SUT>` |
| `compute_scores.py` | Aggregate per-task CSV → leaderboard-style numbers. | `python compute_scores.py` |

`run_dataflow_system.sh` has env knobs:
- `SUT` — class name from `systems/__init__.py` (e.g. `DataflowSystemHaiku45`, `DataflowSystemGPT5Mini`).
- `ORACLE_MODE=true|false` — adds `--use_truth_subset` so the SUT only sees the
  files listed in each task's `data_sources` (easier than the full lake).
- `PARALLEL=true|false` — fan out workloads to background processes vs.
  sequential.
- Hardcoded `WORKLOADS=(…)` array near the top — edit to scope a run.

## DataflowSystem variants

Defined in `systems/dataflow_system.py` and exported from
`systems/__init__.py`. Context is controlled by **`context_mode`** (latest/delta)
+ the two ordinal DECORATE levels **`flow_level`** / **`data_level`** (0–3 each;
see `claude/CONTEXT-DESIGN.md §8b`). The lean canonical set:

| Class | Model | Config |
|---|---|---|
| `DataflowSystem` | base (`claude-haiku-4.5`) | bare (flow_level=0, data_level=0) |
| `DataflowSystemHaiku45` | `claude-haiku-4.5` | bare |
| `DataflowSystemGPT5Mini` | `gpt-5-mini` | bare |
| `DataflowSystemLocalLlm` | `local-llm` (local-react) | bare, max_steps=20 |
| `DataflowSystemGPT52LatestSchemaConverge` | `gpt-5.2` | converge: flow_level=1, data_level=1 (thesis arm) |
| `DataflowSystemGPT5MiniLatestSchemaConverge` | `gpt-5-mini` | converge: flow_level=1, data_level=1 (thesis arm) |
| `DataflowSystemGPT5MiniLatestSchemaConvergeTableStruct` | `gpt-5-mini` | converge + **data_level=2** (structural-profile, the accuracy win) |
| `DataflowSystemGPT5MiniLatestSchemaConvergeFewShot` | `gpt-5-mini` | converge + few-shot prior (W2 cost win) |
| `DataflowSystemGPT5MiniLatestSchemaConvergeCap20` | `gpt-5-mini` | converge + max_steps=20 (#31 step-cap) |
| `DataflowSystemGPT5MiniLatestSchemaConvergeLevels` | `gpt-5-mini` | reference level config: flow_level=2, data_level=2 |

The 30+ one-off A/B + stats-sweep variants from the campaign were removed (their
results live in `claude/seed1/` + `system_scratch/` by SUT name, which the
analysis tools read by string — unaffected by the class pruning).

Add a new variant by subclassing `DataflowSystem`, passing `model_type`, `name`,
`context_mode`, and `flow_level`/`data_level` (plus any ACT-side knobs:
`max_steps`, `max_loaders_per_source`, `attempt_reflection`). Re-export from
`systems/__init__.py` so the dynamic lookup
(`getattr(systems_module, system_name)`) can find it.

## How evaluation works end to end

1. **`evaluate.py`** loads `workload/{domain}.json`, instantiates the SUT,
   and per task calls `system.serve_query(query, query_id, subset_files)`.
2. For DataflowSystem the SUT writes a **per-task scratch dir** under
   `system_scratch/{SUT}/{task_id}/`:
   - `prompt.txt`, `config.json` — what was sent
   - `response.txt`, `react_steps.json`, `workflow.json` — what came back
   - `stats.json` — token usage, runtime, num steps
   - `answer.json` — `{"id": "main-task", "answer": <parsed final answer>}`
   - `ground_truth.json` — the workload's expected answer (when known)
   - `evaluation.json` — added by `evaluate.py:158-166` after metric scoring
3. After all tasks run, the harness writes the **bulk response cache**:
   `results/{SUT}/response_cache/{workload}_{ts}.json` — a JSON list, one
   entry per task in workload order. Used by `--use_system_cache` to skip
   re-running the SUT. **`evaluate.py --task_id` mode never writes this
   cache** (`benchmark.py:177`), which is why partial reruns need
   `evaluate_dataflow_system.sh` to rebuild it before re-scoring.
4. **Scoring** happens via `Evaluator._evaluate_result_for_task` in
   `benchmark/benchmark.py`. Each task's `answer_type` maps to a set of
   metrics (`benchmark/fixtures/answer_type_fixtures.json`):
   - `numeric_exact` / `string_exact` → `success`
   - `numeric_approximate` → `mean_absolute_error`, `mean_squared_error`, `rae_score`
   - `string_approximate` → `llm_paraphrase` (calls gpt-4o-mini)
   - `list_exact` → `f1`, `precision`, `recall`
   - `list_approximate` → `f1_approximate` (gpt-4o-mini for strings, ≤1% rel err for numbers)
5. **Per-task results** land in `results/{SUT}/{workload}_measures_{ts}.csv`
   (one row per `(task_id, metric)`).
6. **Aggregation** in `evaluate.py:11-61` rolls up to
   `results/aggregated_results.csv` keyed by `(sut, workload, metric)`. The
   printed `Total score is: …` divides
   `sum(value_support * value_mean)` over the **headline metrics** —
   `success`, `llm_paraphrase`, `rae_score`, `f1`, `f1_approximate` — by the
   total task count. That's the number the README leaderboard shows.

`precision`, `recall`, `mean_absolute_error`, and `mean_squared_error` are
recorded but **do not** contribute to the headline percentage.

## Standard workflows

### Fresh full run
```bash
export OPENAI_API_KEY=sk-…
SUT=DataflowSystemGPT5Mini PARALLEL=true ./run_dataflow_system.sh
# Per-task scratch + bulk caches + measures CSVs + aggregated_results.csv all written.
```

### Partial rerun (a few failed tasks)
```bash
SUT=DataflowSystemGPT5Mini ./run_dataflow_tasks.sh legal-hard-2 environment-hard-20
# Per-task scratch is rewritten; bulk cache is NOT.

WORKLOADS_OVERRIDE="legal environment" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh
# Rebuilds bulk cache from current scratch, re-scores both workloads.
```

The cache rebuilder treats `answer.json` as the source of truth for the
SUT's final answer (see comments in `evaluate_dataflow_system.sh`). Tasks
with a scratch dir but no `answer.json` (e.g. a run killed mid-task) are
emitted as a stub entry with `"answer": ""`, which preserves
workload-to-cache index alignment so the harness can iterate cleanly.

### Re-score only (no SUT calls)
```bash
WORKLOADS_OVERRIDE="legal" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh
# Useful after editing metrics, or after fixing per-task answer.json by hand.
```

## Finding failed tasks

`scripts/list_failed_tasks.py` is the canonical inspector. It cross-refs
three places for each top-level task in a workload:

- the latest `results/{SUT}/{workload}_measures_*.csv` for the score
- `workload/{workload}.json` for the task's `answer_type` (selects which
  metric is "primary": `success` / `rae_score` / `llm_paraphrase` / `f1` /
  `f1_approximate`)
- `system_scratch/{SUT}/{task_id}/answer.json` and `response.txt` for
  execution-vs-answer signals

Reasons it emits:

| Reason | Trigger |
|---|---|
| `execution_error` | `answer.json` starts with `"Error:"` (SUT raised in `serve_query`) |
| `empty_answer` | `response.txt` empty or `(empty response)` (agent gave up; e.g. legal-hard-2's "No response from agent") |
| `wrong_answer` | primary score == 0 with a real answer |
| `partial` | 0 < primary score < threshold |
| `missing_eval` | task is in workload but no measures row exists |

Examples:
```bash
# every failure for a SUT across every workload that has results
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini

# one workload at a time
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --workload legal

# only true execution failures (skip "answer was just wrong")
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --errors-only

# loosen the threshold — show only tasks scoring below 0.5
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --threshold 0.5

# machine-readable, grouped by workload
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --json
```

To rerun every failed task (extract IDs into a list):
```bash
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --json \
  | python -c "import json,sys; d=json.load(sys.stdin); print(' '.join(t['task_id'] for w in d.values() for t in w))" \
  | xargs -n50 ./run_dataflow_tasks.sh
# …then re-score:
WORKLOADS_OVERRIDE="legal environment" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh
```

## Where everything lives

```
data/{domain}/input/             raw data lake the SUT reads
workload/{domain}.json           tasks (id, query, answer, answer_type, data_sources, subtasks)
benchmark/fixtures/              answer_type → metrics mapping, etc.
systems/                         SUT implementations + dataflow agent client
system_scratch/{SUT}/{task_id}/  per-task SUT artifacts (audit trail)
results/{SUT}/
  response_cache/                bulk SUT responses; reused via --use_system_cache
  {workload}_measures_*.csv      per-(task,metric) scores
results/aggregated_results.csv   per-(sut,workload,metric) rollup
logs/{ts}/{workload}.log         stdout/stderr of background runs
```

## Gotchas seen in practice

- **Bash 3.2 on macOS** lacks `declare -A`, so `run_dataflow_tasks.sh`
  needs bash 4+ for the new "parse workload from task IDs" mode. Fall back
  to legacy mode (`./run_dataflow_tasks.sh <workload> <task_ids…>`) or run
  `evaluate.py --task_id …` directly when on stock macOS bash.
- **Stuck `GENERATING` agents.** The agent's WebSocket `sendMessage` has
  no client-side timeout in `dataflow_agent.py`. If a GPT-5-mini run sits
  in `GENERATING` for >10 min with 0 committed steps, `DELETE
  /api/agents/<id>` won't free the harness's open WS — you have to
  `kill -TERM <python pid>` and use the partial-rerun workflow above to
  recover.
- **OpenAI key not in `.env`** by default. `evaluate.py` doesn't call
  `load_dotenv()`. Either `export OPENAI_API_KEY=…` in the shell or add a
  `dotenv` load at the top of `evaluate.py`.
- **`legal-tiny` quickstart** references `data/legal/tiny/`, which doesn't
  exist. Use `--workload legal` against the full input directory instead.
