# Onboarding: running KramaBench with the DataflowSystem

This is a hands-on getting-started guide for someone new to this fork. It walks
you from a clean checkout to a scored benchmark run with a Texera-backed
DataflowSystem, and shows how to add your own system variant.

It complements the three reference docs already in the repo — read them when you
want the *concepts*; come here for the *first run*:

- `README.md` — what KramaBench is, the leaderboard, task structure.
- `systems/DATAFLOW_README.md` — the DataflowSystem agent and its options.
- `CLAUDE.md` — the operator's quick reference (scripts, artifacts, scoring,
  finding failed tasks). This onboarding doc is the on-ramp to that.

---

## 1. What KramaBench is (30-second version)

KramaBench is a benchmark for **end-to-end data-science agents**. Each task hands
the system a natural-language question plus a lake of raw data files, and the
system must build a *complete* pipeline — load → clean → transform → compute — to
produce the final answer. Because the benchmark ships ground-truth pipelines, it
scores both the final answer and (optionally) the intermediate steps.

Tasks are grouped into six **domains** (a.k.a. workloads):
`archeology`, `astronomy`, `biomedical`, `environment`, `legal`, `wildfire`.

A single task looks like this (`workload/{domain}.json`):

```json
{
  "id": "legal-hard-1",
  "query": "What is the average ...?",
  "answer": "...",
  "answer_type": "numeric_approximate",
  "data_sources": ["file_a.csv", "file_b.csv"],
  "subtasks": [ ... ]
}
```

The `answer_type` decides which metric scores the task — see
[§7 Scoring](#7-how-scoring-works).

---

## 2. Prerequisites

### 2.1 Python environment

Use **Python 3.11** — later versions lack wheels for the pinned
`pyarrow==19.0.1`.

```bash
# from the repo root
uv venv --python 3.11
uv pip install -r requirements.txt
```

(Plain `venv` + `pip` works too: `python3.11 -m venv .venv &&
source .venv/bin/activate && pip install -r requirements.txt`.)

The shell scripts auto-discover `.venv/bin/python`, so you don't have to
activate the venv to use them. Override with `PYTHON=/path/to/python` if needed.

### 2.2 `OPENAI_API_KEY` is mandatory

Export it in **any** shell that runs the harness:

```bash
export OPENAI_API_KEY=sk-…
```

This is required even with `--no_pipeline_eval`: the scoring metrics
`llm_paraphrase` (for `string_approximate`) and `f1_approximate` (for
`list_approximate`) call `gpt-4o-mini` unconditionally. `evaluate.py` does **not**
call `load_dotenv()`, so a `.env` file alone won't be picked up — export it or
add a dotenv load yourself. If you have no key and are only running tasks whose
answer types don't need the LLM metrics, a dummy key (`export
OPENAI_API_KEY=sk-dummy`) avoids the client crashing at import.

### 2.3 Texera services (DataflowSystem only)

DataflowSystem variants are thin clients that talk to a running Texera stack on
`localhost`:

| Service | Default URL | Role |
|---|---|---|
| Backend | `http://localhost:8080` | Texera backend API |
| Computing unit | `http://localhost:8888` | Executes workflow operators |
| Agent service | `http://localhost:3001` | Drives the LLM (routes through the LiteLLM gateway configured in the agent-service `.env`) |

Quick liveness check:

```bash
curl http://localhost:3001/api/agents/
```

The endpoints are defined at the top of `dataflow_agent.py`
(`TEXERA_API_ENDPOINT`, `TEXERA_COMPUTING_UNIT_ENDPOINT`,
`TEXERA_AGENT_SERVICE_ENDPOINT`) if you need to point at non-default hosts.

> If you only want to run a non-Dataflow baseline (e.g. `ExampleBaselineSystem`
> or a `CodeAgentSystem`), you can skip Texera entirely.

---

## 3. Downloading the data

Most domains ship their input lake in the repo under `data/{domain}/input/`. Two
datasets are **too large or license-restricted** to commit and must be fetched:

### 3.1 Wildfire (Kaggle)

`WeatherEvents_Jan2016-Dec2022.csv` and `ZHVI.csv` are pulled from Kaggle:

```bash
# needs: pip install kagglehub, plus a Kaggle token at ~/.kaggle/access_token
python data/wildfire/input/load_data.py
```

It downloads the `sobhanmoosavi/us-weather-events` and
`robikscube/zillow-home-value-index` datasets and prints their cache paths; move
the CSVs into `data/wildfire/input/`.

### 3.2 Astronomy (Space-Track TLE history)

`data/astronomy/input/download_tle.py` fetches historical orbital data from
space-track.org. Open it and set your own credentials / NORAD IDs before running
— do **not** rely on any hardcoded account in the file.

### 3.3 Hugging Face mirror

The full dataset is also published at
[`eugenie-y/KramaBench`](https://huggingface.co/datasets/eugenie-y/KramaBench):

```python
from datasets import load_dataset
kramabench = load_dataset("eugenie-y/kramabench")
```

After fetching, you should have raw files under every `data/{domain}/input/` you
intend to run. The workload JSON in `workload/{domain}.json` is already in the
repo.

---

## 4. Your first run

The fastest path is the wrapper script. It defaults to `ORACLE_MODE=true`, which
adds `--use_truth_subset` so the agent only sees the files each task lists in
`data_sources` (smaller, easier than the full lake).

```bash
export OPENAI_API_KEY=sk-…

# one domain, sequentially, with the default SUT (DataflowSystemHaiku45)
WORKLOADS_OVERRIDE="legal" ./run_dataflow_system.sh
```

To run the harness directly (one workload, no wrapper):

```bash
.venv/bin/python evaluate.py \
  --sut DataflowSystemHaiku45 \
  --workload legal \
  --no_pipeline_eval \
  --verbose \
  --use_truth_subset
```

`--no_pipeline_eval` skips the GPT-based code-evaluation pass. DataflowSystem
emits dataflow *workflows*, not Python, so pipeline eval doesn't apply — always
pass it for Dataflow variants.

When the run finishes you'll have:

```
results/{SUT}/{workload}_measures_{ts}.csv   # one row per (task, metric)
results/aggregated_results.csv               # rollup; "Total score" prints to stdout
system_scratch/{SUT}/{task_id}/              # full per-task audit trail
```

---

## 5. What a DataflowSystem is

A `DataflowSystem` (in `systems/dataflow_system.py`) is a **System Under Test
(SUT)**: the benchmark calls `system.serve_query(query, query_id, subset_files)`
once per task. The class itself is a thin client — it packages the query plus a
config payload and hands it to the **Texera agent service**, which runs an LLM
agent that builds and executes a dataflow workflow, then returns the final answer.

Per task, the SUT writes an audit trail to `system_scratch/{SUT}/{task_id}/`:

| File | Contents |
|---|---|
| `prompt.txt`, `config.json` | what was sent |
| `response.txt`, `react_steps.json`, `workflow.json` | what came back |
| `stats.json` | token usage, runtime, step count |
| `answer.json` | `{"id": "main-task", "answer": <parsed final answer>}` |
| `ground_truth.json` | expected answer (when known) |
| `evaluation.json` | the metric scores (added after scoring) |

### Key configuration knobs

All behavior is set by **constructor arguments** (not env vars) — subclasses pin
the values. The ones you'll touch most:

| Arg | Meaning |
|---|---|
| `model_type` | Model id passed straight to the agent service (e.g. `claude-haiku-4.5`, `gpt-5-mini`, `local-llm`). Default `claude-haiku-4.5`. |
| `driver` | Agent driver. `None` lets the agent service auto-derive it from `model_type`; `local-react` is the text-mode driver for local models. |
| `max_steps` | Max agent steps per query (default 50). |
| `context_mode` | Context selection policy: `latest` (default) or `delta`. |
| `flow_level` / `data_level` | Ordinal 0–3 context-decoration levels (the `CONTEXT-DESIGN` notes referenced in `CLAUDE.md`). Higher = more pipeline/schema context injected. |
| `tool_dialect` | Tool-call format for `local-react`: `qwen-xml` (default) or `react-text`. Ignored by native tool-calling drivers. |
| `max_operator_result_char_limit` / `…cell_char_limit` | Truncation limits on operator results shown to the agent. |
| `stats_enabled`, `include_operator_properties`, `attempt_reflection`, `max_loaders_per_source` | Finer ACT/DECORATE-side knobs. |

See the `DataflowSystem.__init__` docstring and `CLAUDE.md` for the full set and
the canonical list of shipped variants.

---

## 6. Creating your own DataflowSystem variant

Two steps: subclass, then re-export.

### 6.1 Subclass in `systems/dataflow_system.py`

```python
class DataflowSystemMyVariant(DataflowSystem):
    """One-line description of what's special about this variant."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5-mini",     # the model id the agent service uses
            context_mode="latest",
            flow_level=1,                # context decoration (optional)
            data_level=2,
            max_steps=25,
            name="DataflowSystemMyVariant",   # MUST match the class name
            verbose=verbose,
            *args,
            **kwargs,
        )
```

The `name=` string is what shows up in `results/` and `system_scratch/`
directories — keep it identical to the class name to avoid confusion.

### 6.2 Re-export from `systems/__init__.py`

The harness resolves a SUT by `getattr(systems_module, sut_name)`, so the class
must be importable from the `systems` package:

```python
from .dataflow_system import (
    ...,
    DataflowSystemMyVariant,
)
```

### 6.3 Verify it loads

```bash
.venv/bin/python -c "import systems; print(systems.DataflowSystemMyVariant.__name__)"
```

If that prints the class name, the dynamic lookup will find it and you can run it
as `SUT=DataflowSystemMyVariant`.

> Tip: when you spin up a numbered family (e.g. sweep runs `…V1`–`…V10`), keep
> them identical except for `name=`/`max_steps`, and add them all to the
> `__init__.py` import list in one block.

---

## 7. How scoring works

Each task's `answer_type` maps to a metric set
(`benchmark/fixtures/answer_type_fixtures.json`):

| `answer_type` | Primary metric |
|---|---|
| `numeric_exact` / `string_exact` | `success` |
| `numeric_approximate` | `rae_score` (+ MAE / MSE recorded) |
| `string_approximate` | `llm_paraphrase` (gpt-4o-mini) |
| `list_exact` | `f1` (+ precision/recall recorded) |
| `list_approximate` | `f1_approximate` (gpt-4o-mini for strings, ≤1% rel-err for numbers) |

The headline **"Total score"** printed at the end of a run is
`sum(support × mean)` over the headline metrics (`success`, `llm_paraphrase`,
`rae_score`, `f1`, `f1_approximate`) divided by the task count. `precision`,
`recall`, `mean_absolute_error`, and `mean_squared_error` are recorded but do
**not** count toward the headline percentage.

---

## 8. Launching via the shell scripts

All four scripts ultimately call `evaluate.py`. Pick by what you're doing:

| Script | Use it to… | Writes bulk cache? |
|---|---|---|
| `run_dataflow_system.sh` | Run a SUT over whole workloads (fresh scratch + bulk cache). | Yes |
| `run_dataflow_tasks.sh` | Re-run specific task IDs (per-task scratch only). | No |
| `evaluate_dataflow_system.sh` | Rebuild bulk cache from scratch, then re-score with `--use_system_cache`. | Yes (rebuilds) |
| `evaluate_dataflow_tasks.sh` | Re-score specific tasks only (edit the SUT inside). | No |

### 8.1 `run_dataflow_system.sh` — full run

Env knobs:

- `SUT` — class name from `systems/__init__.py` (default `DataflowSystemHaiku45`).
- `ORACLE_MODE=true|false` — `true` adds `--use_truth_subset` (agent sees only
  each task's `data_sources`). Default `true`.
- `PARALLEL=true|false` — fan workloads out to background processes vs. run them
  sequentially.
- `WORKLOADS_OVERRIDE="legal environment"` — space-separated list to scope the
  run without editing the hardcoded `WORKLOADS=(…)` array near the top of the
  script.

```bash
export OPENAI_API_KEY=sk-…
SUT=DataflowSystemGPT5Mini PARALLEL=true ./run_dataflow_system.sh
```

Background logs land in `logs/{ts}/{workload}.log`.

### 8.2 `run_dataflow_tasks.sh` — a few tasks

```bash
SUT=DataflowSystemGPT5Mini ./run_dataflow_tasks.sh legal-hard-2 environment-hard-20
```

The workload is parsed from each task ID (`legal-hard-2` → `legal`). This writes
per-task scratch but **not** the bulk response cache — so re-score with
`evaluate_dataflow_system.sh` afterward (it rebuilds the cache from scratch):

```bash
WORKLOADS_OVERRIDE="legal environment" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh
```

> **macOS bash gotcha:** stock macOS ships bash 3.2, which lacks `declare -A`.
> The "parse workload from task IDs" mode needs bash 4+. On stock macOS, use the
> legacy form `./run_dataflow_tasks.sh <workload> <task_ids…>` or call
> `evaluate.py --task_id …` directly.

### 8.3 Re-score only (no agent calls)

After editing a metric, or hand-fixing a `answer.json`:

```bash
WORKLOADS_OVERRIDE="legal" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh
```

---

## 9. Finding what failed

`scripts/list_failed_tasks.py` is the canonical inspector — it groups every
non-passing task by workload with a reason
(`execution_error` / `empty_answer` / `wrong_answer` / `partial` /
`missing_eval`):

```bash
# every failure for a SUT, all workloads with results
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini

# one workload
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --workload legal

# only true execution failures (skip merely-wrong answers)
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --errors-only

# machine-readable
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --json
```

Re-run every failed task in one pipe, then re-score:

```bash
python scripts/list_failed_tasks.py --sut DataflowSystemGPT5Mini --json \
  | python -c "import json,sys; d=json.load(sys.stdin); print(' '.join(t['task_id'] for w in d.values() for t in w))" \
  | xargs -n50 ./run_dataflow_tasks.sh

WORKLOADS_OVERRIDE="legal environment" SUT=DataflowSystemGPT5Mini ./evaluate_dataflow_system.sh
```

---

## 10. Where everything lives

```
data/{domain}/input/             raw data lake the SUT reads
workload/{domain}.json           tasks (id, query, answer, answer_type, data_sources, subtasks)
benchmark/fixtures/              answer_type → metrics mapping, etc.
systems/                         SUT implementations + dataflow agent client
  dataflow_system.py             DataflowSystem + all variants
  __init__.py                    re-exports (dynamic SUT lookup reads this)
system_scratch/{SUT}/{task_id}/  per-task SUT artifacts (audit trail)
results/{SUT}/
  response_cache/                bulk SUT responses; reused via --use_system_cache
  {workload}_measures_*.csv      per-(task, metric) scores
results/aggregated_results.csv   per-(sut, workload, metric) rollup
logs/{ts}/{workload}.log         stdout/stderr of background runs
```

---

## 11. Common gotchas

- **`OPENAI_API_KEY` not set** → import-time crash or scoring failure. Export it
  (see §2.2).
- **Texera not running** → connection refused. Check `curl
  http://localhost:3001/api/agents/` and the three ports in §2.3.
- **Partial reruns don't update the leaderboard number** because `--task_id`
  mode never writes the bulk cache. Always follow `run_dataflow_tasks.sh` with
  `evaluate_dataflow_system.sh`.
- **Stuck `GENERATING` agent** (GPT-5-mini sitting >10 min with 0 committed
  steps): the WebSocket has no client timeout. `DELETE /api/agents/<id>` won't
  free the harness's open socket — `kill -TERM <python pid>` and recover via the
  partial-rerun workflow.
- **`legal-tiny` quickstart** references `data/legal/tiny/`, which doesn't exist
  in this fork — use `--workload legal` against the full input dir instead.
- **macOS bash 3.2** — see §8.2.
</content>
