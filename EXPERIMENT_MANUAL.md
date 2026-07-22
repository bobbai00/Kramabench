# Dataflow-Agent Experiment Manual

How to configure the agent (and what context each config produces), how to launch
experiments, and how to evaluate them. Companion to `kb.py` and
`judgment_runs/levers_report/*`. All examples below are real renders taken from traces.

---

## 1. Configuring the agent

### 1.1 Where configuration lives

```
SUT class (systems/dataflow_system.py)     <- you define arms here
  -> DataflowSystem.__init__(kwargs)
    -> dataflow_agent.py AgentSettings.to_api_dict()
      -> POST /api/agents {settings}       <- agent-service (bun, :3001)
        -> context assembler + prompts + tools
```

One experiment arm = one SUT class = one frozen combination of knobs. Results land in
`system_scratch/<SUT-name>/<task-id>/`.

### 1.2 The knobs

| Knob (SUT kwarg) | Values | What it changes |
|---|---|---|
| `model_type` | `gpt-5.2`, `gpt-5-mini`, … | the LLM (via litellm :4000) |
| `context_mode` | `latest` / `delta` | snapshot-only vs event-history context (see 1.3) |
| `max_operator_result_char_limit` | 1000 / 3000 / 5000 | chars of each result table rendered — the **rows knob** |
| `column_stats` + `data_level` | off / on + 1/2 | schema line only vs per-column **stats profile** |
| `enable_code_in_snapshot` | bool | LATEST-only: show each op's `Code:` block + short summaries |
| `summarize_params` | deep-partial dict | fine-grained per-operator render control (detail `shape`/`sample`, `includeCode`, history, …) |
| `context_window_tokens` + `static_compaction` / `compaction_strategy` | int + bool/str | DELTA history folding when context grows |
| `enable_inspect_tool` | bool | agent can PULL more of a result on demand |
| `enable_render_prefs` | bool | agent declares per-op render size (v1 tested NEGATIVE) |
| `frontier_decay_config` / `probe_retirement_config` / `fold_resolved_revisions_config` | dicts | history-decay experiments (tested, REJECTED on cache-aware cost) |
| `max_steps` / `max_operator_edits` | int | step budget / per-op edit cap (anti-thrashing) |
| `attempt_reflection`, `error_reflection`, `thought_replay`, `few_shot_prompt` | bool | prompt-level nudges |

Flags are byte-parity when off: an arm with a flag off produces the exact same prompt,
tools, and render as before the flag existed (enforced by
`agent-service/src/agent/context/orthogonality.test.ts`).

### 1.3 What each config renders (real examples)

**LATEST mode** — the agent sees only the current snapshot, no history:

```
# User Task
<the task, verbatim + answer-format line>

# Current Dataflow
## Operators

### Operator `customers` (DataLoading)
Summary: Load customers.csv (clean comma CSV, header row 0)
Result:
  Output Table: 10000 rows, 5 cols
  	customer_id	name	email	signup_date	tier
  0	C001	Alice	alice@example.com	2023-01-15	gold
  ...
```

**DELTA mode** — the agent sees its whole action history; each event carries the
tool call (with `summary` AND `code`) plus the observation:

```
# Agent Events

## Agent Event 2

Action:
- createOrModifyOperator
    operatorId: contributors
    summary: Load file 2024_CSN_Data_Contributors.csv as a CSV into a DataFrame
    code:
      def load():
          return pd.read_csv('data/legal/input/2024_CSN_Data_Contributors.csv')

Observation:
- operator contributors added
  result:
    Output Table: 142 rows, 4 cols
    	Data Contributors	Unnamed: 1 ...
```

> Note: DELTA **always shows code** (inside each event's Action). LATEST shows **no
> code** unless `enable_code_in_snapshot` is on.

**Rows knob (char cap)** — with a 1k cap, a big table renders ~a few rows with an
omitted-middle marker; 5k renders many more rows. The marker line:

```
  ...	...	...	...	   <- rows omitted to fit; true row count is in the shape line
```

**Schema-only vs stats profile** (`column_stats=False` vs `True, data_level=2`):

```
Schema (5 cols): customer_id (str), name (str), email (str), signup_date (datetime), tier (str)
```
vs
```
Column Schema and stats:
- "tier" (str): null=0, distinct=3, top_5={"gold"=4200, "silver"=3800, "bronze"=2000}
- "amount" (numeric): null=0, mean=219.2, min=95.00, max=520.0
```

**Code-in-snapshot** (`enable_code_in_snapshot=True`, LATEST only) — operator blocks
gain a `Code:` block, and the tool asks for ONE-line summaries instead of detailed ones:

```
### Operator `raw_contributors` (DataLoading)
Summary: Raw text preview of 2024_CSN_Data_Contributors.csv
Code:
  def load():
      lines = open('data/legal/input/...csv').read().splitlines()
      return pd.DataFrame({'line': lines[:20] + ...})
Result:
  Output Table: 30 rows, 1 cols
  ...
```

**summarizeParams patch** — surgical render control, e.g. render results as shape-only:

```python
summarize_params={"operators": {"defaults": {"result": {"latest": {"detail": "shape"}}}}}
```

### 1.4 The arms we actually use (worked combinations)

| SUT class | mode | cap | stats | extra |
|---|---|---|---|---|
| `DataflowSystemGPT5MiniDelta1kSchemaOnly` | delta | 1k | off | C1/C2 anchor |
| `DataflowSystemGPT5MiniDelta5kSchemaOnly` | delta | 5k | off | C1 ray (rows knob) |
| `DataflowSystemGPT5MiniDeltaStats1kD2` | delta | 1k | on (d2) | C2 ray (stats knob) |
| `DataflowSystemGPT5MiniLatest1kSchemaOnly` | latest | 1k | off | code-flag baseline |
| `DataflowSystemGPT5MiniLatest1kCodeInSnap` | latest | 1k | off | + `enable_code_in_snapshot` |
| `DataflowSystemGPT52Delta{1k,3k,5k,7k}SchemaOnly` | delta | sweep | off | GPT-5.2 cap sweep |

**Defining a new arm** (pattern):

```python
class DataflowSystemGPT5MiniLatest1kCodeInSnap(_GPT5MiniSchemaOnlySweep):
    """gpt-5-mini, LATEST, 1k, schema-only + code shown in snapshot."""
    _CONTEXT_MODE = "latest"
    _RESULT_CHARS = 1000
    _NAME = "DataflowSystemGPT5MiniLatest1kCodeInSnap"
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(enable_code_in_snapshot=True, verbose=verbose, *args, **kwargs)
```

then add the class name to `systems/__init__.py`. Check it appears in `./kb.py systems`.

---

## 2. Launching experiments

### 2.1 Prerequisites (once per machine session)

1. **Backends up**: 8 JVM services (sbt), agent-service (bun, :3001), litellm (:4000),
   docker infra. See `dataflow-agent/CLAUDE.md` for launch commands.
2. **Health check** before any run:
   ```bash
   curl -s -o /dev/null -w '%{http_code}' localhost:3001/api/agents   # 200
   curl -s -o /dev/null -w '%{http_code}' localhost:4000/health/liveliness  # 200
   ```
3. **Warm-up**: the FIRST request after an agent-service restart cold-start fails
   (~0.05 s, empty). Run one throwaway task before a pool.

### 2.2 The entry points

**`kb.py`** — the daily driver (auto-exports OPENAI_API_KEY, oracle mode on by default,
stall watchdog):

```bash
./kb.py systems                          # list available arms
./kb.py run   --sut <ARM> --parallel     # full 104-task run
./kb.py tasks --sut <ARM> --ids "legal-hard-2 astronomy-easy-1"   # specific tasks
./kb.py rerun-failed --sut <ARM>         # re-run score-0 tasks
./kb.py scores --sut <ARM>               # leaderboard
./kb.py cost --sut <ARM> --by workload   # cost breakdown
./kb.py compare --sut A B                # A-vs-B outcome matrix
./kb.py venn --sut A B                   # exclusive wins + cost split
./kb.py judge --sut A B --tasks-file F   # M3/M4 chunked-judge metrics (see §3)
```

**`evaluate.py`** directly — for one task or the subtask eval:

```bash
.venv/bin/python evaluate.py --sut <ARM> --workload legal \
  --task_id legal-easy-4 --use_truth_subset --no_pipeline_eval \
  [--run_subtasks]        # also run the ~6 gold sub-questions (M2)
```

### 2.3 A/B pool pattern (matched-task comparisons)

Use a **fixed task list** (`judgment_runs/levers_report/tasks50.txt` — the standard
50-task sample) and a per-task pool so one hang can't stall an arm:

```bash
for ARM in ArmA ArmB; do for T in $(cat judgment_runs/levers_report/tasks50.txt); do
  echo "$ARM ${T%%-*} $T"; done; done | xargs -P 4 -L 1 bash -c '
    timeout 600 .venv/bin/python -u evaluate.py --sut "$0" --workload "$1" \
      --no_pipeline_eval --task_id "$2" --use_truth_subset > logs/$0__$2.log 2>&1'
```

**Rules learned the hard way:**
- **Concurrency <= 4.** At 6, engine contention makes hard tasks instant-fail (0.09 s,
  no work, exit 0 — uniform across arms, so matched analysis survives, but avoid it).
- **Same era only.** Never compare arms run weeks apart — service changes confound
  ("vintage artifacts"; bit us twice). Re-run the baseline alongside every new arm.
- **Matched analysis.** Compare only tasks BOTH arms completed
  (`scripts/final_knob_analysis.py` does this).
- **Variance gate.** Before believing an effect, know the noise floor: re-run one arm
  3x on `variance_subset12.txt`, compare per-task scores. Measured floor (mini):
  subtask-mean std 0.025, ~17% of tasks flip binary pass on a pure re-run.

### 2.4 What a run leaves behind (`system_scratch/<ARM>/<task>/`)

| file | contents |
|---|---|
| `evaluation.json` | scores + token/cost usage (the record M1 reads) |
| `answer.json` / `ground_truth.json` | parsed answer / gold |
| `react_steps.json` | full trace: every step's `inputMessages` (byte-exact context the LLM saw), tool calls (with code), usage |
| `workflow.json` | the final DAG (operators + links) |
| `judge_m3m4.json` | M3/M4 judge cache (written by `kb.py judge`) |
| `<task>-N/` dirs | same set per gold subtask (when `--run_subtasks`) |

---

## 3. Evaluating an agent

### 3.1 The metric suite

| # | Metric | Question | Reads | How to run |
|---|---|---|---|---|
| **M1** | End-to-end score | final answer right? | `evaluation.json` (KramaBench native: exact 0/1, RAE for approx, F1 for lists) | any run; `./kb.py scores` |
| **M2** | Subtask score | can it answer each gold sub-question? | subtask dirs' `evaluation.json` | `evaluate.py --run_subtasks`; analyze with `scripts/final_knob_analysis.py` |
| **M3** | Evidence-in-context | did the agent **SEE** each needed value? | last step's `inputMessages` in `react_steps.json`, chunked; LLM judge, binary per subtask | `./kb.py judge --sut A B --tasks-file F` |
| **M4** | Step-performed | did the agent **DO** each needed step? | same chunks, action lens (summary+code) | same command (`--lens both` default) |
| — | **Failure modes** | why did a task fail? | M3 x M4 on failed tasks | `scripts/judge_vs_answers.py` |
| — | **Cost** | at what price? | `evaluation.json` usage fields | `./kb.py cost`; per-arm totals |
| — | **Noise floor** | how much is randomness? | 3x re-run snapshots | variance protocol (§2.3) |

### 3.2 Purpose and rules for each

**M1 (headline).** The number the paper leads with. Continuous where the benchmark says
(RAE/F1), binary for exact types. Use KramaBench's score AS REPORTED — never re-grade
(we tried RAE-relaxing exact tasks; rejected as metric-shopping).

**M2 (finer headline).** ~6 gold sub-questions per task -> 300 data points per 50-task
arm instead of 50 -> detects knob effects M1 is underpowered for. Caveat: subtasks run
in ISOLATION, so it under-scores (13/47 tasks passed main while scoring low on
subtasks) — use it for *comparing* configs, not absolute capability.

**M3 (mechanism / manipulation check).** One judge call per context chunk (event in
delta, operator in latest), all gold subtasks listed, verdicts keyed by subtask-id,
binary; subtask = yes if ANY chunk shows its value; task score = % of subtasks. Its
job: prove the knob physically changed what the agent saw *before* arguing about
accuracy. Validated: shows +0.081 where the rows knob binds (mini 1k->5k) and a clean
0.000 dead-even where it doesn't (5.2 3k->5k) — which *explained* the 5.2 accuracy coin.
Rules: aggregate + direction only; never cite one task's M3; per-task M3-gain does NOT
predict which task flips (delivery != conversion — replicated 5x).

**M4 (plan control).** Same machinery, different prompt: "does this chunk show the step
being PERFORMED (values irrelevant)?" Two flavors: *process* (any event, including
deleted probes) vs *deliverable* (only ops surviving in `workflow.json`). Its value so
far is being FLAT under render knobs — proof they change value-getting, not planning.
Becomes the primary detector only for plan-level interventions (history knobs,
decomposition hints).

**Failure modes (the diagnosis).** For each failed task:
`M4<1 -> mode 1 (step missing)`; `M4=1 & M3<1 -> mode 2 (steps done, value never seen)`;
`both=1 -> mode 3 (had everything, still failed)`. This tells you which lever can help:
render knobs fix mode 2 only (mini 1k->5k: mode-2 failures 2->0; the 5k residue was 100%
mode 1, which no render knob touches).

**Cost (the other axis).** Every verdict is accuracy AND cost. Read
`cost_usd_sut`, input/output tokens, cached share, steps from `evaluation.json`.
Compare cache-aware: a knob that breaks prompt-cache continuity can "win" tokens and
lose dollars (killed rank-3/4).

**Binary pass/fail.** Only for A-vs-B case stories ("1k failed this, 5k passed it"):
pass = exact for exact types, >=0.9 for graded. Pick clean 0->1 flips; threshold-
straddlers are noise. Never use as an aggregate.

**Noise floor (the gate).** Run before claiming: identical config 3x. An effect is real
when it clears the floor AND is directional (asymmetric up/down movement) — symmetric
movement (e.g. 16up/16dn) is the signature of nothing.

### 3.3 The standard verdict procedure for a new knob

1. Define arm + same-era baseline (§1.4); run both on `tasks50.txt` (§2.3, concurrency 4).
2. **M1 + cost** on matched tasks -> is there an effect and at what price?
3. **M2** (`--run_subtasks`) if M1 is suggestive but underpowered.
4. **M3** manipulation check -> did the knob actually change what the agent saw?
   (If M3 is dead-even, stop: any M1 movement is noise or a side-channel.)
5. **M4 + failure modes** -> which failure class moved?
6. Check against the **noise floor**; require directionality, not just a mean delta.
7. Case studies: pick clean binary flips, verify the mechanism by hand in the trace.
