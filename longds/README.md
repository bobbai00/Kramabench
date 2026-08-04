# LongDS-Bench connector

Runs [LongDS-Bench](https://arxiv.org/abs/2605.30434) (`zjunlp/DataMind`, `longds/`)
against the dataflow agent. LongDS is 68 tasks / 2,225 turns of long-horizon data
analysis where turn N+1 depends on analytical state built in turns 1..N; the best
published model reaches 48.45 and its dominant failure class is Cascade Error — a
locally-correct turn poisoned by an earlier wrong state.

Each task runs as **one agent + one workflow**, every turn another
`{type:"message"}` on it. Nothing is reset between turns: the operators, their
code, their materialized results and the DELTA event log are the carried state.
`DataflowSystem.serve_query` is deliberately bypassed — it recreates the agent per
query (`dataflow_system.py:561-577`), which destroys exactly what is measured here.

## Layout

| file | role |
|---|---|
| `numeric_gate.py` | runs upstream's gold Python in our venv and diffs every published answer. Run this for any new task before trusting a score. |
| `prepare.py` | splits the dataset into a manifest (`turn_id`, `context`, `question`) and a gold file the runner never reads |
| `arms.py` | arms as `DataflowSystem` subclasses, unregistered so KramaBench's harness cannot pick them up |
| `run_longds.py` | the turn loop, per-turn checkpointing, dual artifact emission |
| `judge_longds.py` | upstream's `JUDGE_PROMPT` loaded by path, scored via litellm; reports both aggregations and the state-pattern / dependency-breadth / progress breakdowns |
| `analyze_run.py` | per-turn cost, steps, and context growth |

## Setup

Data is **not** in the repo. Download the tasks you need from HF and reach them
through a symlink, so one relative path resolves both in this repo and in the JVM
services (whose cwd is the dataflow-agent root, which symlinks `data/` here):

```bash
cd ~/Desktop/bobflow/DataMind/longds     # read-only upstream clone
hf download zjunlp/LongDS --repo-type dataset --local-dir dataset \
  "task/longds/task_list.json" \
  "task/longds/sports/nfl_big_data_bowl_2023/task1/*" \
  "data/longds/sports/nfl_big_data_bowl_2023/task1/*"

cd ~/Desktop/bobflow/Kramabench
ln -sfn ../../DataMind/longds/dataset/data/longds data/longds   # already gitignored
```

The full dataset is 19.5 GB; a single task is usually well under 200 MB. Use
Kramabench's own venv (`.venv/bin/python`) — the system interpreter lacks the deps.

## Running

```bash
.venv/bin/python longds/numeric_gate.py                        # go/no-go, see below
.venv/bin/python longds/prepare.py --task sports/nfl_big_data_bowl_2023/task1
.venv/bin/python -u longds/run_longds.py --task sports__nfl_big_data_bowl_2023__task1
.venv/bin/python longds/judge_longds.py --sut LongDS_LongDSLunaDelta1k
.venv/bin/python longds/analyze_run.py --sut LongDS_LongDSLunaDelta1k --task sports__nfl_big_data_bowl_2023__task1
```

Use `setsid` and `-u` for long runs: stdout buffers without `-u`, and a session
killed with its parent shell is unrecoverable (below).

## Deviations from the official protocol — state these with any number

1. **Judge model — resolved.** The paper's judge, `deepseek-v4-pro`, is reachable
   through OpenRouter, so scores can be made directly comparable:

   ```bash
   set -a && . ./.env && set +a
   JUDGE_BASE_URL=https://openrouter.ai/api/v1 JUDGE_API_KEY=$OPENROUTER_API_KEY \
     .venv/bin/python longds/judge_longds.py --sut <SUT> \
       --judge-model deepseek/deepseek-v4-pro --overwrite
   ```

   The litellm default (`gpt-5.2`) stays the cheap local option. On the pilot's 41
   judgeable turns the two judges **agree on 38 (92.7%)** — about the paper's own
   93.11% human-vs-LLM agreement — disagreeing only on turns 12, 13 and 22. So
   gpt-5.2 is a sound stand-in for arm-vs-arm work, and deepseek-v4-pro is what to
   use for any number quoted against the paper. Note OpenRouter authorizes each
   request against the remaining account credit, so `--max-tokens` is explicit
   (default 4000) — the model's own 65k default 402s on a zero-credit account.
2. **Runtime.** Not DSGym's pinned Docker Jupyter executor and not their fixed
   40-step `<python>` ReAct loop — the agent builds a workflow with its own tools.
   That is the point of the exercise, but it means this is not a leaderboard score.
3. **Answer contract.** The prompt asks for a single-line JSON object on a
   `**Final Answer:**` line. Upstream's DSGym runner uses `<answer>` tags; their
   Codex/Claude runners use a JSON schema. All three differ; ours matches theirs in
   substance (judge sees a JSON string) but not in syntax.
4. **Numerics gate.** Gold answers were computed on a frozen 181-package stack and
   the judge demands exact equality with no rounding, so version drift can fail
   correct work. Measured on the pilot task: **42/42 answers reproduce exactly** in
   our venv (pandas 2.2.3 / numpy 2.1.0). Re-run `numeric_gate.py` for new tasks.

## Resume granularity is the TASK

Agents live only in agent-service's process memory (`server.ts:281` — no session
table). A crash, a Bun restart, or an engine recycle loses the trajectory even
though the workflow row survives in Postgres, so a half-finished task is re-run
from turn 1. `--skip-done` skips only fully-completed tasks. Corollary for pools:
recycle at task boundaries, never mid-task.

## First result — baseline, 2026-08-03

Arm `luna-delta-1k`: gpt-5.6-luna, DELTA, 1k result chars, no stats, no
code-in-snapshot, `max_steps=40`/turn, **append-only history** (`contextWindowTokens=0`,
nothing trimmed). Task `sports/nfl_big_data_bowl_2023/task1` — 42 turns, the longest
in the benchmark, and it exercises all four state-evolution patterns.

**24.39%** under the paper's own judge, `deepseek-v4-pro` (10 of 41 judgeable turns;
one turn's judge reply was unparsable and is excluded). Under `gpt-5.2`: 23.81%
(10/42). $9.99, 70 min, 7.8 steps/turn mean.

| cut (deepseek-v4-pro judge) | result |
|---|---|
| by pattern | Update 27.3 · Counterfactual 25.0 · Initial 15.4 · **Rollback 15.4** |
| by task progress | 0-25% 30.0 · 25-50% 50.0 · 50-75% 18.2 · **75-100% 0.0** |
| by dependency breadth | 2-3: 23.1 · 4+: 23.1 |
| context growth | 20.6 kB → 920 kB (21.9 kB/turn), ending at ~230k tokens ≈ 85% of the window |
| cost shape | $0.0095 on turn 1 → $0.90 on turn 36 |

Read against the paper: sports is its hardest domain — GPT-5.4 10.52,
DeepSeek-V4-Pro 15.82, Claude-4.6-Sonnet 19.76, Gemini-3.1-Pro 31.85, Kimi-K2.6
32.85. At 24.39 on its longest sports task, an untuned first run sits above Claude
and DeepSeek and below Gemini and Kimi. Treat that as indicative only: this is one
task, the domain has three, and the agent builds a workflow rather than running
DSGym's fixed ReAct loop.

The *shape* replicates their headline finding, and more sharply: accuracy decays to
**zero** in the final quarter of the task, and Rollback is the worst pattern under
both judges.

Three things the pilot establishes:

- **Cascade is the mechanism, and the DAG only fixes half of it.** Turn 3 computes
  one percentile wrong (`first_score` 1.0685 vs gold 0.9755) and turns 5, 7, 8, 9,
  10 then fail on small numeric drift inherited from it. The workflow reliably
  prevents "forgot to recompute a dependent"; it does nothing about "recomputed
  faithfully from a wrong formula". Exact-match grading turns the second into a
  long tail of zeros.
- **Late turns stop answering the question.** Turn 42 returns "unrelated
  shortcut/player/team score lists" instead of the requested distances; turns 29,
  32 and 42 answered in ~1 step, twice by restating an earlier turn's answer. With
  `# User Task` at the top and ~900 kB of appended events after it, the live
  request is buried. This is the concrete case for turn-boundary compaction.
- **Append-only cost is quadratic, and prompt caching does not save it.** Cached
  input is a roughly fixed ~6k tokens per model step regardless of prompt size
  (measured here and in the KramaBench luna/terra arms, where the same fixed amount
  merely *looks* like 70% because prompts are small). Every step re-sends the full
  appended log at full price. At $9.99 for one task, the full 68-task benchmark
  would cost ~$680 on this arm — so compaction is a prerequisite for scale, not an
  optimization.
