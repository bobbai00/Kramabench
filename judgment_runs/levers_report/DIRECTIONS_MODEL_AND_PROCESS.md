# Findings + plan: smaller models, randomness, best-of-5, and process metrics (2026-07-20)

Investigation of Bob's future directions, grounded in existing on-disk data
plus the KramaBench source. TL;DR: the two "new dimensions" he wants (subtasks,
process/LLM-judge) are ALREADY in KramaBench and disabled; smaller models are
nearly as accurate at ~3.5× lower cost and look more consistent; best-of-5 and
run-variance need one small dedicated experiment.

## Takeaways — validated

1. **Randomness ≈ coin flip.** Confirmed: across the probe-star, 24/27
   exclusive-win pairs are chronic variance; only 1 attributed flip per
   comparison. Twin-noise flips ~9-12 tasks between identical configs. Accuracy
   is dominated by per-roll variance, not the knob.
2. **Accuracy+cost alone is too coarse.** All 71 arms cluster 51-84/104; the
   GPT-5.2 dataflow arms sit 73-83. The signal-to-noise on the accuracy axis is
   low → motivates process dimensions (below).
3. **Script/code baseline is competitive.** Best code agent
   (`CodeAgentSystemGpt52Chars5kGuided`) = 80/104 vs best dataflow 83 — a 3-task
   edge. The room over the baseline is small.
4. **GPT-5.2 is already strong** — and (new finding) a much smaller model nearly
   matches it, so the model axis has little headroom either.

## A. Smaller model — strong existing data

| config (3k) | GPT-5.2 acc | $/task | mini acc | $/task | Δacc | cost× |
|---|---|---|---|---|---|---|
| Latest schema-only | 79 | $0.0516 | 77 | $0.0140 | −2 | 3.7× |
| Delta schema-only | 80 | $0.0514 | 76 | $0.0139 | −4 | 3.7× |
| DeltaStats D2 | 83 | $0.0634 | 75 | $0.0145 | −8 | 4.4× |
| LatestStats D2 | 77 | $0.0451 | 75 | $0.0150 | −2 | 3.0× |

- **mini ≈ 5.2 on accuracy (−2 to −8), at ~3.5× lower cost.** Best cost/accuracy
  operating point on the board.
- **The stats knob helps mini least** (−8 gap on the stats arm vs −2 on lean) —
  smaller model extracts less from the extra profile. Knob value is
  model-dependent.
- Open-source is plausible: the stack already has a `local-react` driver +
  `qwen-xml` tool dialect (Qwen path) and an ollama interface in the benchmark.

## B. Does a smaller model reduce randomness? — proxy says maybe

- **Cross-config fail-set agreement: mini = 0.72 vs 5.2 = 0.65 Jaccard.** The
  mini arms fail more of the *same* tasks regardless of knob → its failures are
  more structural, less coin-flippy. Weak-but-directional support for the
  hypothesis (smaller model can't "luck into" a hard task the easy way).
- CAVEAT: this is cross-*config* agreement, not same-config *run* variance. The
  real test (repeat the identical config N times) is not in the data yet.

## C. Best-of-5 / majority-≥3 standard — not answerable from current data

- Recovery rounds only re-run FAILED tasks, so passes have 1 attempt and fails
  have 2-3 — asymmetric, can't estimate a clean per-task pass rate.
- Needs a dedicated experiment: run a fixed config 5× single-shot (NO recovery)
  over all 104, score majority-≥3. Cheap at mini (~$15 for 5×104).
- Combine with B: run {mini config, 5.2 config} × 5 → get BOTH the majority
  standard AND the run-variance comparison in one shot.

## D. Subtasks — KramaBench's built-in partial-achievement metric (DISABLED)

- **Every one of the 104 tasks carries subtasks** (avg ~6, ~600 total). Each is
  a `{id, step, query, answer, answer_type}` — a scorable sub-question.
- They make hidden steps explicit and independently gradable. Example —
  biomedical-hard-5 subtask 2: *"What cases are included in the study and have a
  serous histologic type?"* answer = the **12** kept case IDs (S006…S103, i.e.
  S043 dropped). This directly scores the `Case_excluded` step that the final
  answer hides — exactly the process signal we want.
- Two uses: (i) **subtask execution** (`run_subtasks=True`) re-runs each
  sub-question — measures whether the agent CAN solve each step, but costs ~6×
  the runs; (ii) feed the subtask steps as the reference for the pipeline judge
  (E) — no re-run.

## E. Process / correctability metric — the LLM judge already exists (DISABLED)

- `GPTInterface.evaluate_data_pipeline(sut_generated_pipeline, task)` +
  `PIPELINE_EVALUATION_PROMPT`: an LLM judge (gpt-4o-mini) reads the generated
  pipeline CODE + the task's subtask steps as "key functionalities" and returns
  a **Yes/No per functionality** → fraction implemented = a process score.
- It is **retroactive and cheap** (one judge call per task, no re-run) — but the
  dataflow SUT currently passes `pipeline_code: ""  # Skip pipeline eval`
  (dataflow_system.py:519,655). So it was deliberately turned off because the
  agent emits an operator DAG, not a script.
- FIX = a small **DAG→pipeline_code serializer**: walk `workflow.json` operators
  in topological order, concatenate their `code`, feed to the existing judge.
  Then compute a process score for every trace already on disk.
- **Two complementary process metrics to build:**
  1. *Pipeline coverage* (KramaBench, LLM-judged): fraction of reference
     functionalities the code implements. Retroactive.
  2. *Plan edit-distance to gold* (novel, ours): reduce the SUT DAG and the gold
     `solutions/*.py` to op-sequences (the semantic-walk representation) and
     compute structural edit distance / step-coverage. Deterministic; directly
     proxies **correction effort** — Bob's "how easy for a human to fix it."
- **Why this matters (the reframe):** on binary accuracy the counter-intuitive
  cases are coins. On a process/distance axis they separate — e.g.
  archeology-hard-7's L2-vs-L∞ arm is **one op-edit from gold** (trivially
  correctable: "use Euclidean"), whereas a task missing 3 ops is high-effort.
  Context selection that moves the agent *closer to gold* is valuable even when
  the final answer flips — a far more sensitive axis than pass/fail.

## Next plan (prioritized)

1. **Build the DAG→pipeline_code serializer + turn pipeline eval back on.**
   Compute the pipeline-coverage score for every existing trace (all arms, no
   re-runs). Deliver a per-arm, per-task process score. (Highest value, cheapest;
   unlocks D+E on data we already have.)
2. **Re-run the C1/C2/C3 analysis on the process axis.** Does stats/rows/history
   raise subtask/pipeline coverage even where the final answer is a coin? This is
   the "more dimensions" payoff and likely de-noises the knob comparison.
3. **Build the plan-edit-distance metric** (op-graph diff vs `solutions/*.py`),
   as the deterministic correctability companion to the LLM judge.
4. **One 5× variance/best-of-5 experiment**: {mini Delta3kSchemaOnly, 5.2
   Delta3kSchemaOnly} × 5 single-shot over 104 → majority-≥3 accuracy + per-task
   pass-rate variance. Answers B + establishes C. ~$65 total.
5. **Shift the substrate to mini** for future knob sweeps (3.5× cheaper, ≈acc,
   more consistent) — and re-test whether knob effects are cleaner/larger at mini
   under the majority-≥3 standard.

## Caveats
- Pipeline eval uses an LLM judge (gpt-4o-mini) → itself has variance; use a
  fixed judge + report judge-agreement on a sample.
- Subtask *execution* is expensive (~6× runs); prefer the retroactive pipeline
  judge unless we specifically want per-step answerability.
- mini↔5.2 cost figures are per-task averages on the same 104 tasks; cross-model
  accuracy gaps are single-roll and inherit the chronic-variance caveat.
