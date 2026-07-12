# GPT-5.2 3k Delta vs Latest Audit

Audit date: 2026-07-10

## Scope and protocol

This audit compares the two matched GPT-5.2 3k context-mode pairs:

| Information setting | Latest | Delta |
| --- | --- | --- |
| StatsD2 | `DataflowSystemGPT52LatestStats3kD2` | `DataflowSystemGPT52DeltaStats3kD2` |
| SchemaOnly | `DataflowSystemGPT52Latest3kSchemaOnly` | `DataflowSystemGPT52Delta3kSchemaOnly` |

The arms hold model, result-character limit, flow level, step limit, reflection,
and all information settings fixed within a pair. Only `context_mode` changes.

Selection rules:

- Pass: score `>= 1.0`, matching `kb.py compare`.
- Primary cost: `stats.json.cost_usd`, which includes cached-input pricing.
- Material cost gap: at least `$0.005` and at least `10%` of the cheaper arm.
- Strict behavior proxy: both pass, normalized answers match, final operator
  count/link count/operator-type multiset match, and step counts match.
- A workflow-shape match is only a candidate filter. Raw traces and final code
  determine the manual judgment.

There are 104 task directories per arm. `astronomy-hard-11` has no completed
`stats.json` in both Latest arms and in Delta SchemaOnly, so cost comparisons
use the 103 tasks with paired cost artifacts. Delta StatsD2 has a completed
artifact for that task, but it is excluded from paired totals.

Commands:

```bash
./kb.py compare --sut DataflowSystemGPT52LatestStats3kD2 DataflowSystemGPT52DeltaStats3kD2 --top 30
./kb.py compare --sut DataflowSystemGPT52Latest3kSchemaOnly DataflowSystemGPT52Delta3kSchemaOnly --top 30
python scripts/analyze_signal_cases.py --out-dir judgment_runs/signal_analyzer
./kb.py traces --sut <SUT> --task <task_id>
./kb.py tokens --sut <SUT> --task <task_id>
```

## Aggregate result

### Accuracy

| Setting | Both pass | Latest only | Delta only | Both fail | Net Delta pass gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| StatsD2 | 68 | 7 | 8 | 20 | +1 |
| SchemaOnly | 68 | 4 | 7 | 24 | +3 |

Delta has a small aggregate pass advantage, but manual inspection accepts
`0/26` flips as clean evidence that retained Delta history caused the accuracy
win. The first divergence is instead a parser, transform, source, metric, or
answer-evaluation choice. This does not prove history is never useful; these
independent runs do not provide the checkpointed intervention needed for that
causal claim.

Four tasks reverse the winning mode across the two information settings:
`astronomy-hard-8`, `environment-hard-8`, `environment-hard-13`, and
`wildfire-hard-12`. This is strong evidence against a stable task-level rule
such as "this task needs Delta."

### Cost over the 103 paired tasks

| Setting | Arm | Cost | Total tokens | Input | Cached | Uncached input | Steps | Cache hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| StatsD2 | Latest | $4.648 | 6,341,854 | 6,187,400 | 5,296,640 | 890,760 | 729 | 85.6% |
| StatsD2 | Delta | $6.057 | 7,660,153 | 7,478,098 | 6,081,792 | 1,396,306 | 729 | 81.3% |
| SchemaOnly | Latest | $5.028 | 6,033,912 | 5,841,073 | 5,011,840 | 829,233 | 774 | 85.8% |
| SchemaOnly | Delta | $5.750 | 6,721,661 | 6,515,819 | 5,418,752 | 1,097,067 | 707 | 83.2% |

The main invariant is per-step context, not step count:

| Setting | Arm | Input/step | Uncached input/step |
| --- | --- | ---: | ---: |
| StatsD2 | Latest | 8,488 | 1,222 |
| StatsD2 | Delta | 10,258 | 1,915 |
| SchemaOnly | Latest | 7,547 | 1,071 |
| SchemaOnly | Delta | 9,216 | 1,552 |

StatsD2 uses exactly the same aggregate number of steps, yet Delta costs
`$1.409` more. SchemaOnly Delta uses 67 fewer steps and still costs `$0.722`
more. Retained history therefore has a measurable per-turn carrying cost.

## Both-pass cost direction

Raw direction is descriptive because workflows can differ:

| Setting | Latest cheaper | Gross Latest savings | Delta cheaper | Gross Delta savings |
| --- | ---: | ---: | ---: | ---: |
| StatsD2 | 48/68 | $0.756 | 20/68 | $0.143 |
| SchemaOnly | 42/68 | $0.422 | 26/68 | $0.420 |

After the material-gap filter:

| Setting | Latest materially cheaper | Gross savings | Delta materially cheaper | Gross savings |
| --- | ---: | ---: | ---: | ---: |
| StatsD2 | 27 | $0.707 | 6 | $0.117 |
| SchemaOnly | 18 | $0.371 | 14 | $0.387 |

Most material wins in either direction involve different step counts. Delta
can be cheaper when its run simply converges earlier; that is not evidence that
Delta context is intrinsically cheaper.

## Static-control cohort

The strict proxy holds answer, final coarse shape, and steps constant:

| Setting | Cases | Delta pricier | Delta cheaper | Sum Delta-Latest cost | Median Delta-Latest uncached input |
| --- | ---: | ---: | ---: | ---: | ---: |
| StatsD2 | 22 | 13 | 9 | +$0.0469 | +620 |
| SchemaOnly | 21 | 20 | 1 | +$0.0440 | +709 |

Delta has more uncached input in `16/22` strict StatsD2 cases and `19/21`
strict SchemaOnly cases. The effect also exists without operator revisits:

| Setting | Monotone Delta traces | Delta pricier | Median cost gap | Median uncached-input gap |
| --- | ---: | ---: | ---: | ---: |
| StatsD2 | 16 | 11 | +$0.00123 | +620 |
| SchemaOnly | 17 | 16 | +$0.00105 | +510 |

Thus obsolete revisions are not the only removable material. A settled
monotone ancestor's earlier result remains carrying cost after its child has
encoded the useful transformation.

## Manual cost-principle audit

The deterministic miner found 29 same-answer/same-shape cases per pair. Ten
had Delta materially more expensive. Manual inspection accepts 3 and rejects
7. Only `wildfire-hard-20` is also a strong same-step example.

| Setting | Task | Judgment | Mechanism | Latest cost | Delta cost | Steps L/D | Delta-Latest uncached input |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| StatsD2 | `legal-hard-23` | REJECT | Different cleaning/join transform despite same coarse shape | $0.034046 | $0.067867 | 7/7 | +18,779 |
| StatsD2 | `wildfire-hard-4` | ACCEPT | Same raw-parse/join answer path; Delta carries failed loader/parser history | $0.045664 | $0.056685 | 6/7 | +762 |
| StatsD2 | `legal-easy-19` | REJECT | Delta needs an extra scalar-string correction | $0.013791 | $0.021359 | 4/5 | +2,205 |
| StatsD2 | `environment-easy-1` | REJECT | Same pipeline, but gap is cache placement; Delta has 147 fewer input tokens | $0.011404 | $0.018956 | 4/4 | +4,589 |
| StatsD2 | `legal-easy-25` | REJECT | Delta needs an extra numeric-cleaning repair | $0.019250 | $0.026013 | 5/6 | +1,676 |
| SchemaOnly | `legal-hard-22` | REJECT | Same pipeline, cache effect dominates | $0.022704 | $0.031675 | 7/7 | +4,219 |
| SchemaOnly | `legal-hard-7` | ACCEPT | Same relative-growth pipeline; superseded cleaner revision persists | $0.019318 | $0.028077 | 5/6 | +1,519 |
| SchemaOnly | `astronomy-hard-10` | REJECT | Delta has fewer steps; cost gap is mostly 579 extra output tokens | $0.065697 | $0.074237 | 8/7 | +184 |
| SchemaOnly | `wildfire-easy-3` | REJECT | Delta uses fewer steps and input; lower cache reuse/parallel construction dominates | $0.048967 | $0.055989 | 9/7 | +6,019 |
| SchemaOnly | `wildfire-hard-20` | ACCEPT | Same six-turn logical path; corrected encoding loader's prior state is no longer useful | $0.025292 | $0.031453 | 7/7 | +1,330 |

### Accepted mechanisms

`wildfire-hard-20` is the cleanest case. Both modes load the same NOAA data and
variable description, correct the description loader to `cp1252`, filter 2008,
sort housing damage, compute the 90% cutoff, and answer `0.0465`. Delta's
per-agent-step input grows from `5,033` to `8,505`; Latest grows from `5,174`
to `7,874`. Once the corrected loader succeeds, the failed loader version and
its old output/error are no longer decision-relevant.

`legal-hard-7` has the same loader-cleaner-growth chain and answer. Delta first
builds a cleaner with the wrong column order, repairs it, and then creates the
answer operator. After the corrected cleaner is successfully consumed, the
old cleaner code/result can be replaced by a one-line resolution fact.

`wildfire-hard-4` has the same final human-caused-acre/suppression-cost join
logic and answer. Both explore failed CSV parsing before switching to raw-line
loaders. Delta retains one more construction turn. Once the raw loaders and
cleaners succeed, initial isolated CSV-loader attempts and superseded parse
results are no longer useful to the answer path.

## Accuracy-flip audit

### StatsD2

| Direction | Task | Manual mechanism |
| --- | --- | --- |
| Latest only | `archeology-hard-7` | Delta uses an axis-aligned 0.1-degree box; gold uses Euclidean radius 0.1. |
| Latest only | `astronomy-hard-8` | Different target/model construction; Delta builds a much larger 15-operator path and misses both RMSEs. |
| Latest only | `environment-hard-10` | Repeated station-column parser/scope difference. |
| Latest only | `environment-hard-13` | Threshold/censored-value parsing; Delta rewrites the tidy operator four times and returns stale `12`. |
| Latest only | `wildfire-hard-12` | Same yearly start-day data, different trend decision/Yes-No interpretation. |
| Latest only | `wildfire-hard-17` | RAWS station identifier/elevation mapping and verification. |
| Latest only | `wildfire-hard-18` | Both answers are semantically the gold conclusion; evaluator/paraphrase instability. |
| Delta only | `astronomy-easy-6` | Endpoint period rate versus averaging successive interval rates. |
| Delta only | `astronomy-hard-9` | OMNI parser/AP-column/lag construction. |
| Delta only | `environment-hard-7` | Fraction (`0.11`) versus percentage points (`10.87`). |
| Delta only | `environment-hard-8` | Per-file, multi-station beach parsing versus a generic parser. |
| Delta only | `environment-hard-9` | Combined-beach layout parsing and beach-name normalization. |
| Delta only | `legal-hard-1` | Duplicate multi-state MSA aggregation/deduplication. |
| Delta only | `legal-hard-15` | Cross-state statistical-area row deduplication. |
| Delta only | `wildfire-easy-3` | State-geometry dissolve versus taking one county geometry per state. |

### SchemaOnly

| Direction | Task | Manual mechanism |
| --- | --- | --- |
| Latest only | `biomedical-easy-2` | Latest filters `Case_excluded`; Delta concatenates sheets and omits that filter. |
| Latest only | `environment-easy-2` | Delta compares rounded rates and drops borderline 2015. |
| Latest only | `environment-easy-3` | Beach identity/grouping key differs; Delta groups only by beach name. |
| Latest only | `environment-hard-8` | Beach layout/parser difference. |
| Delta only | `astronomy-hard-8` | Target alignment and train/test split differ from Latest. |
| Delta only | `astronomy-hard-9` | OMNI parser and AP lag construction. |
| Delta only | `biomedical-hard-5` | B-APM source-sheet selection versus metadata proxy. |
| Delta only | `environment-hard-12` | Per-file repeated-station parsing versus generic parser/churn. |
| Delta only | `environment-hard-13` | Threshold/censored-value parsing; the winning direction reverses from StatsD2. |
| Delta only | `environment-hard-9` | Combined-beach parser and name normalization. |
| Delta only | `wildfire-hard-12` | Trend decision/Yes-No interpretation; the winning direction reverses from StatsD2. |

Broadly, 10 flips are parser/layout/source-mapping choices, 15 are
transform/aggregation/metric/interpretation choices, and one is evaluator
instability. No flip exposes a fact available only because Delta retained an
older operator state.

## Rules supported by this comparison

### 1. Fold resolved operator revisions

After a replacement version executes successfully and a healthy child consumes
it, retain:

- current code and schema,
- current result at the normal lifecycle-selected detail,
- a compact fact such as `previous encoding error resolved with cp1252`.

Drop prior code bodies, prior samples/stats, and repeated success/error payloads.
This is the highest-confidence Delta-specific rule.

### 2. Decay settled ancestors, including monotone ones

When an operator is outside the frontier cone, error-free, unchanged, and has a
healthy materialized child, reduce its sample/stats detail. The monotone strict
cohort shows that carrying cost exists even without revisions. This comparison
supports `full result -> smaller sample/schema`; it does not causally establish
that the entire current operator can be omitted.

### 3. Retire superseded isolated probes

An isolated loader/parser probe may be reduced after a later connected operator
successfully encodes its discovered file, sheet, delimiter, or encoding. Keep a
short extracted fact rather than its table and all attempts.

### Safety guards

Do not fold or decay:

- current frontier operators or their direct inputs,
- unresolved errors or recently edited operators,
- the latest answer-bearing sink,
- a fan-out parent with any unresolved consumer,
- the only evidence for a still-open parsing, join-key, deduplication, or source-selection decision.

The four accuracy-direction reversals show why `Delta wins on task X` is not a
safe static rule. Lifecycle state is a better predicate than task identity or
operator type.

## Artifact locations

- Aggregate/manual labels: `judgment_runs/delta_vs_latest_3k/manual_audit.md`
- Existing deterministic candidates: `judgment_runs/signal_analyzer/accuracy_cases.csv`
  and `judgment_runs/signal_analyzer/cost_cases.csv`
- Existing prior manual labels: `judgment_runs/signal_analyzer/manual_validation.csv`
- Raw traces: `system_scratch/<SUT>/<task_id>/react_steps.json`
- Final DAGs: `system_scratch/<SUT>/<task_id>/workflow.json`
- Cache-aware usage: `system_scratch/<SUT>/<task_id>/stats.json`
- Scores: latest `results/<SUT>/*_measures_*.csv` per workload

## Ranked static rules and falsification gates

This section maps the evidence above to the two new implementation rules in
the `feat/agent-context-frontier-decay` worktree and ranks the next candidates.

### Rank 1: suppress redundant stats for a provably complete tiny table

**Evidence/safety:** highest. This removes a derived summary only when the full
underlying table is already visible. It does not remove rows or schema.

Exact implemented eligibility:

- total row count is known and `< 5`;
- the backend explicitly reports `truncated === false`;
- all reported rows are held by the renderer;
- the renderer did not sample away rows;
- the renderer did not truncate columns;
- the result is not being rendered as shape-only.

Action: omit `Column Schema and stats:` while leaving the complete table and
ordinary schema visible. At five rows, with unknown completeness, or under any
row/column truncation, stats remain.

Why this is fundamental: for zero through four fully visible rows, null count,
distinct count, extrema, top values, and mean are functions of the exact rows.
There is no hidden data-distribution signal to lose.

Falsification:

- A checkpointed treatment repeatedly changes a correct next action or final
  answer specifically because the model no longer sees the derived statistic.
- A supposedly eligible result is later shown to have hidden rows or columns;
  that is an implementation failure, not an acceptable trade-off.
- Cache-aware `cost_usd` rises rather than falls because the shorter block
  disrupts prompt-prefix reuse. In that case retain the semantic rule but move
  it to a cache-stable boundary.

### Rank 2: decay a settled single-consumer ancestor to three rows and schema

**Evidence/safety:** high enough for an isolated experiment, not yet a universal
default. It is the most conservative form of the consumer-barrier hypothesis.
The earlier cost audit found all 96 consumed ancestors in the strongest 27
same-step cases had one consumer. The Delta-vs-Latest strict cohort also shows
extra uncached context in `16/22` StatsD2 and `19/21` SchemaOnly cases, including
monotone workflows.

Exact implemented eligibility:

- role is neither `frontier` nor `near-frontier`;
- role is not terminal, blocked, or deleted;
- the operator has a result and no error;
- it is not a dirty loader with unresolved generic/header-like structure;
- it has exactly one outgoing consumer;
- that consumer has a result and no error;
- parent age is at least `minStepsSinceEdit` (default `1`);
- consumer age is at least `minConsumerStepsSinceEdit` (default `1`).

Action: retain current schema and a first/last sample budget of three rows, but
remove column stats. The overlay does not remove the operator, current code,
history policy, graph topology, or base-mode decorations.

Explicit protections:

- current frontier and its direct inputs;
- latest healthy leaf/answer candidates;
- all erroring, blocked, deleted, or dirty-load states;
- fan-out parents, even when one branch is healthy;
- a parent or consumer still inside its grace period.

Falsification:

- Any reproducible baseline-pass to treatment-fail transition after the agreed
  recovery rounds, manually attributable to one of the decayed rows/stats.
- At the same saved checkpoint, the treatment chooses a wrong next action,
  recreates the parent, or calls `inspectResult` for information the rule hid.
- Eligible parents are commonly edited again after decay. That falsifies the
  grace/liveness predicate even if final accuracy happens to survive.
- Median steps, tool calls, or operator revisions increase enough to erase the
  cache-aware saving.
- Same-answer both-pass cases fail the existing material saving threshold:
  at least `$0.005` and `10%` relative to the cheaper baseline on selected
  cases, with an aggregate `cost_usd` reduction rather than only fewer raw
  tokens.

Negative controls must include dirty multi-header loaders, parser recovery,
fan-out, deduplication/join-key decisions, `biomedical-easy-2`,
`wildfire-easy-3`, and the Boston Harbor repeated-column tasks.

### Rank 3: fold superseded revisions after resolution

**Evidence/safety:** high Delta-specific promise, but not implemented by the
frontier-decay overlay. Three of ten manually audited material Delta-cost
candidates follow this mechanism. `wildfire-hard-20` is the clean same-step
example: after `cp1252` succeeds, the old UTF-8 code body, large error payload,
and failed result do not need to remain verbatim.

Proposed eligibility:

- current version executes successfully;
- no unresolved error remains;
- a healthy child has consumed the current version;
- neither parent nor child changed for one grace turn;
- the operator is outside the frontier cone and has one consumer initially.

Action: keep current code/schema/result and one resolution fact, for example
`UTF-8 failed; current cp1252 loader succeeded`; fold prior code bodies,
samples, stats, and repeated error text.

Falsification:

- Recovery later depends on a detail present only in the folded error/version.
- The model repeats a resolved failed attempt because the resolution fact was
  insufficient.
- Prefix-cache loss makes the folded history more expensive, as happened in
  the earlier global static-compaction experiment.

### Rank 4: retire a superseded isolated probe

**Evidence/safety:** medium; defer until the first two rules are measured.
Eligible examples are obsolete sheet-list, file-format, or raw-parser probes
whose discovered choice is demonstrably encoded by a later connected operator.

Required guard: retain a compact extracted fact naming the chosen sheet, file,
delimiter, encoding, or key. Age and disconnection alone are insufficient.

Falsification: the model later needs the probe to distinguish sources, rebuilds
it, or makes a source-selection error. `biomedical-hard-5` is a required
negative control because source-sheet choice caused an accuracy flip.

### Do not adopt as static rules

- Do not remove current results entirely; the evidence supports reducing
  sample/stats detail, not `shape -> omit`.
- Do not decay by operator type (`source`, `integer table`, or `intermediate`)
  or task identity. Four accuracy tasks reverse the winning mode across
  information settings.
- Do not decay fan-out after only one healthy consumer.
- Do not protect every leaf as an answer forever, but do not relax that rule in
  this experiment; answer-sink identification needs separate evidence.
- Do not globally rewrite Delta history. The prior compaction run lost four
  accuracy tasks and increased cost by 22% through cache churn.

### Experiment acceptance gate

Run each rule independently before composing them. Use GPT-5.2, the requested
3,000-character sample setting, a full run, and two symmetric failed-task
recovery rounds. Freeze scores only after re-evaluation.

A rule graduates only when all of the following hold:

1. No reproducible accuracy regression attributable to the rule. A newly
   failed baseline-pass case is a hard stop until its trace is explained.
2. Same-answer cases retain the same or near-same logical dataflow; savings
   cannot come from silently doing less work or changing the interpretation.
3. `stats.json.cost_usd` decreases after accounting for cached input. Raw token
   reduction alone is insufficient.
4. Step count, tool errors, rebuilds, and inspections do not increase in a way
   that offsets the context saving.
5. A paired checkpoint probe confirms the treatment preserves the next action
   immediately before and after the eligibility transition.
