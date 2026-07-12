# Rank-3 / Rank-4 Static-Rule Experiments — Protocol

Date: 2026-07-10. Implements the two next-ranked static rules from
`judgment_runs/delta_vs_latest_3k/manual_audit.md` (§Ranked static rules), run
separately per its acceptance gate.

## Rules under test

### Rank 3 — fold resolved operator revisions (DELTA-only)

Implementation: `agent-service/src/agent/context/select/revision-folds.ts`
(worktree `feat-agent-context-frontier-decay`), flag `foldResolvedRevisionsConfig`.

Eligibility (per prior revision of an operator, all historical facts —
latching, never un-folds):

- a LATER landed revision of the same operator exists;
- that revision (or a successor) executed successfully (own healthy result at
  event ≥ its submission);
- the operator's SOLE leaf consumer produced a healthy result at/after that
  execution;
- `graceEvents` (default 1) have elapsed since that consumption;
- deleted / never-landed operators and fan-out (≠1 consumer) never fold.

Action: the superseded revision's Action `code:` block renders as
`(folded — superseded revision; resolved by the revision at Event N)`; its
Observation payloads render as
`(folded superseded attempt — failed: <first error line> | output was R rows;
resolved by the revision at Event N)`. Rejected submissions fold as
`(folded — rejected submission: <tool error>)`. The resolving/current revision
and all its events stay raw.

### Rank 4 — retire superseded isolated probes (LATEST)

Implementation: `agent-service/src/agent/context/selector/probe-retirement.ts`,
flag `probeRetirementConfig`.

Eligibility (per operator, evaluated on the leaf DAG):

- orphan: zero outgoing consumers;
- has a healthy current result; not dirty-load; not frontier (not the latest
  edit, not erroring); not `deleted`/`blocked`;
- ≥ `minStepsSinceEdit` (default 2) events since its last edit;
- NOT the most recently produced healthy output (answer-candidate protection);
- **discovery-encoded-downstream**: some CONNECTED operator has a landed code
  revision AFTER the probe's last edit whose quoted string literals contain a
  probe output value (string cells, ≥ `minValueLength` = 4 chars, non-numeric),
  and that operator executed healthily at/after that revision.

Action: the probe's Result body renders as its Schema line plus
`(probe retired — its R rows × C cols output is omitted: the discovery is
already encoded downstream — "<values>" used by \`<op>\` (Event N).)`.
Age + disconnection alone are NEVER sufficient (the `biomedical-hard-5`
open-source-selection guard).

## Arms

| Arm | SUT | Role |
| --- | --- | --- |
| Rank-3 control | `DataflowSystemGPT52DeltaStats3kD2FoldControl` | fresh Delta 3k StatsD2 under current code, all overlays off |
| Rank-3 treatment | `DataflowSystemGPT52DeltaStats3kD2FoldResolved` | + `foldResolvedRevisionsConfig={graceEvents:1}` |
| Rank-4 control | `DataflowSystemGPT52LatestStats3kD2SmallTableControl` | existing recovered run (same code, overlay off) |
| Rank-4 treatment | `DataflowSystemGPT52LatestStats3kD2ProbeRetire` | + `probeRetirementConfig={minStepsSinceEdit:2, minValueLength:4}` |

All arms: GPT-5.2, 3,000-char result cap, StatsD2 (flow 1 / data 2 + column
stats), 25 max steps, attempt reflection, parallel tool calls, oracle file
subsets. The rank-3 pair needed a fresh control because the historical
`DataflowSystemGPT52DeltaStats3kD2` predates the permanent small-table
stats-suppression renderer rule.

Config parity asserted by `systems/test_static_rule_configs.py` (KramaBench)
and end-to-end by the analyzer's configuration check.

## Run protocol

Driver: `run_rank34_experiments.sh` — sequential arms, each:

1. `kb.py run --sut <ARM> --parallel --watchdog-min 8` (full 104 tasks)
2. `kb.py rerun-failed --sut <ARM> --all-failed --parallel --isolate --watchdog-min 8` ×2
3. `kb.py scores --sut <ARM>` snapshot

Stack: JVM services from the repo root (copilot enabled), agent-service from
the `feat-agent-context-frontier-decay` worktree (bun, :3001), litellm :4000,
docker infra (postgres/lakefs/minio).

## Analysis

`scripts/analyze_rank34.py --control <C> --treatment <T> --rule fold|probe`:
config parity, rendered-signature activation (from each agent step's
`inputMessages`), paired accuracy (metric ≥ 0.9), fair paired cache-aware cost.

## Acceptance gate (from the audit)

1. No reproducible accuracy regression attributable to the rule (each
   baseline-pass → treatment-fail flip must be trace-explained).
2. Same-answer cases retain the same/near-same logical dataflow.
3. Cache-aware `cost_usd` decreases — raw token reduction alone is
   insufficient.
4. Steps / tool errors / rebuilds / inspections do not offset the saving.
5. (Deferred to probe infra) checkpointed next-action preservation.
