# Rank-3 + Rank-4 Static-Rule Experiments — Results Summary

Date: 2026-07-11. Full protocol in `protocol.md`; per-rule analyses in
`rank3_fold_paired_analysis.md` / `rank4_probe_paired_analysis.md`.
Both rules from `judgment_runs/delta_vs_latest_3k/manual_audit.md` were
implemented flag-gated in the `feat-agent-context-frontier-decay` worktree,
verified live, and run through the full 104-task GPT-5.2 benchmark with two
symmetric `--all-failed` recovery rounds per arm.

## Headline

| | Rank 3 (fold revisions, DELTA) | Rank 4 (retire probes, LATEST) |
| --- | --- | --- |
| Activation | 22/104 tasks, 62 folds | 14/104 tasks, 21 retirements |
| Accuracy | 79 → 77 (−2, none attributable) | 80 → 74 (−6; **1 plausible attributable**) |
| Cache-aware cost | **+5.68%** | +0.99% (flat) |
| Cost on activated tasks | **+21.4%** (+43% uncached) | +45.0% (selection-biased; same-step cases ≈ flat) |
| Acceptance gate | **does not graduate** (cost) | **does not graduate** (cost + 1 hard-stop flip) |

## What the experiments established

1. **Both mechanisms work exactly as designed.** Live traces show the intended
   renders (resolution facts naming the failed attempt + resolving event;
   retirement facts naming the encoded discovery + consumer). Flag-off arms
   are byte-clean (zero signature leakage; config parity 104/104).

2. **The cost failure is structural, not a bug — third replication.** Static
   compaction v3 (+22% cost), frontier decay (flat, −4.4pt cache hit), and now
   rank-3 fold (+5.7%, +21% on activated) all die the same way: an in-place
   context mutation invalidates the prompt-cache prefix, and every remaining
   step of the task re-reads the suffix uncached. Same-step activated cases
   make it airtight for rank 3: identical trajectories, fold on → +29–65%
   cost. The per-turn carrying saving never amortizes at KramaBench trajectory
   lengths (~7 steps).

3. **Two anticipated hazards were observed in the wild:**
   - *Un-latching* (rank 3, `astronomy-easy-1`): a late fan-out broke the
     sole-consumer predicate and un-folded 6 folds at the final step — the
     audit's non-monotone-eligibility concern, realized.
   - *Re-opened source selection* (rank 4, `astronomy-hard-8`): CDF
     variable-inventory probes retired after their chosen variable was encoded
     downstream; the model then produced degenerate zero RMSEs with the
     inventory hidden (control hit gold exactly). "Settled" discoveries can
     re-open under downstream failure, and the fact line does not carry the
     full inventory.

4. **The guards that were supposed to hold, held.** `biomedical-hard-5` (the
   canonical open-source-selection negative control) activated AND passed —
   treatment-only win. No agent retried a folded resolved attempt outside one
   both-fail max-steps spiral where the control loops identical code on its
   own operators too.

## Direction

The audit's falsification clause named the remedy in advance: *"retain the
semantic rule but move it to a cache-stable boundary."* The semantics
(resolution facts, retirement facts) are validated and cheap; the delivery
mechanism (free-running mid-trajectory rewrite) is what loses. Concretely:

- fold resolved revisions **inside the edit-convergence compaction deck** —
  when the deck folds anyway, superseded revisions become resolution facts
  instead of stats blocks (no extra cache break);
- apply retirement/fold state **only at trajectory boundaries that already
  rewrite** (task phase changes, compaction events), latched permanently;
- for list-shaped probes, the retirement fact must carry the full small
  inventory, not just the matched values.

## Artifacts

- Arms: `system_scratch/DataflowSystemGPT52DeltaStats3kD2FoldControl`,
  `…FoldResolved`, `…LatestStats3kD2ProbeRetire`
  (rank-4 control: `…LatestStats3kD2SmallTableControl`)
- Driver + phase logs: `logs/rank34-20260710_223905/`
- Analyzer: `scripts/analyze_rank34.py`
- Agent-service implementation (worktree `feat-agent-context-frontier-decay`):
  `context/select/revision-folds.ts`, `context/selector/probe-retirement.ts`
  + params/server plumbing and tests
- KramaBench plumbing: `dataflow_agent.py`, `systems/dataflow_system.py`,
  `systems/test_static_rule_configs.py`
