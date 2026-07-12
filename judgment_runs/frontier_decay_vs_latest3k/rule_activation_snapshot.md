# Rule Activation Snapshot: Latest 3k Control vs Frontier Decay

Generated 2026-07-10 from the frozen pre-recovery copies:

- Control: `full_pass_control_scratch/`
- Treatment: `full_pass_treatment_scratch/`

The live `system_scratch` directories were undergoing `rerun-failed
--all-failed` recovery while this audit was written. This report therefore
answers whether the rules fired and gives a stable initial cost snapshot; it is
not the final recovered accuracy comparison.

## Configuration check

There are 91 tasks with configs in both frozen arms. Every matched
`agent_settings` object is identical except for `frontier_decay_config`:

| Arm | `frontier_decay_config` |
| --- | --- |
| Control | `null` |
| Treatment | `sampleRows=3`, all three grace periods `=1` |

Both use GPT-5.2, Latest context, a 3,000-character result cap, StatsD2,
column stats, 25 maximum steps, reflection, and parallel tool calls. The
complete-table-under-five renderer rule is in the current server code and is
therefore shared by both arms.

## Detection method

The audit parses every agent step's serialized `inputMessages` and separates
the rules using their rendered signatures.

Tiny-table suppression requires:

- `Output Table: N rows` with `N < 5`;
- all `N` row indices visibly present and no row-gap marker;
- a typed `Schema` line, proving column metadata existed;
- no `Column Schema and stats:` block.

Frontier decay requires:

- `N >= 5`, a schema, a row-gap marker, at most three visible rows, and no
  stats block;
- an earlier rendering of the same task/operator/table shape with full stats.

That prior-full to later-decayed transition rules out missing backend stats as
the explanation. The same frontier signature occurs zero times in control.

## Rule A: complete table under five rows

| Measure | Control | Treatment |
| --- | ---: | ---: |
| Completed ReAct traces | 90 | 98 |
| Rule-signature renderings | 499 | 367 |
| Distinct task/operator/result renderings | 146 | 143 |
| Tasks with at least one firing | 75 (83.3%) | 87 (88.8%) |

Representative control evidence:

```text
archeology-easy-11, agent step 3, capital_avg_lat
Output Table: 1 rows, 1 cols
    avg_lat
0   17.162443697478988
Schema (1 cols): avg_lat (numeric)
```

The typed schema is present, the exact one-row answer is visible, and the stats
block is absent. Because frontier decay is disabled in control, this is direct
evidence that the permanent renderer rule fired.

Historical pre-rule traces provide the only direct approximation of the text
that is now absent. The baseline audit found 147,013 stats characters across
440 comparable repeated renderings, about 36,753 token occurrences using the
repository's `chars / 4` approximation, or 0.59% of historical baseline input.
The new control has a similar number of firings, so the order of magnitude is
credible, but it is not a billed-token estimate.

Attribution consequence: this rule cannot explain a treatment-versus-control
difference. It is deliberately present in both arms. Different firing counts
only show that independent runs constructed different tiny tables.

## Rule B: stable frontier decay

| Measure | Treatment |
| --- | ---: |
| Tasks with a proven transition | 46 / 98 (46.9%) |
| Distinct decayed task/operator results | 101 |
| Repeated decayed renderings | 267 |
| DataLoading operators | 67 |
| DataProcessing operators | 34 |
| Unique transitions removing stats and rows | 87 |
| Unique transitions removing stats only | 14 |

The stats-only cases are usually wide tables already limited to two or three
rows by the 3,000-character cap. Frontier decay still removes their large
per-column stats block.

Direct text estimate from each treatment operator's preceding full rendering:

| Removed component | Characters | `chars / 4` token occurrences |
| --- | ---: | ---: |
| Column stats | 246,348 | 61,587 |
| Additional sample rows | 158,789 | 39,697 |
| **Total** | **405,137** | **101,284** |

On the 89 tasks with paired cost artifacts, 41 treatment tasks fire the rule.
Their repeated decays remove an estimated 361,503 characters, or 90,376 token
occurrences.

Activation is concentrated where expected:

| Workload | Tasks | Operators | Renderings | Approx. token occurrences removed |
| --- | ---: | ---: | ---: | ---: |
| Archeology | 6 | 10 | 21 | 9,134 |
| Astronomy | 4 | 9 | 12 | 5,506 |
| Biomedical | 5 | 11 | 23 | 10,908 |
| Environment | 10 | 38 | 128 | 44,936 |
| Legal | 8 | 9 | 23 | 7,051 |
| Wildfire | 13 | 24 | 60 | 23,748 |

Representative transitions:

- `archeology-easy-11/worldcities`, step 3 to 4: 44,691 x 11 table,
  13 to 3 visible rows, 819 stats characters removed.
- `astronomy-easy-6/quiet`, step 3 to 4: 8 x 40 table, already limited to two
  visible rows, but 4,222 stats characters removed.
- `biomedical-hard-1/global_prot`, step 5 to 6: 10,999 x 154 table, rows remain
  capped at two while 3,404 stats characters disappear.

Five operators later re-expand to full stats when their graph/activity state
changes (`biomedical-hard-8` has three, plus `legal-hard-23` and
`legal-hard-18`). This confirms that decay is recomputed; it is not permanent
destruction of operator detail.

## Can the observed A/B cost gap be credited to the rules?

Not from the aggregate run.

Frozen paired usage over 89 tasks:

| Measure | Control | Treatment | Treatment - Control |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $3.737657 | $3.693239 | -$0.044418 (-1.2%) |
| Input tokens | 5,282,424 | 4,678,368 | -604,056 |
| Cached input | 4,621,184 | 3,939,968 | -681,216 |
| Uncached input | 661,240 | 738,400 | +77,160 |
| Output tokens | 126,556 | 122,253 | -4,303 |
| Steps | 623 | 598 | -25 |

The paired frontier-decay estimate of about 90,376 token occurrences is only
15% of the observed 604,056-input-token reduction. Treatment also takes 25
fewer steps. Therefore most of the raw-token gap is behavioral trajectory,
not static rendering.

The cache result is also adverse: treatment has fewer input tokens but 77,160
more uncached input. Its apparent total cost saving is only 1.2%, much smaller
than the raw-token reduction.

The affected-task aggregate is dominated by different behavior:

- 41 affected tasks sum to `$0.286774` lower treatment cost, but their median
  task is `$0.006718` *more* expensive in treatment.
- Large wins come from runs with many fewer steps, for example
  `wildfire-easy-3` (19 to 8), `archeology-hard-1` (24 to 11), and
  `environment-hard-17` (24 to 8).
- The 48 paired tasks without a frontier firing sum to `$0.242356` higher
  treatment cost, showing that independent-run variance is large enough to
  nearly cancel the affected group.

Only 23 paired tasks have the same normalized answer, step count, and coarse
workflow shape. Just one of them, `wildfire-hard-14`, actually fires frontier
decay, so that cohort cannot estimate a general treatment effect.

### `wildfire-hard-14`: clean firing, adverse cache economics

Both arms answer `0.42` with five steps and a four-operator/three-link pipeline.
At the final response step, treatment decays `aqi_2024`:

```text
986 x 18 table: 8 rows + 1,301 stats chars
             -> 3 rows + no stats
direct removal: about 1,626 chars (407 token occurrences)
```

Treatment uses 1,617 fewer aggregate input tokens, so a real raw-context saving
is plausible. However it also receives 6,016 fewer cached tokens and therefore
4,399 more uncached tokens. Cost rises from `$0.018524` to `$0.024064`.

This case demonstrates why raw input reduction is not sufficient: the rule
fires and shortens context, but cache placement can reverse the dollar result.

### `wildfire-hard-20`: apparent accuracy difference is unrelated

Both arms take seven steps and have a five-operator/three-link shape. Treatment
uses 2,428 fewer input tokens, close to its roughly 2,272-token frontier-decay
text estimate, but costs `$0.013501` more because cache reuse falls.

Control answers `0.0465`; treatment answers `0.0317`. The treatment bug is in
the final full-detail operator: it uses the sorted frame's original index via
`(csum >= threshold).idxmax() + 1` instead of a positional count. Decay affects
older `wildfires`/`wf_2008` results, while the direct `wf2008_sorted` input is
still protected. The accuracy difference is therefore a transform/index bug,
not evidence that frontier decay hid decisive information.

## Conclusion

Both new rules demonstrably fire:

- Tiny-table stats suppression is broad but small and shared by both arms.
- Frontier decay fires on nearly half of completed treatment traces and removes
  materially more rendered text, especially on settled sources and wide tables.

The frozen A/B aggregate does not yet establish cache-aware cost savings.
Trajectory and step-count differences dominate, and the clearest same-behavior
firing loses money through cache alignment. Final judgment must wait for both
recovery rounds and should emphasize checkpointed same-trajectory probes or
instrumented per-operator rendered-token counters rather than aggregate raw
tokens alone.
