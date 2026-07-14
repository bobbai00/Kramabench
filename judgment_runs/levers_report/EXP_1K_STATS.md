# Experiment: profiling at a starved render budget — Delta1kSchemaOnly vs DeltaStats1kD2

One-knob pair at the **1k** result-char cap (the starvation end of the
sampling axis), recovery-first (full + 2× `--all-failed` each arm, symmetric).
The knob: `column_stats` + `data_level=2` (Output Table profile) ON vs OFF —
identical otherwise (config-parity verified). Question: does the profile
lever **substitute** for sample rows when the render budget is too small to
show them?

## Headline

| arm | pass | % | total $ | $/task |
|---|---|---|---|---|
| Delta1kSchemaOnly | 70/104 | 67.3% | $5.27 | 0.0512 |
| **DeltaStats1kD2** | **75/104** | **72.1%** | $6.35 | 0.0617 |

**At 1k, profiling buys +5 tasks (+4.8 pts) of accuracy, for +21% cost.**

Compare the SAME knob at 3k (C2 in the levers report):

| cap | schema-only pass | stats pass | stats accuracy delta |
|---|---|---|---|
| 3k | 80 (76.9%) | 83 (79.8%) | **+3** |
| 1k | 70 (67.3%) | 75 (72.1%) | **+5** |

Two things move together: starving the cap 3k→1k costs **~10 pts on both
arms** (rows carry real information), and the **stats advantage grows +3 → +5**
as the budget starves. Profiling matters *more* when you can't afford sample
rows — it substitutes for them.

## Venn

```
Delta1kSchemaOnly-only: 4    both pass: 66    DeltaStats1kD2-only: 9    both fail: 25
```
both-pass cost split: schema-only cheaper on 42 (saves $1.52), stats cheaper
on 24 (saves $0.34) → schema-only is the cheaper arm overall, as expected
(stats adds render bytes).

## Where the stats accuracy gain comes from (B-only, 9 tasks)

The 9 tasks stats wins and schema-only loses are the **dirty / weird-format,
tiny-source** family — exactly the CASE_METRICS F2/F5 prediction:

- files med **5KB / 58 rows** (vs schema-only-wins 220KB/1,388 rows),
  **61% dirty**, formats `.csv:14 .txt:5 .lst:2 .xlsx:2 .cdf:1` (5 of 9 are
  non-plain-CSV), **22% multi-edit** (parse iteration).
- includes environment-hard-8, environment-hard-9, biomedical-easy-2,
  astronomy-hard-8, legal-easy-19 — beach datasheets (unnamed headers,
  high-null), sheet/format-quirk loads.

Mechanism: at 1k the sample-row window is too small to reveal an unnamed-header
/ multi-station / wrong-sheet structure, so schema-only gropes (higher
multi-edit, more sinks) and often fails; the profile lines (`headers: N of M
unnamed`, per-column nulls, duplicate counts) deliver that structure in a
fixed, cap-independent number of bytes. This is the profile acting as an
**anti-iteration substitute** precisely where sampling is starved.

The 4 tasks schema-only wins (A-only) are the opposite: big clean CSVs
(220KB/1,388 rows, 5% multi-edit) where the profile is pure noise/tax and its
extra bytes crowd the 1k budget.

## Caveats

- Exclusive-win cells are small (4 / 9) and several are chronic flippers
  (`*` in the venn) — the +5 aggregate is within ~2× the twin-noise band, so
  treat "+5" as "clearly positive, order-of-a-few," not exact. The
  *direction* is what matters and it is consistent with the 3k→1k trend and
  the file-family concentration.
- Cost is cache-aware `cost_usd`. Stats' +21% is the render-byte premium at
  1k (smaller than at 3k in absolute $ because the base render is tiny).

## Takeaway for the paper

This de-confounds the 3k "stats is pure tax" reading: **profiling's accuracy
value is budget-dependent.** With enough sample rows (3k+) the profile is
near-redundant (+3, mostly cost); starve the rows (1k) and it becomes a
genuine +5 substitute, concentrated on the dirty/weird-format sources where
rows would have carried the structure. Sampling and profiling are
**substitutable evidence channels for the same structural facts**, trading
off at a budget-dependent rate — the cleanest support yet for per-operator
delivery (cheap rows when clean, profile when dirty/starved) over a global
default.
