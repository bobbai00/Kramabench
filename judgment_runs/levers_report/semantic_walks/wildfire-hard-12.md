# Semantic walk — wildfire-hard-12

## Task + gold answer

Q: "Has fire start distribution around the year shifted earlier or later over time? Answer with **only** 'Yes' or 'No'."
Gold answer: **No**. Task is in `chronic_flippers.json` (flips between identical configs) — default verdict CHRONIC-VARIANCE unless accept rules are met.

Numeric ground truth (computed from the per-year tables both arms rendered):
- Pearson r(median start-DOY vs year) = **+0.2565** → t = 0.92, df = 12, **p ≈ 0.38** — statistically nothing at n = 14.
- Pearson r(mean start-DOY vs year) = **−0.0912** — opposite sign from the median trend.
- The two central-tendency statistics drift in opposite directions and neither approaches significance → textbook no-signal → gold "No".

Judge marks: Delta3kSchemaOnly answered **No** (PASS), DeltaStats3kD2 answered **Yes** (FAIL).
Config deltas between the arms (`config.json`): `column_stats` False→True, `data_level` 1→2 (the C2 lever pair), plus default-equivalent nulls (`enable_inspect_tool`, `enable_render_prefs`).

## Gold semantic plan

Source: `solutions/wildfire/wildfire-hard-12.py`

| # | Plan item |
|---|---|
| G1 | Load `data/wildfire/input/Fire_Weather_Data_2002-2014_2016.csv`, default `read_csv` → 6658×37 |
| G2 | `to_datetime(start_date, errors='coerce')`, drop NaT |
| G3 | Per year in [2002..2014, 2016]: take the **month-of-start** distribution (position within the year) |
| G4 | Fit Gaussian per year (`norm.fit`) → (mu, sigma); build mu/sigma-by-year table across all 14 years |
| G5 | Judge the trajectory of mu (the distribution's center) over years: essentially flat → answer **"No"** |

Note: "position within year" can be carried by month (gold) or day-of-year (both arms) — equivalent signal at different granularity; the judgment call is G5.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (WINNER)

**PASS — Final Answer: No.**

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 1 | fire_weather | `read_csv(...Fire_Weather_Data_2002-2014_2016.csv)` → 6658×37 | G1 |
| 2 | start_doy_trend | `to_numeric` start_year + start_day_of_year, dropna, keep DOY∈[1,366]; groupby(start_year) → n_fires, **mean_doy, median_doy**; sort by year → 14×4 | G2–G4 (day-of-year in place of month; same grain: per-year center of within-year position) |
| 3 | TEXT | Final Answer: **No** | G5 ✓ |

**No divergence.** The winner never coded a decision rule — it eyeballed the rendered table and judged "flat".

**Evidence at decision (final context, `react_steps.json` step 3 `inputMessages` — ALL 14 rows rendered):**
> ```
> Output Table: 14 rows, 4 cols
> 	start_year	n_fires	mean_doy	median_doy
> 0	2002	403	189.6848635235732	193
> 1	2003	465	212.13118279569892	210
> ...
> 8	2010	333	208.4024024024024	214
> 9	2011	565	199.8778761061947	216
> ...
> 13	2016	479	198.42797494780794	205
> ```
Mean oscillates 189.7–212.1 with no monotone drift (2003 is the max, year 2); the eyeball verdict "No" is correct and matches gold G5.

## Walk: DataflowSystemGPT52DeltaStats3kD2 (loser)

**FAIL — Final Answer: Yes** (gold No).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 1 | fire_weather | identical `read_csv` → 6658×37 | G1 |
| 2 | start_doy_by_year | groupby(start_year) → **median** start_day_of_year, sort → 14×2 | G2–G4 (coarser: median only, no dropna — column has null=0 anyway) |
| 3 | trend_test | `r = corr(median_start_doy, start_year)`; return `shifted := (r != 0 and abs(r) > 0.2)` → **true** | **DIVERGES from G5**: replaces "inspect the center trajectory" with an arbitrary threshold; |r| = 0.2565 barely clears 0.2 while p ≈ 0.38 — the rule fires on n=14 noise |
| 4 | TEXT | Final Answer: **Yes** (obeys its own test) | — |

**First divergence: step 3** (Agent Event 4) — the decision *rule*, not the data pipeline. Steps 1–2 are gold-faithful.

**Evidence at divergence (step-3 `inputMessages` — the loser saw the FULL median table plus stats):**
> ```
> Output Table: 14 rows, 2 cols
> 	start_year	median_start_doy
> 0	2002	193
> 1	2003	210
> ...
> 9	2011	216
> ...
> 13	2016	205
> Schema (2 cols): start_year (numeric), median_start_doy (numeric)
> Column Schema and stats:
> - "start_year" (numeric): null=0, mean=2009, min=2002, max=2016
> - "median_start_doy" (numeric): null=0, mean=204.4, min=193, max=216
> ```
At the final answer (step 4) the context additionally rendered `trend_test ... shifted | 0	true` — and still retained the full 14-row table and both stats blocks (no compaction).

## Render comparison at the loser's divergence (direction of interest)

The pair brief asked: did the stats arm's render displace/underrender something the schema-only arm saw? **Falsified, byte-for-byte:**

- Step-1 pre-observation contexts are **byte-identical** across arms (699 B each).
- The shared `fire_weather` observation is **byte-identical** in rows + schema line; the stats arm *appends* a 37-line `Column Schema and stats:` block (the arms' first rendered difference).
- The loser's 14-row median table rendered **complete** — its `median_start_doy` values are byte-equal to the winner's `median_doy` column — plus an extra stats summary line. Final contexts: winner 4.9 KB, loser 8.2 KB; nothing hit the 3k render cap, nothing was compacted out.
- Mechanism scans: `[ERROR` count 0 in both final contexts; no re-edits (each operator created once); no probes; 2–3-op DAGs (churn flag n/a).

The loser had strictly **more** rendered evidence than the winner at every aligned point, including every byte of the signal the winner used. Weak stats-side note: the profile line `min=193, max=216` happens to put the min at the first year (2002) and max at 2011 — mildly "later"-suggestive — but the loser's own operator summary shows the decision was mechanical from the corr test, not from that line.

Counterfactual that seals it as a coin flip: the loser's identical test applied to the winner's **mean_doy** column gives r = −0.09 → shifted = **false** → "No". The flip lives entirely in (a) which central-tendency statistic was grouped and (b) delegating the judgment to an arbitrary |r|>0.2 cutoff instead of eyeballing — both free stylistic choices made on top of equivalent (loser: superset) rendered evidence.

Cost side note (not the accuracy claim): loser $0.0176 vs winner $0.0134 (+31%), +1 step plus stats bytes.

## Pair verdicts

**C2 Delta3kSchemaOnly > DeltaStats3kD2 — CHRONIC-VARIANCE.**

Chronic task; accept rules not met. The winner's evidence does explain its action (full 14-row table → eyeball flat → No), but the loser has **no evidence absence** that explains its error: it rendered the complete per-year median table (byte-equal values), extra stats, and its own test output, and answered Yes anyway because its self-authored threshold test (|r|>0.2 at n=14, p≈0.38) fired on noise. The displacement/underrender hypothesis is affirmatively falsified. The divergence is a decision-procedure choice postdating the arms' first rendered difference but with no causal path from that difference (the stats block) to the choice — so not render-ATTRIBUTED; on a chronic flipper this defaults to variance. No coin-flip rerun evidence for this arm beyond the chronic tag itself; no dual-lever convergence claimed.
