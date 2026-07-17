# wildfire-hard-12 — semantic walk (PROBE-STAR, raw-probe prompt)

**C2p A-only flip.** Winner `DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt`
(No, ✓) vs loser `DataflowSystemGPT52DeltaStats1kD2ProbePrompt` (Yes, ✗).
One knob: `column_stats` false→true (+ `data_level` 1→2). Both **chronic
flippers** (`*`). Cost near-identical: schema $0.0199 / stats $0.0191, 5 steps
each. **Question:** "Has fire start distribution around the year shifted
earlier or later over time? Yes/No." **Gold: No.**

## Gold plan
`solutions/wildfire/wildfire-hard-12.py`: load Fire_Weather CSV → per-year
`norm.fit` of fire-start **month** (mu, sigma) across the 14 years
(2002–2014, 2016) → inspect whether mu (mean start month) trends. It does not
→ **No** (distribution stable year-to-year; ~5-day drift on a 365-day scale is
noise).

## Per-arm divergence table
| step | arm | action | semantics | vs gold |
|---|---|---|---|---|
| 0–1 | both | raw preview → `fire_weather` load | 6658×37 CSV | ✓ ok |
| 2 | **winner** | `start_doy_by_year` | groupby start_year → **table** of n_fires + mean_start_doy + median_start_doy (14×4) | preserves the per-year distribution (gold-shaped) |
| 3 | **winner** | TEXT | reads the 14-row table (means bounce 189.7↔212.1, no monotonic trend) → **No** | ✓ correct |
| 2 | loser | `start_doy_trend` | groupby start_year median → fit **one linear slope** of median-vs-year = **0.404** | over-reduces to a scalar; discards scale/noise |
| 3 | loser | TEXT | slope ≠ 0 → **Yes** | ✗ mis-thresholds a trivial 0.404 day/yr (~5.6 days/14 yr) as a "shift" |

**First divergence = step 2**, and it is a *self-chosen aggregation method*,
identical in timing across arms (both act one step after the same
`fire_weather` load).

## Evidence at decision time
- **Winner (schema-only)** saw only schema entering step 2. Its own step-2
  output (rendered entering step 3) is the decisive evidence: `mean_start_doy`
  = 2002:189.7, 2003:212.1, 2004:198.3, 2005:209.1 … 2013:197.3, 2014:194.2,
  2016:198.4 — visibly bouncing, no trend → No.
- **Loser (stats)** saw the mmc-style column stats block entering step 2. The
  only start-date signal rendered was **univariate**:
  `"start_day_of_year" (numeric): null=0, mean=200.1, min=1, max=359` and
  `"start_year" (numeric): mean=2008, min=2002, max=2016`. **No correlation /
  no year×start_doy relationship is rendered** at 1k. The loser then *computed
  its own* slope (0.404) and read nonzero → Yes.

## Verdict: prior mechanism does NOT recur — REJECTED-method-choice / CHRONIC-VARIANCE
The prior-vintage failure was a `|corr|>0.2`-on-noise **read off the rendered
stats**. That cannot happen here: the 1k stats block renders only per-column
mean/min/max — **no correlation coefficient exists in either context**. Both
arms independently chose their step-2 method; the loser's error is a
self-generated scalar-slope + nonzero-threshold *interpretation* mistake, not a
misread of a rendered pseudo-statistic. The winner's schema-only context does
not explain *why* it built a full table (a schema-only agent could equally have
fit a slope), so the loser's stats presence does not explain its error via the
one knob. On a **chronic** task with equal cost and no rendered-evidence
difference bearing on the year-trend decision, this fails the accept test
(winner-evidence-explains-action AND loser-absence-explains-error). Any residual
effect is at most weak "numeric-framing" priming — indistinguishable from
chronic variance.
