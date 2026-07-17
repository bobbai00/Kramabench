# wildfire-hard-12 — deep dive (PROBE-STAR vintage; counter-intuitive: schema-only beat stats @1k)

Counter-intuitive C2p case: **`Delta1kSchemaOnlyProbePrompt`** (schema-only,
mode Y) answered **No ✓** and **`DeltaStats1kD2ProbePrompt`** (the stats arm,
mode X) answered **Yes ✗**. One-knob validity gate passes: the arms differ only
in the C2 lever pair `column_stats` False→True (`data_level` 1→2); config diff
confirmed. Both arms are **chronic flippers** (`chronic_flippers.json`). Cost and
work are near-identical — schema $0.0199 / stats $0.0191, **5 steps each** — so
this is a pure accuracy flip, not a work gap. GPT-5.2 raw-probe prompt. Traces
via `python3 scripts/extract_walk.py --sut <ARM> --task wildfire-hard-12`.

The prior-vintage hypothesis was the **stats-trap**: a rendered `|corr| > 0.2`
line misread on n=14 noise (the 3k levers deep-dive, r=+0.2565). This dive
falsifies its recurrence at 1k: **no correlation coefficient renders in either
1k context** (`grep -c corr` on the loser's full final context = **0**), because
the 1k stats block is **univariate-only**. Verdict: **CHRONIC / rejected
method-choice** — a self-authored decision-procedure coin, not a lever effect.

## Task

Q: "Has fire start distribution around the year shifted earlier or later over
time? Answer with **only** 'Yes' or 'No'. No explanation needed."

Gold answer: **No** (`string_approximate`).

D: one CSV, `data/wildfire/input/Fire_Weather_Data_2002-2014_2016.csv` —
6,658 rows × 37 cols, one row per large US-West wildfire incident, years
2002–2014 **plus 2016** (2015 absent entirely, per the filename). Real rows
(relevant columns):

```
start_year  incident_number  state  start_date   start_day_of_year
2002        CA-ANF-3518      CA     9/22/2002    265
2002        CA-AEU-16666     CA     10/21/2002   294
2012        UT-CCD-120552    UT     7/25/2012    207
2006        OR-WWF-660       OR     8/22/2006    234
2016        35_4413752       NM     5/21/2016    142
```

Relevant columns / quirks:

- `start_year` (int, null=0) — fire-start year; **14 distinct values**; per-year
  incident count swings 270 (2014) → 817 (2006), so year-level central
  tendencies are noisy.
- `start_date` (string, US `M/D/YYYY`, null=0) — what the gold parses with
  `to_datetime`.
- `start_day_of_year` (int, 1–359, null=0) — **precomputed** day-of-year of
  `start_date`. Both arms used this ready-made column instead of parsing dates:
  same within-year-position signal as the gold's months, at day rather than
  month grain.
- The other ~33 columns (fire-weather `avrh_mean`/`wind_med`/`erc_med`, strategy
  fields, threat aggregates, lat/long, `station_verified_in_psa`) are irrelevant.

Numeric ground truth (recomputed from the raw CSV; matches the per-year tables
the arms rendered):

- r(**median** start-DOY vs year) = **+0.2565**; r(**mean** start-DOY vs year) =
  **−0.0912** — two central-tendency statistics drifting in *opposite* signs,
  neither near significance at n=14 → textbook no-signal.
- OLS slope of median start-DOY on year = **+0.404 day/yr** ≈ **+5.7 days over
  the 2002–2016 span** — a ~5-day drift on a 365-day scale, i.e. noise.
- Per-year `mean_start_doy` spans **189.7–212.1**, peaking at **2003** (the
  second year) — non-monotone → "No".

## Solution

From `solutions/wildfire/wildfire-hard-12.py` (notebook-style script), as an
operator graph:

```
load(Fire_Weather_Data_2002-2014_2016.csv)              # default read_csv → 6658x37
   → to_datetime(start_date, errors='coerce')           # US M/D/YYYY parses cleanly
   → dropna(subset=[start_date])                        # no-op: null=0
   → for y in [2002..2014, 2016]:
        months_y = df[start_year==y].start_date.dt.month     # within-year position, MONTH grain
        (mu_y, sigma_y) = norm.fit(months_y)                 # Gaussian center + spread per year
   → mu_sigma_df (14x2, index=year)                     # mu-by-year table
   → plot mu & sigma over years → EYEBALL trajectory    # G5: no coded threshold anywhere
   → mu ≈ early July every year, no drift → "No"
```

Node annotations: load spec = default `read_csv`; no filters beyond the (no-op)
NaT-drop; grain = per-`start_year`; the decisive final compute is a **visual
judgment** over the mu trajectory — **the gold never codes a decision rule**.
"Position within year" may be carried by month (gold) or day-of-year (both arms)
— equivalent signal; the load-bearing plan item is **G5, the flat/not-flat
call**.

## What DeltaStats1kD2 does (mode X — loser, answer **Yes**, FAIL)

3 tool ops + text, 5 steps, zero errors, zero re-edits.

- **step 0** `raw_preview`: reads first 15 lines → 15×1. Header sniff.
- **step 1** `fire_weather`: `pd.read_csv(...)` → 6658×37. Matches G1. **This is
  the step whose rendered observation carries the stats knob's payload** (the
  37-line `Column Schema and stats:` block — see Why).
- --> **step 2** `start_doy_trend`: `groupby('start_year')['start_day_of_year']
  .median()`, then a hand-coded OLS slope
  `slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²` → **0.4041420118343194** (1×1). **Diverges
  from G5**: over-reduces the 14-year distribution to a single scalar slope,
  discarding scale and noise — the gold *inspects the trajectory*, it never fits
  a slope.
- --> **step 3** TEXT: **Yes**. Reads its own rendered `slope 0.404` and applies
  a **nonzero** threshold (slope ≠ 0 → "shifted"). Mis-thresholds a trivial
  +0.404 day/yr (~5.7 days over 14 yr) as a real shift.

## What Delta1kSchemaOnly does (mode Y — winner, answer **No**, PASS)

3 tool ops + text, 5 steps, zero errors, zero re-edits, no near-miss to recover.

- **step 0** `raw_fire_weather`: first 15 lines → 15×2. Header sniff.
- **step 1** `fire_weather`: byte-identical `read_csv` → 6658×37. Matches G1.
- **step 2** `start_doy_by_year`: `groupby('start_year').agg(n_fires=size,
  mean_start_doy=mean, median_start_doy=median)` → **14×4** table sorted by year.
  Matches G2–G4 at day grain — and **richer than the loser**: it surfaces BOTH
  central-tendency statistics side by side, preserving the per-year distribution
  (gold-shaped).
- **step 3** TEXT: **No** — it **never codes a decision rule**. It eyeballs the
  rendered 14×4 table (the gold's own G5 judgment mode): `mean_start_doy` peaks
  at 2003 (189.7→212.1) then oscillates 195–208 with no monotone drift → flat →
  No.

## Why Delta1kSchemaOnly succeeded but DeltaStats1kD2 failed

**First divergence = step 2, and it is a self-chosen aggregation method,
identical in timing across the arms** (both act one step after the same
`fire_weather` load).

**What X's input messages actually contained** — the stats knob's *entire*
rendered contribution, verified from `react_steps.json`, is a 37-line
per-column `Column Schema and stats:` block. Its only fire-start-relevant lines
are **univariate**:

```
- "start_day_of_year" (numeric): null=0, mean=200.1, min=1, max=359
- "start_year" (numeric): null=0, mean=2008, min=2002, max=2016
```

There is **no bivariate line anywhere** — `grep -c corr` and `grep -c
relationship` over X's full final context both return **0**. So the stats block
renders no year×start-date relationship at 1k; the only start-date signal it
adds over schema-only is a mean/min/max. Entering step 3, X's own decisive
rendered evidence was the scalar it computed itself: `slope | 0
0.4041420118343194`.

**What Y's input messages contained** at its decision (step 3) — schema-only, no
stats block, the full 14×4 table rendered complete:

```
Output Table: 14 rows, 4 cols
    start_year  n_fires  mean_start_doy       median_start_doy
0   2002        403      189.6848635235732    193
1   2003        465      212.13118279569892   210
...
9   2011        565      199.8778761061947    216
13  2016        479      198.42797494780794   205
```

The full table is the whole trigger: mean peaks in year 2 and oscillates —
"flat" is the correct read, and Y answered "No".

**The stats-trap does NOT recur, two ways.** (a) *Rendered-stat reading*: the
historic trap was a `|corr| > 0.2` value **misread off a rendered stats line**;
at 1k no correlation coefficient renders (verified count 0), because the 1k
stats block is univariate-only — the misread-able artifact does not exist in
either context. (b) *Decision-procedure reading*: X didn't even reuse the 3k
loser's `|r| > 0.2` rule — it self-authored a *different* coin (OLS slope +
**nonzero** threshold). Both readings converge: **stats' harm here is
budget-dependent** — the bivariate line that could be misread only appears at
larger budgets; at 1k the flip is a self-authored decision-procedure choice, not
a lever-surfaced pseudo-statistic.

**Accept test fails.** The winner's schema-only context does not explain *why*
it built a full 14×4 table rather than a slope — a schema-only agent could
equally have fit a slope — so the loser's stats presence does not explain its
error via the one knob. The counterfactual that seals it as a coin: **X's own
nonzero test applied to Y's `mean_start_doy` gives slope −0.140 day/yr → still
"nonzero" → the threshold is meaningless either way**, and X's answer
mechanically follows whichever statistic it happened to group. The flip lives
entirely in (i) which central-tendency statistic got aggregated and
(ii) scalar-threshold vs eyeball — two free stylistic choices with no causal
path from the one rendered difference (the univariate block).

## Per-arm divergence table

| arm | first divergence | what it does wrong vs gold | answer |
|---|---|---|---|
| Delta1kSchemaOnly (Y) | step 2 `start_doy_by_year`: full 14×4 per-year table | none — matches G5 (eyeball trajectory) | **No** PASS |
| DeltaStats1kD2 (X) | step 2 `start_doy_trend`: reduce to scalar OLS slope 0.404 | replaces G5 judgment with a self-authored slope+nonzero threshold on n=14 noise | **Yes** FAIL |

**Verdict: CHRONIC / rejected method-choice.** wildfire-hard-12 is a chronic
flipper; accept rules are not met (winner-evidence explains its action, but the
loser has no evidence-absence that explains its error — the one rendered
difference is univariate and answer-irrelevant to a trend call). Any residual
effect is at most weak numeric-framing priming, indistinguishable from chronic
variance. Cost side note (not the accuracy claim): X $0.0191 vs Y $0.0199,
effectively tied.
