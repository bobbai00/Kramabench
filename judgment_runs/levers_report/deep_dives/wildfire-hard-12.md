# wildfire-hard-12 — deep dive

Counter-intuitive case: **Delta3kSchemaOnly** (schema-only) beat **DeltaStats3kD2**
(the stats arm), whose context was a strict superset — it rendered every byte the
winner saw plus two column-stats blocks. Config delta between the arms is the C2
lever pair (`column_stats` False→True, `data_level` 1→2). The walk falsified the
displacement hypothesis byte-for-byte: the loser did not fail for lack of (or
crowded-out) evidence — it failed because it delegated the final judgment to a
self-authored `|corr| > 0.2` threshold that fired on n=14 noise (r=0.257, p≈0.38).
Label: **method-choice / CHRONIC** (task is in `chronic_flippers.json`).

## Task

Q: "Has fire start distribution around the year shifted earlier or later over
time? Answer with **only** 'Yes' or 'No'. No explanation needed."

Gold answer: **No**.

D: one CSV, `data/wildfire/input/Fire_Weather_Data_2002-2014_2016.csv` —
6,658 rows x 37 cols, one row per large US-West wildfire incident, years
2002–2014 plus 2016 (**2015 is absent entirely**, as the filename warns).
Real rows (relevant columns):

```
start_year  incident_number  state  start_date   start_day_of_year  control_year  control_day_of_year
2002        CA-ANF-3518      CA     9/22/2002    265                2002          288
2002        CA-AEU-16666     CA     10/21/2002   294                2002          294
2010        MT-SOS-000009    MT     8/21/2010    233                2010          247
2016        35_4413752       NM     5/21/2016    142                2016          210
```

Relevant columns / quirks:

- `start_year` (int, null=0) — fire-start year; 14 distinct values; per-year
  incident count swings 270 (2014) to 817 (2006), so year medians are noisy.
- `start_date` (string, US `M/D/YYYY`, null=0) — what the gold parses with
  `to_datetime`.
- `start_day_of_year` (int, 1–359, null=0) — **precomputed** day-of-year of
  `start_date`. Both arms used this ready-made column instead of parsing dates:
  same within-year-position signal as the gold's months, at day rather than
  month grain.
- The other ~33 columns (fire-weather `avrh_mean`/`wind_med`/`erc_med`/`rain_sum`,
  suppression-strategy fields, threat/injury aggregates, lat/long) are
  irrelevant to this task.

Numeric ground truth (recomputed from the raw CSV; matches the per-year tables
both arms rendered):

- r(**median** start-DOY vs year) = **+0.2565** → t = 0.92, df = 12, **p ≈ 0.38**
  — statistically nothing at n = 14.
- r(**mean** start-DOY vs year) = **−0.0912** — opposite sign.
- Gold's grain, r(Gaussian mu of start **month** vs year) = **−0.1144**; mu
  spans 6.76–7.49, i.e. every year centers in early July.
- Two central-tendency statistics drifting in opposite directions, neither
  near significance → textbook no-signal → "No".

## Solution

From `solutions/wildfire/wildfire-hard-12.py` (notebook-style script), as an
operator graph:

```
load(Fire_Weather_Data_2002-2014_2016.csv)              # default read_csv → 6658x37
   → to_datetime(start_date, errors='coerce')           # US M/D/YYYY parses cleanly
   → dropna(subset=[start_date])                        # no-op: null=0
   → per year in [2002..2014, 2016]:
        months_y = df[start_year==y].start_date.dt.month    # within-year position, MONTH grain
        (mu_y, sigma_y) = norm.fit(months_y)                 # Gaussian center + spread per year
   → mu_sigma_df (14x2, index=year)                     # mu-by-year table
   → plot mu & sigma over years → EYEBALL trajectory    # G5: no coded threshold anywhere
   → mu ≈ 6.76–7.49 (all early-July), no drift → "No"
```

Node annotations: load spec is default `read_csv`; no filters beyond NaT-drop;
grain is per-`start_year`; the final compute is a **visual judgment** over the
mu trajectory — the gold never codes a decision rule. "Position within year"
may be carried by month (gold) or day-of-year (both arms) — equivalent signal;
the decisive plan item is G5, the flat/not-flat call.

## What DeltaStats3kD2 does

Mode X — loser. FAIL, Final Answer: **Yes**. 3 operators + text, zero errors,
zero re-edits.

- `fire_weather`: `pd.read_csv('data/wildfire/input/Fire_Weather_Data_2002-2014_2016.csv')`
  → 6658x37. Matches G1.
- `start_doy_by_year`: `groupby('start_year')['start_day_of_year'].median()`
  → 14x2 `median_start_doy`, sorted by year. Matches G2–G4 at day grain —
  coarser than the winner (median only, no dropna; harmless, column has
  null=0).
- --> `trend_test`: `slope = x['median_start_doy'].corr(x['start_year']);
  return shifted := bool(slope != 0 and abs(slope) > 0.2)` → **true**.
  **Diverges from G5**: replaces the gold's "inspect the center trajectory"
  judgment with a self-authored `|r| > 0.2` cutoff on a single statistic;
  r = +0.2565 barely clears 0.2 while p ≈ 0.38 at n = 14 — the rule fires on
  noise, and 0.2 is an arbitrary constant with no support anywhere in the
  rendered context.
- TEXT: Final Answer: **Yes** — mechanically obeys its own test's rendered
  output (`shifted | 0  true`).

## What Delta3kSchemaOnly does

Mode Y — winner. PASS, Final Answer: **No**. 2 operators + text, zero errors,
zero re-edits, no near-misses to recover from.

- `fire_weather`: byte-identical `read_csv` → 6658x37. Matches G1.
- `start_doy_trend`: `to_numeric` both columns, dropna, keep DOY in [1,366];
  `groupby(start_year)` → `n_fires`, **`mean_doy`, `median_doy`**; sort by year
  → 14x4. Matches G2–G4 — and richer than the loser's pipeline: it surfaces
  BOTH central-tendency statistics side by side.
- TEXT: Final Answer: **No** — it never codes a decision rule. It eyeballs the
  rendered 14x4 table (the gold's own judgment mode, G5): mean oscillates
  189.7–212.1 with its maximum at 2003 (the second year), median wanders
  193→216→205 — no monotone drift.

## Why Delta3kSchemaOnly succeeded but DeltaStats3kD2 failed

**What Y's input messages contained at its decision** (final agent event,
context 4.9 KB, all 14 rows rendered, no stats blocks — schema-only):

```
Output Table: 14 rows, 4 cols
	start_year	n_fires	mean_doy	median_doy
0	2002	403	189.6848635235732	193
1	2003	465	212.13118279569892	210
...
8	2010	333	208.4024024024024	214
9	2011	565	199.8778761061947	216
...
13	2016	479	198.42797494780794	205
```

The full table is the whole trigger: mean peaks in year 2 and oscillates —
"flat" is the correct read, and Y answered "No".

**What X's input messages contained at its problematic step** (the
`trend_test`-authoring event, context 7.3 KB — re-verified from
`react_steps.json`):

```
Output Table: 14 rows, 2 cols
	start_year	median_start_doy
0	2002	193
1	2003	210
...
9	2011	216
...
13	2016	205
Schema (2 cols): start_year (numeric), median_start_doy (numeric)
Column Schema and stats:
- "start_year" (numeric): null=0, mean=2009, min=2002, max=2016
- "median_start_doy" (numeric): null=0, mean=204.4, min=193, max=216
```

The evidence was not merely identical — **X's was a strict superset**, and the
displacement hypothesis is affirmatively falsified byte-for-byte:

- First agent-event (pre-observation) contexts are **byte-identical** across
  the arms.
- The shared `fire_weather` observation is byte-identical in rows + schema
  line; the stats arm *appends* a 37-line `Column Schema and stats:` block —
  the arms' first rendered difference.
- X's 14-row median table rendered **complete**, its `median_start_doy` values
  byte-equal to Y's `median_doy` column, plus an extra stats summary. Final
  contexts: Y 4.9 KB, X 8.1 KB. Nothing hit the 3k render cap, nothing was
  compacted out; `[ERROR` count 0 in both; no probes, no re-edits (churn flag
  n/a on 2–3-op DAGs).
- So X held strictly MORE rendered evidence than Y at every aligned point,
  including every byte of the signal Y used to answer correctly.

The failure is therefore a **decision-procedure choice**, not an evidence gap:
facing equivalent (superset) evidence, X delegated the G5 judgment to an
arbitrary `|r| > 0.2` threshold on the one statistic it happened to group,
while Y eyeballed the trajectory the way the gold does. The counterfactual
that seals it as a coin flip: **X's own test applied to Y's `mean_doy` column
gives r = −0.0912 → `shifted = false` → "No"** — the flip lives entirely in
(a) which central-tendency statistic got grouped and (b) threshold-vs-eyeball,
both free stylistic choices. There is no causal path from the stats block (the
only rendered difference) to either choice; the one weakly suggestive stats
line (`min=193, max=216`, min at the first year, max at 2011 — mildly
"later"-flavored) is ruled out as the driver because X's answer mechanically
follows its rendered `trend_test` output.

**Verdict: method-choice / CHRONIC.** wildfire-hard-12 is a chronic flipper;
accept rules are not met (winner's evidence explains its action, but the loser
has no evidence-absence explaining its error), so this is not
render-ATTRIBUTED and defaults to variance. Cost side note (not the accuracy
claim): X $0.0176 vs Y $0.0134 (+31%), one extra step plus stats bytes.
