# wildfire-easy-9 — semantic walk (PROBE-STAR, raw-probe prompt)

**C3p, A-only pair.** Winner **`…Delta5kSchemaOnlyProbePrompt` → -0.0053 (✓,
rae 0.908)** vs loser **`…Latest5kSchemaOnlyProbePrompt` → -0.0240 (✗, rae
0.246)**. One knob: `context_mode` delta vs latest (both schema-only, 5k,
raw-probe prompt). **Not chronic** (confirmed absent from
`chronic_flippers.json`) — strict accept rules; a Latest miss here is genuine.
**Question:** "How many more or less fatalities occurred due to wildfires on
days with humidity less than 30% compared to the average? … Round to 4 dp."
**Gold: -0.0059** (`numeric_approximate`).

## Gold plan
`solutions/wildfire/wildfire-easy-9.py`: load
`Fire_Weather_Data_2002-2014_2016.csv` **at incident-row grain** (6658 rows,
one row per incident) → `low = df[df['avrh_mean']<30]['fatalities_last'].mean()`
→ answer = `low − df['fatalities_last'].mean()` = **-0.0059**. **No daily
aggregation** — the metric is computed directly over incident rows.

## Per-arm divergence table
| arm | grain of the compute | rows fed to the mean | answer | vs gold |
|---|---|---|---|---|
| **Delta (win)** | incident row (as gold) | 6658 | -0.0053 | ✓ within approx tol (MAE 0.0006) |
| Latest (lose) | **daily** (groupby `start_year`+`start_day_of_year`) | 2195 | -0.0240 | ✗ MAE 0.0181 |

- Delta `fatality_humidity_diff` (step 1): `fat[humid<30].mean() -
  fat.mean()` straight over the 6658 rows → **-0.00529** → "-0.0053". (The
  -0.0053 vs gold -0.0059 gap is a benign `fillna(0)` on `fatalities_last`,
  inside the approximate tolerance.)
- Latest `fatalities_by_day` (**step 2 = the hinge**): `groupby(['start_year',
  'start_day_of_year']).agg(fatalities=('fatalities_last','sum'),
  avrh_mean=('avrh_mean','mean'))` → collapses 6658 incidents into **2195
  daily rows** — a grain the gold plan does not have. Step 3 then takes the
  diff over those daily means → **-0.0240**. Summing fatalities per day and
  re-averaging humidity is the entire error.

## Evidence at decision time (both arms)
Symmetric. Both rendered the same `fire_weather` load — **6658×37,
schema-only**, same header list exposing both `start_year` and
`start_day_of_year` (which Latest seized for the groupby) and
`fatalities_last`/`avrh_mean`; **no values** in either. The raw-probe (step 0)
was **benign in both**: it printed head lines truncated at ~110 chars —
`fatalities_last` is column 23, past the cutoff — so the probe confirmed only
delimiter/header and surfaced nothing about grain (Latest's probe even read
extra middle rows and still exposed no fatality values). Nothing rendered to
one arm and not the other.

## Diagnosis of "what broke" (task's question)
- **Not churn:** Latest has 4 ops (< 8 threshold), no resubmission/thrash,
  low sink-share.
- **Not probe overhead:** single benign header probe, truncated before the
  answer columns.
- **Not wrong column:** both arms use the correct `avrh_mean` + `fatalities_last`.
- **It is wrong grain / over-aggregation:** Latest added a spurious
  daily-rollup step, reading the question's "**on days** with humidity <30%"
  as a daily-aggregation instruction. The failure literally *rode on Latest's
  extra step* (6 st / $0.023 vs Delta 4 st / $0.0156) — the fuller-context arm
  over-elaborating a grain the gold computes at row level.

## Verdict — REJECTED-method-choice (wrong-grain interpretation)
Both arms had identical schema-only evidence and an identical benign probe;
no rendered difference distinguishes the delta and latest contexts at the
hinge. The divergence is a question-interpretation call ("days" → daily
groupby vs row filter), not a lever effect. The context_mode↔extra-step
correlation is directional only (LATEST's known tendency to elaborate), not an
attribution. Not chronic, so a real wrong answer — but **not attributable to
the compaction knob**.
