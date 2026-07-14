# archeology-hard-7 — deep dive (counter-intuitive: the LEAST-informed mode won)

The leanest render config of four beat three better-informed ones. **Delta3kSchemaOnly**
(mode Y, 3k char limit, no column stats, delta history) answered **274 = gold**, while
**Delta5kSchemaOnly** (295), **DeltaStats3kD2** (294), and **Latest3kSchemaOnly** (199) all
failed — including the arm that rendered strictly *more* (Stats). `archeology-hard-7` is in
`chronic_flippers.json`, so the default verdict is CHRONIC/VARIANCE unless the accept rules
are met. Verdict up front: **all three pairs CHRONIC/VARIANCE — method-choice on a metric
that no render lever can carry.** Every number below was re-executed on the real data with
`.venv/bin/python`.

| Arm | role | context_mode | col_stats / data_level | char_limit | steps | cost_usd | answer | verdict |
|---|---|---|---|---|---|---|---|---|
| **Delta3kSchemaOnly** (mode Y) | **WINNER** | delta | false / 1 | 3k | 6 | 0.0252 | **274** | PASS |
| Delta5kSchemaOnly (X, C1) | loser | delta | false / 1 | **5k** | 6 | 0.0344 | 295 | FAIL |
| DeltaStats3kD2 (X, C2) | loser | delta | **true / 2** | 3k | 6 | 0.0260 | 294 | FAIL |
| Latest3kSchemaOnly (X, C3) | loser | **latest** | false / 1 | 3k | 6 | 0.0241 | 199 | FAIL |

## Task
Q: "How many modern cities with a population of over 100k are within 0.1 degrees of ancient Roman-era cities?"

D: two CSVs under `data/archeology/input/`.

`roman_cities.csv` — **1388 rows × 12 cols**, Hanson 2016 gazetteer of ancient
Roman-era city sites. Default `read_csv` (comma sep, header row 0; a UTF-8 BOM sits on
the first header cell but pandas absorbs it). Real rows (relevant cols):

```
Primary Key    Ancient Toponym  Modern Toponym  Country  Longitude (X)  Latitude (Y)
Hanson2016_1   Abae             Kalapodi        Greece   22.933333      38.583333
Hanson2016_2   Acharnae         Acharnes        Greece   23.734088      38.083473
Hanson2016_3   Acraephia        Akraifnio       Greece   23.219702      38.452606
Hanson2016_4   Aegina           Aigina          Greece   23.428500      37.750074
```

- `Longitude (X)` / `Latitude (Y)` (numeric): the site coordinates in decimal degrees.
  Note the **explicit (X, Y) = (lng, lat) labeling** — the columns are longitude-first,
  which the pairing must respect. **Zero NaNs in either coordinate column** (verified), so
  any `dropna()` on the roman coords is a no-op.
- Other cols (`Province`, `Barrington Atlas Rank` = e.g. `"4 or 5"`/`"3"`, `Start Date`,
  `End Date` = mostly NaN) are irrelevant to this task.

`worldcities.csv` — **44691 rows × 11 cols**, SimpleMaps world-cities database, one row per
modern city. Default `read_csv`. Real rows (relevant cols):

```
city       lat      lng       country    population  id
Tokyo      35.6897  139.6922  Japan      37732000    1392685764
Jakarta    -6.1750  106.8275  Indonesia  33756000    1360771077
Delhi      28.6100  77.2300   India      32226000    1356872604
Guangzhou  23.1300  113.2600  China      26940000    1156237133
```

- `lat` / `lng` (numeric): modern-city coordinates in decimal degrees (lat-first naming here,
  opposite the roman file's X/Y labels — the pairing order is the one modeling trap in the
  coordinate prep).
- `population` (numeric): **307 NaNs**; filter is strict `> 100000` → **5873 qualifying rows**.
- `id` (int): **unique per row** (verified) — so counting distinct `id` == counting rows.
- `city` (str): **NOT unique** — among the 5873 qualifiers there are 178 duplicate names.
  This is the quirk that separates the 294-vs-295 losers (below).

The phrase **"within 0.1 degrees"** is geometrically ambiguous: it can mean a **Euclidean
(L2) disc of radius 0.1** or a **Chebyshev (L∞) box of half-width 0.1** (`|Δlat|≤0.1 AND
|Δlng|≤0.1`). The box circumscribes the disc, so it is a strict superset. **Nothing in any
arm's rendered data disambiguates this** — the two files carry only coordinates and counts;
the metric is decided from the question wording alone, which all four arms received identically.
Gold defines it as L2.

## Solution
From `solutions/archeology/archeology-hard-7.py`, as an operator graph:

```
read_csv(roman_cities.csv)  ─── roman_loc = (Longitude (X), Latitude (Y))  # 1388 pts, lng-first
                                        │
read_csv(worldcities.csv) ── filter(population > 100000) ── global_loc = (lng, lat)  # 5873 pts
                                        │
                                        ▼
        cKDTree(roman_loc).query_ball_point(global_loc, r=0.1)   # L2 disc, radius 0.1, <= inclusive
                                        │  a global city qualifies if ANY roman pt is inside its disc
                                        ▼
        count global ROWS with ≥1 match  (row grain; id is unique so id-count is equivalent)
                                        │
                                        ▼
                                    answer = 274
```

Annotated: **load** roman + worldcities, defaults · **filter** `population > 100000` (strict,
→ 5873) · **coordinate pairing** consistent (lng, lat) both sides · **proximity** = L2 disc
`r=0.1`, boundary-inclusive, any-roman-neighbor · **grain** = one qualifying worldcities row
each (row / unique-`id`, **not** distinct city name) · **output** integer 274.

## What Delta5kSchemaOnly does (mode X, C1 — 295, FAIL)
- **step 0** `worldcities`, `roman_cities`: `read_csv` both, defaults. Load ✓ (plan 1, 2).
- **step 1** `modern_100k`: `population.fillna(0) > 100000`, keep `id,city,lat,lng,population`
  → **5873×5**. Filter cardinality exactly matches gold ✓ (plan 3).
- **step 2** `roman_coords`: rename `Longitude (X)→lng`, `Latitude (Y)→lat`, `to_numeric`,
  keep numeric rows → 1388×5. Benign coordinate prep ✓ (plan 4).
- **--> step 3** `near_roman_modern`: bins each side at `round(x/0.1)`, generates candidates
  over the **correct full ±1 neighbor offset set** `[(i,j) for i,j in (-1,0,1)²]`, then filters
  `|Δlat| ≤ 0.1 AND |Δlng| ≤ 0.1` and takes `id.nunique()` → **295**. **Divergence from plan 5:**
  the predicate is an **L∞ box** of half-width 0.1, not gold's L2 disc of radius 0.1. The box
  circumscribes the disc, admitting corner cities up to `0.1·√2 ≈ 0.141` away → superset → 295.
  The ±1 binning and `id` grain are both correct; **the metric is the entire gap** (box-by-id
  = 295 exactly, verified).
- **step 4** Final Answer: **295**.

## What DeltaStats3kD2 does (mode X, C2 — 294, FAIL)
- **step 0** `worldcities`, `roman_cities`: `read_csv` both, defaults. Load ✓ (plan 1, 2).
- **--> step 1** `modern_over_100k`: `(population > 100000) & population.notna()`, keeps only
  `city,lat,lng,population` → **5873×4**. Cardinality is right, but it **drops `id`** — the
  unique row key — leaving only the non-unique `city` name to count on later. This sets up the
  step-3 grain error (near-miss it never recovered from).
- **--> step 2** `near_roman_modern`: full cross join (`_k=1`, 5873×1388 pairs), then
  `|Δlat| ≤ 0.1 AND |Δlng| ≤ 0.1` → 347 (city, roman) pairs. **Divergence from plan 5:** same
  **L∞ box** metric as C1 (box-by-id = 295).
- **--> step 3** `count_modern_near_roman`: **`city.nunique()`** → **294**. **Second divergence
  from plan 6:** counts distinct city *names*, not rows/ids. Exactly one box qualifier is a
  duplicate name — **Tripoli, Libya (id 1434201852) and Tripoli, Lebanon (id 1422886065)** —
  so name-dedup collapses 295 → **294** (verified). Two compounding modeling choices (L∞ box +
  name grain), both made against the same evidence the winner had — and this is the arm that
  rendered the *most*.
- **step 4** Final Answer: **294**.

## What Latest3kSchemaOnly does (mode X, C3 — 199, FAIL)
- **step 0** `roman_cities`, `worldcities`: `read_csv` both, defaults. Load ✓ (plan 1, 2).
- **step 1** `modern_over_100k`: `population.fillna(0) > 100000`, keep `id,city,lat,lng,population`
  → **5873×5**. Filter ✓ (plan 3).
- **--> step 2** `modern_near_roman`: bins each side at `round(x/0.1)`, merges **only on equal
  bins** (`on=['lat_bin','lng_bin']`, **no ±1 neighbor offsets**), then box-filters
  `|Δlat| ≤ 0.1 AND |Δlng| ≤ 0.1`, `id`+`city` distinct → **199×2**. **Divergence from plan 5:**
  a genuine spatial-candidate-**pruning bug** — same-bin-only joining silently discards any
  neighbor that landed in an adjacent 0.1° cell (e.g. lng 0.04 vs 0.06 → bins 0 vs 1, true
  distance 0.02, dropped). C1 wrote the *same* bin scheme *with* the ±1 offsets; C3 simply
  omitted the offset loop. The L∞ box metric is also wrong, but the pruning dominates:
  same-bin-only + box = **199** exactly (verified) — −96 vs the box answer, −75 vs gold.
- **step 3** `count_modern_near_roman`: `id.nunique()` → 199. Grain is correct here (plan 6).
- **step 4** Final Answer: **199**.

## What Delta3kSchemaOnly does (mode Y, WINNER — 274, PASS)
- **step 0** `roman_cities`, `worldcities`: `read_csv` both, defaults. Load ✓ (plan 1, 2).
- **step 1** `modern_over_100k`: `to_numeric(population) > 100000`, keep
  `id,city,lat,lng,population` → **5873×5**. Filter ✓ (plan 3).
- **step 2** `modern_near_roman`: brute-force per modern city over all roman rows, predicate
  **`(r_lat − lat)² + (r_lng − lng)² <= 0.1²`** — i.e. **L2 disc, radius 0.1, boundary-inclusive**,
  `.any()` over roman pts; collects `id`, `drop_duplicates` → **274×1**. Matches plan 4-6 exactly.
  (The roman `dropna()` is a no-op — 0 NaN coords.) No KD-tree, but the predicate is
  *identically* gold's.
- **step 3** `count_modern_near_roman`: `id.nunique()` → **274**. Row/id grain ✓ (plan 6, 7).
- **step 4** Final Answer: **274**.

No divergence anywhere. Note the winner is the **leanest** of the four render configs (3k,
schema-only, delta) — it succeeded with strictly *less* rendered evidence than every loser.

## Why Y succeeded but X failed
**The rendered evidence at each arm's divergence step was identical to the winner's — this is
method-choice, not a lever effect.** I state this honestly rather than manufacture a context
story. The decision that separates 274 from 295/294/199 is *which distance metric "within 0.1
degrees" means* (and, for C3, a same-bin coding slip) — and that decision is made from the
question text, which all four arms received verbatim, against schema/count observations that
did not differ in any answer-relevant way.

What was rendered just before each divergence:

- **Winner, before its L2 step:**
  `[modern_over_100k] Output 5873x5: Inputs: worldcities (44691 rows, 11 cols) | ... | id  city  lat  lng  population | 0  1392685764  Tokyo  35.6897  139.6922  37732000`
- **C1 (Delta5k), before its L∞ step:**
  `[modern_100k] Output 5873x5: ... | id  city  lat  lng  population | 0  1392685764  Tokyo  35.6897  139.6922  37732000`
  and `[roman_coords] Output 1388x5: ... | Primary Key  Ancient Toponym  Modern Toponym  lat  lng | 0  Hanson2016_1  Abae  Kalapodi  38.583333  22.933333`
- **C2 (Stats3kD2), before its L∞ step:**
  `[modern_over_100k] Output 5873x4: Inputs: worldcities (44691 rows, 11 cols) | ... | city  lat  lng  population | 0  Tokyo  35.6897  139.6922  37732000`
- **C3 (Latest3k), before its same-bin step:**
  `[modern_over_100k] Output 5873x5: Summary: From worldcities, filter to modern cities with population > 100000 and keep id, city, lat, lng, population | ... | 0  1392685764  Tokyo  35.6897  139.6922  37732000`

Every arm saw the same thing: the 5873 filtered count and the coordinate schema. **Why no
render lever could have flipped the losers:**

- **More rows (Delta5k's 5k char limit).** A distance metric is a property of the *formula*,
  not of any sample row. Showing more of the 5873×5 preview surfaces additional (city, lat,
  lng) tuples, none of which tell you whether "within 0.1°" is a disc or a box. C1 in fact had
  the *largest* budget and produced the superset answer — extra rows moved it away from gold,
  not toward it.
- **Column stats (DeltaStats3kD2).** This arm rendered **strictly more** — 192 stats-ish
  tokens in its trace vs 7 for the winner (verified: `column_stats:true, data_level:2`).
  Distribution/null/distinct stats on `lat`, `lng`, `population`, `id` describe the marginal
  shape of each column; they say nothing about the *pairwise distance metric* between two
  coordinate sets, and nothing that would prompt row-grain over name-grain. The most-informed
  arm made *two* wrong modeling choices — the direct refutation of a context-evidence attribution.
- **History (Delta vs Latest).** Delta's per-op history and Latest's last-observation view
  differ only in what prior steps are re-shown. C3's failure is a coding slip *inside a single
  step* (omitting the ±1 bin offsets), and the observation rendered *after* it —
  `[modern_near_roman] Output 199x2: ... | id  city | 0  1792756324  Istanbul` — is a plausible
  table with **no error, no anomaly, nothing to react to**. There is no prior step whose
  re-rendering would flag that a bin-equality join is geometrically lossy. C3 ran clean in 6
  steps with no re-edits, no `[ERROR`, no probe thrash, no sink-share churn — nothing a richer
  context could have surfaced and repaired.

The L2-vs-L∞ distinction lives in the semantics of the English phrase, not in any table the
engine can render — so it is orthogonal to every knob these arms vary (rows, stats, history,
budget). The winner drew the disc; the losers drew the box (C1, C2) or pruned the grid (C3).
That is an unforced modeling coin-flip on an ambiguous question, sampled here as one lucky
draw for the lean arm.

## Pair verdicts
- **C1 Delta3k > Delta5k: CHRONIC/VARIANCE.** Loser's only divergence is the L∞-box reading of
  "within 0.1 degrees" (295 = box-by-id exactly), a method choice made from the question text at
  rendered evidence equivalent to the winner's; it predates any rendered difference between the
  arms. Listed chronic flipper. **REJECTED-method-choice.**
- **C2 Delta3k > Stats3kD2: CHRONIC/VARIANCE.** Same box metric plus a name-grain dedup
  (294 = box-by-name exactly, the Tripoli Libya/Lebanon collision). The stats arm rendered
  strictly more than the winning schema-only arm and still made *both* modeling errors — no
  winner-evidence / loser-absence story exists. **REJECTED-method-choice.**
- **C3 Delta3k > Latest3k: CHRONIC/VARIANCE.** Same-bin-only join (199 = same-bin-box exactly)
  is a spatial-candidate-pruning coding bug committed at identical rendered evidence, with no
  error/thrash/churn signal afterward for any render mode to surface; not attributable to the
  Latest-vs-Delta or budget lever. **REJECTED-method-choice.**
