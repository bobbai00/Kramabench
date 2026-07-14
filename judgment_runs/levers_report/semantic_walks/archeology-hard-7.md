# Semantic walk — archeology-hard-7

## Task + gold answer

Q: "How many modern cities with a population of over 100k are within 0.1 degrees of ancient Roman-era cities?"
Gold answer: **274**. Task is in `chronic_flippers.json` (flips between identical configs) — default verdict CHRONIC/VARIANCE unless accept rules are met.

Numeric ground truth (re-executed on the real data, `.venv/bin/python`):
- L2 disc r=0.1 (gold `cKDTree.query_ball_point`, boundary-inclusive): **274**
- L∞ box |Δlat|≤0.1 AND |Δlng|≤0.1, counted by unique `id`: **295** — exactly Delta5k's answer
- L∞ box counted by distinct city NAME (`city.nunique()`): **294** — exactly Stats3kD2's answer
- Same-bin-only join (round(x/0.1), no ±1 neighbor expansion) + box filter: **199** — exactly Latest3k's answer
- `roman_cities` coordinates have 0 NaNs → winner's `dropna()` is a no-op vs gold.

The question's "within 0.1 degrees" is ambiguous between L2 disc and L∞ box; the gold defines it as L2. Nothing in any arm's rendered data speaks to the metric — it is decided from the question wording alone, which all four arms had identically.

## Gold semantic plan

1. Load `data/archeology/input/roman_cities.csv`, default `read_csv` (cols incl. `Longitude (X)`, `Latitude (Y)`); 1388 rows.
2. Load `data/archeology/input/worldcities.csv`, default `read_csv` (cols incl. `lat`, `lng`, `population`, `city`, `id`); 44691 rows.
3. Filter worldcities: `population > 100000` (strict >) → 5873 rows.
4. Coordinate pairs: roman = (`Longitude (X)`, `Latitude (Y)`); global = (`lng`, `lat`) — consistent (x, y) order.
5. Proximity predicate: `cKDTree(roman).query_ball_point(global, r=0.1)` = **Euclidean L2 distance ≤ 0.1** in degree space; a global city qualifies if ANY roman city is within the disc.
6. Grain: count each qualifying worldcities ROW once (row indices; `id` is unique per row so id-dedup is equivalent; NOT distinct city names).
7. Output: integer count → 274.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (WINNER)

**PASS — Final Answer: 274.**

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | roman_cities, worldcities | `read_csv` both files, defaults | 1, 2 |
| 1 | modern_over_100k | `to_numeric(population) > 100000`, keep id/city/lat/lng/pop → 5873×5 | 3 |
| 2 | modern_near_roman | per modern city: `((r_lat−lat)² + (r_lng−lng)² <= 0.1²).any()` over all roman rows (roman coords dropna = no-op); collect ids, dedup → 274×1 | 4, 5, 6 |
| 3 | count_modern_near_roman | `id.nunique()` → 274 | 6, 7 |
| 4 | TEXT | Final Answer: 274 | 7 |

No divergence. Alternate path (brute-force loop instead of KD-tree) but the predicate is exactly gold's: L2, radius 0.1, boundary-inclusive (`<=`). Note the winner is the *leanest* render config of the four (3k schema-only) — it did not have extra evidence.

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (loser, C1)

**FAIL — Final Answer: 295** (gold 274).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | worldcities, roman_cities | `read_csv` both, defaults | 1, 2 |
| 1 | modern_100k | `population.fillna(0) > 100000` → 5873×5 | 3 |
| 2 | roman_coords | rename to lat/lng, `to_numeric`, keep numeric rows → 1388×5 | 4 (prep, benign) |
| 3 | near_roman_modern | 0.1-degree bins with correct ±1 neighbor offsets, then **`|Δlat| ≤ 0.1 AND |Δlng| ≤ 0.1`** (L∞ box), `id.nunique()` → 295 | **DIVERGES from 5** |
| 4 | TEXT | Final Answer: 295 | — |

**First divergence: step 3** — distance predicate is a Chebyshev/L∞ square of half-width 0.1 instead of gold's L2 disc of radius 0.1. The square circumscribes the disc → superset → 295 vs 274. Everything else (filter cardinality 5873, ±1 bin candidate generation, id grain) matches gold; the metric is the entire gap (verified: box-by-id = 295 exactly).

**Evidence at divergence (rendered before step 3):**
> `[modern_100k] Output 5873x5: ... | id	city	lat	lng	population | 0	1392685764	Tokyo	35.6897	139.6922	37732000`
> `[roman_coords] Output 1388x5: ... | Primary Key	Ancient Toponym	Modern Toponym	lat	lng | 0	Hanson2016_1	Abae	Kalapodi	38.583333	22.933333`

Winner's evidence before its distance step was the same schema/count information (`[modern_over_100k] Output 5873x5 ...`). **No evidence gap** — the L2-vs-L∞ reading comes from the question text, identical in both arms.

## Walk: DataflowSystemGPT52DeltaStats3kD2 (loser, C2)

**FAIL — Final Answer: 294** (gold 274).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | worldcities, roman_cities | `read_csv` both, defaults | 1, 2 |
| 1 | modern_over_100k | `(population > 100000) & notna`, keep city/lat/lng/pop — **drops `id`** → 5873×4 | 3 (cardinality OK; key dropped, sets up step-3 grain error) |
| 2 | near_roman_modern | cross join (`_k=1`), then **`|Δlat| ≤ 0.1 AND |Δlng| ≤ 0.1`** (L∞ box) → 347 (city, roman) pairs | **DIVERGES from 5** |
| 3 | count_modern_near_roman | **`city.nunique()`** → 294 | **DIVERGES from 6** (name grain, not row/id grain) |
| 4 | TEXT | Final Answer: 294 | — |

**First divergence: step 2** — same L∞ box metric as Delta5k (box by id = 295). **Second divergence: step 3** — counting distinct city NAMES instead of rows/ids loses 1 duplicate-named city (295 → 294; verified). Two compounding modeling choices, both made against the same evidence the winner had.

**Evidence at divergence (rendered before step 2):**
> `[modern_over_100k] Output 5873x4: Inputs: worldcities (44691 rows, 11 cols) | ... | city	lat	lng	population | 0	Tokyo	35.6897	139.6922	37732000`

Nothing metric- or grain-relevant. This is the stats-enriched arm — it rendered MORE than the winner and still chose the wrong metric, which argues directly against a context-evidence attribution.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (loser, C3)

**FAIL — Final Answer: 199** (gold 274).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | roman_cities, worldcities | `read_csv` both, defaults | 1, 2 |
| 1 | modern_over_100k | `population.fillna(0) > 100000` → 5873×5 | 3 |
| 2 | modern_near_roman | bins `round(x/0.1)`, join **only on equal bins (no ±1 neighbor expansion)**, then box filter `|Δlat| ≤ 0.1 & |Δlng| ≤ 0.1` → 199×2 | **DIVERGES from 5** |
| 3 | count_modern_near_roman | `id.nunique()` → 199 | 6, 7 (grain correct) |
| 4 | TEXT | Final Answer: 199 | — |

**First divergence: step 2** — a genuine candidate-pruning bug: joining only on identical 0.1-degree bins misses neighbors in adjacent cells (e.g. lng 0.04 vs 0.06 → bins 0 and 1, distance 0.02, pruned). Delta5k wrote the same bin scheme WITH the ±1 offsets; Latest3k omitted them. Same-bin-only + box = 199 exactly (verified). The box metric is also wrong, but the pruning dominates (−96 vs the box answer, −75 vs gold).

**Evidence at divergence (rendered before step 2):**
> `[modern_over_100k] Output 5873x5: Summary: From worldcities, filter to modern cities with population > 100000 and keep id, city, lat, lng, population | ... | 0	1392685764	Tokyo	35.6897	139.6922	37732000`

Identical information to the winner's. And the observation rendered AFTER step 2 gave no repair signal — a plausible table, no error:
> `[modern_near_roman] Output 199x2: ... | id	city | 0	1792756324	Istanbul`

No render variant (Latest vs Delta, 3k vs 5k, stats or not) could have flagged that a bin-equality join is geometrically lossy; the failure is a code-generation slip, and the trace shows no error/thrash/churn (5 steps, no re-edits) that a richer context could have averted.

## Pair verdicts

- **C1 Delta3k > Delta5k: CHRONIC/VARIANCE.** The loser's only divergence is the L∞-box reading of "within 0.1 degrees" (295 = box exactly), a method choice made from the question text at rendered evidence equivalent to the winner's (same schemas, same 5873 count); it predates any rendered difference between the arms, and the task is a listed chronic flipper.
- **C2 Delta3k > Stats3kD2: CHRONIC/VARIANCE.** Same box-metric choice plus a name-grain dedup (294 = box-by-name exactly); the stats arm rendered strictly more than the winning schema-only arm and still made both modeling errors, so there is no winner-evidence/loser-absence story — chronic default stands.
- **C3 Delta3k > Latest3k: CHRONIC/VARIANCE.** The loser's same-bin-only join (199 = same-bin-box exactly) is a spatial-candidate-pruning coding bug committed at identical rendered evidence, with no error signal afterward for any render mode to surface; not attributable to the Latest-vs-Delta or budget lever.
