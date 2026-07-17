# archeology-hard-7 — deep dive (counter-intuitive: the LEAST-informed mode won)

PROBE-STAR vintage (all arms raw-probe prompt, delta history, code mode; one-knob diffs
verified from `config.json`). The leanest render config beat both better-informed ones:
**Delta1kSchemaOnlyProbePrompt** (mode Y: 1k char limit, no stats, D1) answered **274 =
gold**, while **Delta5kSchemaOnlyProbePrompt** (5k budget) and **DeltaStats1kD2ProbePrompt**
(column stats + D2) both answered **295**. `archeology-hard-7` is in
`chronic_flippers.json` (old-vintage set, advisory here), so the default verdict is
CHRONIC/VARIANCE unless the accept rules are met. Verdict up front: **both pairs
CHRONIC/VARIANCE — method-choice on a distance metric that no render lever can carry.**
This is the 3rd+ observation of the same coin-flip on this task (prior vintage:
`judgment_runs/levers_report/deep_dives/archeology-hard-7.md`, same mechanism, same
verdict). Every fingerprint below was re-executed on the real data with `.venv/bin/python`.

| Arm | role | char_limit | col_stats / data_level | agent steps | input tok | cost_usd | answer | verdict |
|---|---|---|---|---|---|---|---|---|
| **Delta1kSchemaOnlyProbePrompt** (Y) | **WINNER** (C1p + C2p A-only) | 1k | false / 1 | 6 | 50,373 | 0.0320 | **274** | PASS |
| Delta5kSchemaOnlyProbePrompt (X, C1p) | loser | **5k** | false / 1 | 7 | 82,917 | 0.0504 | 295 | FAIL |
| DeltaStats1kD2ProbePrompt (X, C2p) | loser | 1k | **true / 2** | 6 | 56,648 | 0.0347 | 295 | FAIL |

## Task
Q: "How many modern cities with a population of over 100k are within 0.1 degrees of ancient Roman-era cities?"

D: two CSVs under `data/archeology/input/`.

`roman_cities.csv` — **1388 rows × 12 cols**, Hanson 2016 gazetteer of ancient Roman-era
city sites. Comma sep, header row 0, **UTF-8 BOM on the first header cell** (`﻿Primary
Key`) — the probe beat surfaced it and all three arms loaded with `encoding='utf-8-sig'`.
Real rows (relevant cols):

```
Primary Key    Ancient Toponym  Modern Toponym  Country  Longitude (X)  Latitude (Y)
Hanson2016_1   Abae             Kalapodi        Greece   22.933333      38.583333
Hanson2016_2   Acharnae         Acharnes        Greece   23.734088      38.083473
Hanson2016_3   Acraephia        Akraifnio       Greece   23.219702      38.452606
Hanson2016_4   Aegina           Aigina          Greece   23.4285        37.750074
```

- `Longitude (X)` / `Latitude (Y)` (numeric): site coordinates in decimal degrees, with
  explicit **(X, Y) = (lng, lat)** labeling — longitude-first, opposite the other file.
  **Zero NaNs in either coordinate column** (verified) — every `dropna()` on them is a no-op.
- Other cols (`Province`, `Barrington Atlas Rank` e.g. `"4 or 5"`, `Start Date`,
  `End Date` mostly NULL) are irrelevant here.

`worldcities.csv` — **44691 rows × 11 cols**, SimpleMaps modern-city database, fully
quoted CSV (every field in `"..."`, numerics included — the probe beat showed this too).
Real rows (relevant cols):

```
city       lat      lng       country    population  id
Tokyo      35.6897  139.6922  Japan      37732000    1392685764
Jakarta    -6.1750  106.8275  Indonesia  33756000    1360771077
Delhi      28.6100  77.2300   India      32226000    1356872604
Guangzhou  23.1300  113.2600  China      26940000    1156237133
```

- `lat` / `lng` (numeric): decimal degrees, lat-first naming (the one pairing trap vs the
  roman file's X/Y order).
- `population` (numeric): **307 NaNs**; strict `> 100000` → **5873 qualifying rows** (verified).
- `id` (int): **unique per row** (verified) — distinct-`id` count ≡ row count.
- `city` (str): NOT unique (178 duplicate names among the 5873) — irrelevant this vintage,
  both losers counted by `id`.

The load-bearing ambiguity is in the QUESTION, not the data: **"within 0.1 degrees"** can
be an **L2 (Euclidean) disc of radius 0.1** or an **L∞ (Chebyshev) box** `|Δlat| ≤ 0.1 AND
|Δlng| ≤ 0.1`. The box circumscribes the disc (corner reach 0.1·√2 ≈ 0.141) → strict
superset. Nothing in either file disambiguates it. Re-executed fingerprints: **L2 disc =
274 (gold) · L∞ box by id = 295 · box by name = 294 · same-bin-only box = 199.** Both
losers landed exactly on the box-by-id fingerprint.

## Solution
From `solutions/archeology/archeology-hard-7.py`, as an operator graph:

```
read_csv(roman_cities.csv)  ─── roman_loc = (Longitude (X), Latitude (Y))   # 1388 pts, lng-first
                                        │
read_csv(worldcities.csv) ── filter(population > 100000) ── global_loc = (lng, lat)   # 5873 pts
                                        │
                                        ▼
        cKDTree(roman_loc).query_ball_point(global_loc, r=0.1)   # L2 disc, radius 0.1, <= inclusive
                                        │   a modern city qualifies if ANY roman pt is inside its disc
                                        ▼
        count qualifying worldcities ROWS  (row grain; id unique ⇒ id-count equivalent)
                                        │
                                        ▼
                                    answer = 274
```

Annotated: **load** both files, defaults (BOM absorbed) · **filter** strict
`population > 100000` → 5873 · **pairing** consistent (lng, lat) both sides · **proximity**
= L2 disc `r=0.1`, boundary-inclusive, any-roman-neighbor · **grain** = qualifying
worldcities row / unique `id` · **output** integer 274. The single load-bearing semantic
choice is the metric: gold fixes "within 0.1 degrees" as an L2 disc.

## What Delta5kSchemaOnlyProbePrompt does (mode X, C1p — 295, FAIL)
- **step 0** `raw_worldcities`, `raw_roman_cities`: raw-text probes, head 5 + mid 5 lines
  each (probe beat). Surfaces the BOM and the fully-quoted CSV. Recon ✓.
- **step 1** `worldcities`, `roman_cities` + **delete both probes**: plain `read_csv` for
  world, `encoding='utf-8-sig'` for roman. Loads ✓ (plan 1–2).
- **step 2** `modern_over_100k`: `to_numeric(population) > 100_000` → **5873×7**. Filter
  cardinality exactly gold's ✓ (plan 2).
- **step 3** `roman_coords`: rename `Longitude (X)→lng`, `Latitude (Y)→lat`, `to_numeric`,
  dropna → 1388×7. Benign prep ✓ (coordinate NaNs are zero).
- **--> step 4** `modern_near_roman`: bins both sides at `round(x/0.1)`, expands roman
  candidates over the full **±1 neighbor offset set** (correct pruning — avoids the prior
  vintage's 199 same-bin bug), then filters
  `(lat_m − lat_r).abs() <= 0.1` AND `(lng_m − lng_r).abs() <= 0.1`,
  `drop_duplicates(subset=['id'])` → **295**. **Divergence from the solution's proximity
  node:** the predicate is an **L∞ box**, not gold's L2 disc — a superset admitting corner
  cities up to ~0.141° away. Binning and `id` grain are both correct; **the metric is the
  entire gap** (box-by-id = 295 exactly, re-executed).
- **step 5** `answer`: `id.nunique()` → 295. Grain ✓.
- **step 6** Final Answer: **295**.

## What DeltaStats1kD2ProbePrompt does (mode X, C2p — 295, FAIL)
- **step 0** `raw_worldcities`, `raw_roman_cities`: raw-text probes (head 5 + mid 6 /
  head 8 lines). Recon ✓.
- **step 1** `worldcities`, `roman_cities` + **delete both probes**: same two loads as the
  other arms (`utf-8-sig` for roman). Loads ✓.
- **step 2** `modern_over_100k`: `population > 100_000 & notna` → **5873×5**. Filter ✓.
- **--> step 3** `modern_near_roman`: brute-force per modern city over all roman points,
  predicate `np.any((dlat <= 0.1) & (dlng <= 0.1))` with `dlat/dlng = np.abs(...)` →
  **L∞ box** → 295 ids. **Same divergence from the proximity node, same box-by-id
  fingerprint** — and this arm even counts by `id`, fixing the prior vintage's 294
  name-grain slip, which lands it on exactly the 5k loser's answer.
- **step 4** `count_modern_near_roman`: `id.nunique()` → 295. Grain ✓.
- **step 5** Final Answer: **295**.

## What Delta1kSchemaOnlyProbePrompt does (mode Y, WINNER — 274, PASS)
- **step 0** `raw_roman`, `raw_world`: raw-text probes, head 7 + mid 7 lines each. Recon ✓.
- **step 1** `roman_cities`, `worldcities` + **delete both probes**: `read_csv(...,
  encoding='utf-8-sig')` for roman (BOM seen in the probe), plain `read_csv` for world.
  Loads ✓ (plan 1–2).
- **step 2** `modern_over_100k`: `to_numeric(population) > 100000` → **5873×7**. Filter ✓.
- **step 3** `modern_near_roman_0p1deg`: brute-force per modern city, predicate
  `((r_lat − m_lat[i])**2 + (r_lng − m_lng[i])**2).min(...) <= (0.1**2)` — **L2 disc,
  radius 0.1, boundary-inclusive**, any-roman-neighbor; collects unique `id` → **274**.
  **Identically gold's metric** (no KD-tree, same predicate).
- **step 4** `count_modern_near_roman_0p1deg`: `id.nunique()` → **274**. Grain ✓.
- **step 5** Final Answer: **274**.

No divergence from the gold plan at any step — and this is the arm with strictly the
LEAST rendered evidence of the three.

## Why Y succeeded but X failed
**The rendered evidence at each arm's divergence step was semantically identical to the
winner's — this is method-choice on the question's English, not a lever effect.** The
decision separating 274 from 295 is *which metric "within 0.1 degrees" denotes*, and that
phrase occurs exactly **once per arm's decision context — in the question itself**
(verified scan of the full `inputMessages` at each divergence step; zero hits for
euclidean/radius/disc/box/chebyshev in all three). No renderable table carries the metric.

What each arm had rendered at its metric-writing step:

- **Winner (ctx 6,777 chars), before its L2 step:**
  `[modern_over_100k] Inputs: worldcities (44691 rows, 11 cols) | Output Table: 5873 rows,
  7 cols | 0 1392685764 Tokyo 35.6897 139.6922 37732000 Japan Tōkyō | ... | Schema (7
  cols): city (str), lng (numeric), ... lat (numeric)`
- **5k loser (ctx 16,310 chars — the RICHEST of the three), before its box step:** the
  same filtered block PLUS the near-full 1388-row roman table:
  `[roman_coords] Output Table: 1388 rows, 7 cols | 0 Hanson2016_1 Abae Kalapodi Greece
  Achaea 38.583333 22.933333 | ... | 1387 Hanson2016_1388 Seuthopolis Kazanlak Bulgaria
  Thracia 42.628866 25.274047 | Schema (7 cols): ... lng (numeric), lat (numeric) ...`
- **Stats loser (ctx 10,183 chars), before its box step:** the same schema/count facts
  PLUS D2 column stats on both tables —
  `- "lng" (numeric): null=0, mean=33.02, min=-157.8, max=178.4` /
  `- "lat" (numeric): null=0, mean=23.38, min=-53.17, max=69.33` (filtered moderns; the
  raw worldcities pair mean=14.53/25.93 also rendered) — marginal column shape, nothing
  about a *pairwise distance metric*.

All three arms saw the same decision-relevant facts: the 5873 filter count and the
coordinate schemas. The losers' extra bytes (more roman rows; lat/lng min/max/mean)
contain nothing metric-disambiguating — and symmetrically, the winner's context contains
nothing that explains its L2 choice better than the losers' contexts explain their box.
Note the losers sit on **opposite sides of the winner in information richness** (5k > 1k
budget; stats+D2 > schema-only), so no monotone lever story is even constructible: the
shared winner trace is a single lucky L2 draw counted in two pairs.

**Probe-beat note:** the raw-probe beat was executed identically and correctly in all
three arms — step-0 `raw_*` head+mid text previews (surfacing the BOM and full quoting),
both probes deleted at the load step. Uniform across arms, orthogonal to the divergence;
it contributed nothing to the flip. The probe lever here beat neither uniformly rendered
alternative — it was simply irrelevant to the one decision that mattered.

## Pair verdicts
- **C1p Delta1kSchemaOnly > Delta5kSchemaOnly: CHRONIC/VARIANCE (REJECTED-method-choice).**
  The loser's only divergence is the L∞-box reading of "within 0.1 degrees" (295 =
  box-by-id fingerprint exactly, re-executed), made at rendered evidence semantically
  equivalent to the winner's — indeed a strict superset (full roman table at 5k). No
  winner-evidence/loser-absence story exists. Listed chronic flipper; same mechanism and
  verdict as the prior-vintage C1 walk.
- **C2p Delta1kSchemaOnly > DeltaStats1kD2: CHRONIC/VARIANCE (REJECTED-method-choice).**
  Same box-metric coin-flip, same 295 fingerprint (this vintage's stats arm even fixed
  the prior vintage's name-grain slip). The stats lever's entire rendered delta —
  lat/lng/population marginals — is metric-irrelevant. Not attributable to the stats
  lever; chronic default stands.
