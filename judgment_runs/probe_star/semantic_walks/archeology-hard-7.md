# archeology-hard-7 — semantic walk (PROBE-STAR vintage)

## Task + gold

Q: "How many modern cities with a population of over 100k are within 0.1 degrees of
ancient Roman-era cities?" — numeric_exact, gold answer **274**.
Data: `roman_cities.csv` (1388 rows; coords in `Longitude (X)` / `Latitude (Y)`, BOM in
header), `worldcities.csv` (44691 rows; `lat`/`lng`/`population`/`id`).

Arms (all PROBE-STAR raw-probe prompt, DELTA context, code mode; one-knob diffs
confirmed from `config.json`):

| arm | knob values | answer | pass |
|---|---|---|---|
| Delta1kSchemaOnlyProbePrompt (WINNER C1p+C2p) | char_limit=1000, stats=off, D1 | **274** | ✓ |
| Delta5kSchemaOnlyProbePrompt (loser C1p) | char_limit=5000, stats=off, D1 | 295 | ✗ |
| DeltaStats1kD2ProbePrompt (loser C2p) | char_limit=1000, stats=on, D2 | 295 | ✗ |

Chronic tag: **archeology-hard-7 IS in the OLD-vintage `chronic_flippers.json`**
(advisory) — default verdict CHRONIC/VARIANCE unless accept rules are met.
Prior-vintage precedent (`judgment_runs/levers_report/semantic_walks/archeology-hard-7.md`):
numeric fingerprints re-executed on real data — L2 disc r=0.1 = **274**; L∞ box
|Δlat|≤0.1 ∧ |Δlng|≤0.1 counted by unique `id` = **295**; box-by-name = 294;
same-bin-only = 199. That vintage's verdict on the same pairs: CHRONIC/VARIANCE,
REJECTED-method-choice.

## Gold semantic plan

From `solutions/archeology/archeology-hard-7.py`:

1. Load `roman_cities.csv` (plain read_csv) → take `["Longitude (X)", "Latitude (Y)"]`.
2. Load `worldcities.csv` → filter `population > 100000` (no dropna beyond that; 5873 rows).
3. Take global `['lng','lat']`.
4. `cKDTree(roman_loc)` + `query_ball_point(global_loc, r=0.1)` — **L2 (Euclidean) disc,
   radius 0.1 deg, boundary-inclusive** (cKDTree default p=2).
5. Count global cities with ≥1 roman match, row grain (index-level, effectively per city
   row) → `len(...)` = **274**.

The single load-bearing semantic choice is item 4: the metric. "within 0.1 degrees" is
ambiguous in English; the gold fixes it as an L2 disc.

## Walk — Delta1kSchemaOnlyProbePrompt (winner, 274)

| step | action | semantics | gold item |
|---|---|---|---|
| 0 | create `raw_roman`, `raw_world` | raw-text probes: head 7 + mid 7 lines of each CSV (probe beat) | recon (pre-1/2) |
| 1 | create `roman_cities`, `worldcities`; delete both probes | `read_csv(..., encoding='utf-8-sig')` for roman (BOM seen in probe); plain `read_csv` for world | 1, 2-load ✓ |
| 2 | create `modern_over_100k` | `to_numeric(population) > 100000` → 5873×7 | 2 ✓ (5873 = gold) |
| 3 | create `modern_near_roman_0p1deg` | per modern city i: `min((r_lat−m_lat)² + (r_lng−m_lng)²) ≤ 0.1²` → **L2 disc, boundary-inclusive** → 274 ids | 4 ✓ **exact gold metric** |
| 4 | create `count_...` | `id.nunique()` → 274 | 5 ✓ |
| 5 | text | Final Answer: 274 | ✓ |

No divergence from the gold plan at any step.

## Walk — Delta5kSchemaOnlyProbePrompt (loser C1p, 295)

| step | action | semantics | gold item |
|---|---|---|---|
| 0 | create `raw_worldcities`, `raw_roman_cities` | raw-text probes: head 5 + mid 5 lines each (probe beat) | recon |
| 1 | create loads; delete probes | same two loads as winner (utf-8-sig for roman) | 1, 2-load ✓ |
| 2 | create `modern_over_100k` | `to_numeric(population) > 100_000` → 5873×7 | 2 ✓ |
| 3 | create `roman_coords` | rename to lat/lng, to_numeric, drop NaN → 1388×7 | 3 ✓ (benign) |
| 4 | create `modern_near_roman` | 0.1-deg bin join ±1 neighbor, then `|Δlat| ≤ 0.1` ∧ `|Δlng| ≤ 0.1`, dedup by `id` → **L∞ box** → 295 | **✗ diverges from item 4** |
| 5 | create `answer` | `id.nunique()` → 295 | 5 ✓ (grain fine) |
| 6 | text | Final Answer: 295 | ✗ |

**First divergence: step 4** — the metric. The bin-join candidate pruning is correct
(±1 neighbor expansion avoids the prior vintage's 199 same-bin bug); the box filter is
the sole error. 295 = L∞-box-by-id fingerprint exactly.

## Walk — DeltaStats1kD2ProbePrompt (loser C2p, 295)

| step | action | semantics | gold item |
|---|---|---|---|
| 0 | create `raw_worldcities`, `raw_roman_cities` | raw-text probes: head 5 + mid 6 lines / head 8 lines (probe beat) | recon |
| 1 | create loads; delete probes | same two loads (utf-8-sig for roman) | 1, 2-load ✓ |
| 2 | create `modern_over_100k` | `population > 100_000 & notna` → 5873×5 | 2 ✓ |
| 3 | create `modern_near_roman` | per modern city i: `any(|r_lat−m_lat| ≤ 0.1 & |r_lng−m_lng| ≤ 0.1)` → **L∞ box** → 295 ids | **✗ diverges from item 4** |
| 4 | create `count_modern_near_roman` | `id.nunique()` → 295 | 5 ✓ (by id — avoids prior vintage's 294 name-grain slip) |
| 5 | text | Final Answer: 295 | ✗ |

**First divergence: step 3** — the metric, same L∞-box-by-id fingerprint (295).

## Evidence at the divergence (rendered context, per arm)

Scanned the decision step's full `inputMessages` in each arm for anything bearing on the
metric choice: the string "within 0.1" occurs exactly **once per arm — in the question
itself**; zero hits for euclidean/circle/radius/disc/distance/box in all three contexts.

- Winner (ctx 6,777 chars) — last rendered block before writing the L2 op:
  `[modern_over_100k] Inputs: worldcities (44691 rows, 11 cols) | Output Table: 5873 rows,
  7 cols | ... 0 1392685764 Tokyo 35.6897 139.6922 ... Schema (7 cols): city (str), lng
  (numeric), ... lat (numeric)`.
- 5k loser (ctx 16,310 chars — the RICHEST context of the three): additionally rendered
  the near-full 1388-row `roman_coords` table (`... 1387 Hanson2016_1388 Seuthopolis
  Kazanlak Bulgaria Thracia 42.628866 25.274047 | Schema (7 cols): ... lng (numeric),
  lat (numeric) ...`) and still wrote the box.
- Stats loser (ctx 10,183 chars): additionally rendered D2 column stats
  (`- "lng" (numeric): null=0, mean=33.02, min=-157.8, max=178.4 | - "lat" (numeric):
  null=0, mean=23.38, min=-53.17, max=69.33 ...`) — nothing metric-disambiguating — and
  wrote the box.

All three arms had semantically equivalent decision-relevant evidence (same headers, same
coordinate schemas, same 5873 filter count); the extra bytes the losers rendered contain
nothing about the metric. The winner's evidence does NOT explain its L2 choice any better
than the losers' contexts explain theirs — the choice lives in the English phrase, not in
any renderable table. Note the losers sit on OPPOSITE sides of the winner on information
richness (5k > 1k budget; stats > schema-only), so no monotone lever story is even
possible; the shared winner trace is a single lucky L2 draw counted in two pairs.

**Probe-beat note:** all three arms executed the raw-probe beat identically (step-0
`raw_*` head+mid text previews → BOM + quoted-CSV facts → probes deleted at the load
step). Beat behavior is uniform and correct in all arms; it contributed nothing to the
divergence.

## Pair verdicts

- **C1p Delta1kSchemaOnly > Delta5kSchemaOnly: CHRONIC/VARIANCE (REJECTED-method-choice).**
  The loser's only divergence is the L∞-box reading of "within 0.1 degrees" (295 =
  box-by-id fingerprint exactly), a question-interpretation choice made at rendered
  evidence semantically equivalent to the winner's — indeed the loser rendered strictly
  MORE (full roman table at 5k). No winner-evidence/loser-absence story exists. Listed
  chronic flipper; identical mechanism and verdict as the prior-vintage C1 walk
  (3rd+ observation of the L2-vs-L∞ coin-flip on this task).
- **C2p Delta1kSchemaOnly > DeltaStats1kD2: CHRONIC/VARIANCE (REJECTED-method-choice).**
  Same box-metric coin-flip (295 = box-by-id; this vintage's stats arm even fixed the
  prior vintage's name-grain slip, landing on the same 295 as the 5k arm). The stats
  arm's extra rendered stats (lat/lng min/max/mean) contain nothing metric-relevant.
  Chronic default stands; not attributable to the stats lever.
