# wildfire-hard-17 — flip attribution walk

Anchor `DataflowSystemGPT52Delta3kSchemaOnly` WINS both pairs
(C1 vs `Delta5kSchemaOnly`, C3 vs `Latest3kSchemaOnly`). Task is on
`chronic_flippers.json` — both pairs annotated chronic*. The Latest arm is the
levers-report churn poster child (25 ops / 20 sinks / 25 agent steps / $0.189 /
no answer, vs the anchor's 9 ops / 12 steps / $0.093).

Arm stats (`stats.json` + final `workflow.json`):

| arm | result | answer | agent steps | ops | sinks | cost_usd |
|---|---|---|---|---|---|---|
| Delta3kSchemaOnly (anchor) | PASS | 4793.1 | 12 | 9 | 5 | $0.093 |
| Delta5kSchemaOnly (C1 loser) | FAIL | 3317.4 | 16 | 12 | 9 | $0.149 |
| Latest3kSchemaOnly (C3 loser) | FAIL | none (step cap) | 25 | 25 | 20 (80%) | $0.189 |

## Task + gold answer

QUESTION: "What is the average elevation (in feet) of the weather stations
used for fire site monitoring in the NOAA dataset?"

GOLD ANSWER: **4830.9**

## Gold semantic plan

From `solutions/wildfire/wildfire-hard-17.py`:

1. **Sources**: `noaa_wildfires_sylvia.csv` (= the input `noaa_wildfires.csv`,
   6658 fire incidents × 37 cols) + `PublicView_RAWS_-3515561676727363726.csv`
   (2965 RAWS station-metadata rows × 34 cols). Plain `read_csv`, no special
   load params.
2. **Fire-side key**: `used_stations = fires['station_verified_in_psa'].unique()`
   → 779 numeric ids, range 20107–482106.
3. **Station-side key: `NWS ID`**, coerced `pd.to_numeric(errors='coerce')` —
   the column is high-null (28.3% NaN; 2127/2965 numeric, range 1196–661001,
   the ONLY RAWS id column overlapping the NOAA id space).
4. **Filter**: stations where `NWS ID ∈ used_stations` → **759 of 779** match.
5. **Final compute**: `Elevation.dropna().mean()` over matched stations,
   1 dp → **4830.9 ft**.

Load-time key detail: RAWS carries FIVE id-like columns — `WX ID` and
`Station ID` are 8-digit (16,777,528–19,907,847: disjoint from the NOAA ids),
`NESS ID` is mostly hex strings (457 numeric), `MesoWest Station ID` is
alphanumeric, and only `NWS ID` joins. The task is a join-key discovery
problem on a high-null column. **Trap shared by all three arms**: the single
rendered sample row of `raws_publicview` (row 0, YNP PORTABLE) has
`NWS ID = NaN` — the gold key's only visible sample value looked empty in
every arm's context from step 0 onward.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (WINNER — PASS, answer 4793.1)

Passed inside grader tolerance: 4793.1 vs gold 4830.9 (−0.78%), via an
off-plan geo approximation, not via the gold key.

| step | action | semantics (from code) | matches gold? |
|---|---|---|---|
| 0 | load `noaa_wildfires_vars`, `noaa_wildfires`, `raws_publicview` | 3 sources; vars fails UTF-8 → loader hint rendered | ✓ (plan 1) |
| 1 | reload vars `encoding='cp1252'` | 37×3 variable descriptions | ✓ (aux) |
| 2 | `noaa_station_ids`: to_numeric, dropna, unique, sort | 779 ids | ✓ (plan 2) |
| 3 | `stations_with_elev`: left-join ids on RAWS **`WX ID`** (renamed) | 779 rows, Elevation ALL NaN | ✗ plan 3 — **first divergence** (wrong key) |
| 4 | edit `stations_with_elev` → join on **`Station ID`** | also all-NaN (seen at step 6) | ✗ plan 3 (wrong key #2) |
| 5 | `raws_id_ranges` probe | WX ID 16777533–19907847 (2960 distinct); Station ID 16777528–19891205 (2916); Elevation −229–12120 | probe |
| 6 | `avg_station_elevation_ft` from `stations_with_elev` | NaN | consequence of ✗ |
| 7 | `noaa_station_id_ranges` probe | 779 ids, 20107–482106 | probe |
| 8 | **PIVOT**: `noaa_station_elev_from_noaa` — per unique station id take fire `latitude/longitude`, find nearest RAWS station (haversine) | NameError (`raws_publicview` missing from signature) | ~ plan 3/4 approximated by geography |
| 9 | fix signature, BallTree k=1 | 779×5; row0 nearest_raws_km 1.98, Elevation_ft 950 | ~ plan 4 (proxy population) |
| 10 | `avg_station_elevation_ft` ← mean of `Elevation_ft` | 4793.1 | ≈ plan 5 (approximation) |
| 11 | Final Answer: 4793.1 | — | PASS (0.78% off gold) |

**First divergence: step 3 — join key `WX ID` instead of gold's `NWS ID`.**
Never repaired: the winner never probes `NWS ID` either (its step-5 range
probe SELECTED the `NWS ID`/`NESS ID` columns but only computed ranges for
WX ID / Station ID / Elevation). It recovers by matching each fire's
coordinates to the nearest RAWS station — even though the fire lat/lon are
integer-degree coarse (rendered row 0: `34 -118`) — and the proxy mean lands
0.78% from gold. A tolerance-assisted approximation, not plan recovery.

Evidence at the pivot (step-8 context; all 8 op blocks rendered, DELTA history
intact): `[raws_id_ranges] ... 0 WX ID 2961 16777533 19907847 2960 / 1
Station ID 2916 16777528 19891205 2916 / 2 Elevation 2949 -229 12120 2091`,
`[noaa_station_id_ranges] ... 0 779 20107 482106 779`,
`[stations_with_elev] ... 20107 NaN / 20108 NaN / ...`, plus the
`raws_publicview` schema block with `Latitude`/`Longitude` columns visible and
row-0 `NWS ID = NaN`. I.e. the winner pivoted on "ID spaces are disjoint +
coordinates exist" — facts both losers also had rendered.

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (C1 loser — FAIL, answer 3317.4)

| step | action | semantics (from code) | matches gold? |
|---|---|---|---|
| 0 | load 3 sources (same UTF-8 error + hint) | — | ✓ (plan 1) |
| 1 | reload vars cp1252 | 37×3 | ✓ (aux) |
| 2 | `noaa_station_elev_avg`: unique ids inner-join RAWS **`WX ID`** → mean Elevation | NaN | ✗ plan 3 — **first divergence** (same wrong key as winner) |
| 3 | `noaa_station_join_debug` (1×16 wide) | n_joined=0, n_joined_elev_nonnull=0; ranges 20107–482106 vs 16777533–19907847 | probe |
| 4 | edit avg op → join on `WX ID` **fillna `NESS ID`** | still NaN | ✗ plan 3 |
| 5 | `noaa_station_join_debug2` | wx_matches=0, **ness_matches=0** (457 numeric NESS) | probe |
| 6 | `noaa_station_id_profile` | 779 distinct, 20107–482106, sample 45421… | probe |
| 7 | `raws_wxid_profile` | 16777533–19907847 | probe |
| 8 | `station_elev_from_noaa`: mean of nonexistent col `station_elevation` | NaN | ✗ (hallucinated column) |
| 9–11 | `noaa_station_to_rawwx_map` ×3 | 3 consecutive AttributeErrors (`.tolist`/`.to_list` on the DataFrame that `apply(axis=1)` returns for an EMPTY join) | churn on a sample probe |
| 12 | 4th version runs | matched_on_wx=0, matched_on_ness=0 | probe |
| 13 | **GIVE-UP**: `noaa_station_elev_direct_avg` = mean `Elevation` over ALL 2965 RAWS rows | 3317.37 | ✗ plan 4 — **terminal divergence** (population error: all stations, not used stations) |
| 14 | round → 3317.4 | — | ✗ |
| 15 | Final Answer: 3317.4 | — | FAIL |

**First divergence: step 2 (`WX ID` join)** — identical in kind to the
winner's step 3. **Terminal divergence: step 13** — fallback = unfiltered mean
over every RAWS station, where the winner's fallback (same knowledge state)
was nearest-station-by-coordinates over the used-station population.

Evidence at the give-up (step-13 context, 10 blocks, DELTA history intact —
the wide debug table rendered with FULL values despite 16 cols):
`[noaa_station_join_debug] ... 0 6658 779 779 20107 482106 float64 int64 2961
2960 2949 779 779 2961 2960 0 0`, `[noaa_station_join_debug2] ... 0 779 20107
482106 2961 457 0 0`, `[raws_wxid_profile] ... 16777533 19907847 ...`, plus
the same `raws_publicview` schema block (Latitude/Longitude visible, row-0
`NWS ID = NaN`). This is a superset of the winner's evidence at its pivot:
the loser knew everything the winner knew — 0 matches on WX/Station/NESS,
disjoint ranges, coordinates available — and chose the wrong fallback anyway.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (C3 loser — FAIL, no answer)

Ran to the step cap (25 agent steps) and produced "No response from agent".
Churn flag fires: 25 ops, 20 sinks (80% ≥ 50%, ops ≥ 8). Steps condensed by
family; every op below except the 3 loads + `noaa_station_ids(_to_int)` +
`noaa_fire_station_ids` ends a dead-end sink.

| step | action | semantics (from code) | matches gold? |
|---|---|---|---|
| 0–1 | load 3 sources; cp1252 fix | same as other arms | ✓ (plan 1) |
| 2 | `station_elevations_used`: ids `.isin` **`WX ID`** | 0×0 | ✗ plan 3 — **first divergence** |
| 3 | edit → `.isin` **`Station ID`** | 0×0 | ✗ plan 3 |
| 4 | `raws_station_ids_profile` head(20) — cols Station ID/WX ID/Name/Elev/State/Lat/Lon (**skips `NWS ID`**) | 8-digit ids visible | probe |
| 5 | `noaa_station_ids_profile` head(30) | 5–6-digit ids (45421…) | probe |
| 6 | `noaa_station_id_stats` | 779 distinct, 20107–482106 | probe |
| 7 | `noaa_station_ids` | 779×1 | ✓ (plan 2) |
| 8 | `noaa_station_id_join_diagnostics` | **matches_on_station_id=0, matches_on_wx_id=0** + all four ranges | probe (decisive fact) |
| 9 | `noaa_station_elevation_avg`: mean of `station_verified_in_psa` **as if it were elevation** | 203996.6 | ✗ absurd unit reinterpretation |
| 10 | `noaa_station_ids_to_int` | 779 | dup of 7 |
| 11 | `noaa_station_elevation_from_raws`: `.isin` Station-ID-OR-WX-ID | 0×0 | ✗ re-probe of 2+3 |
| 12 | `noaa_fire_station_ids`: all 6658 rows ("incident weighting") | — | grain probe |
| 13 | decode id//10 + read `/data/wildfire/input/raws_data/<id>.csv` per station | NaN (files don't exist) | ✗ numerology |
| 14 | `noaa_station_elevation_via_raws_wxid`: **WX ID inner join AGAIN** | 0×0 | ✗ 3rd WX probe |
| 15 | `noaa_station_elevation_avg_from_raws_stationid`: **Station ID join AGAIN** | NaN | ✗ 3rd Station-ID probe |
| 16 | `noaa_station_elevation_from_incidents`: **IDENTICAL semantics to step 9** (difflib 0.925 > 0.92 gate) | 203996.6 again | ✗ identical-probe repetition |
| 17 | NESS `//10` decode join | NaN | ✗ numerology |
| 18 | `noaa_station_ids_decoded` (id//10, id%10) | table | probe |
| 19 | `raws_ness_id_to_elev` | 457 numeric NESS ids | probe |
| 20 | hex-parse `NESS ID` join | NaN | ✗ NESS probe #2 |
| 21 | state-grouped mean of ids-as-elevation (variant of 9/16, difflib 0.748) | 245593.9 | ✗ absurd #3 |
| 22 | NESS direct + //10 concat join | NaN | ✗ NESS probe #3 |
| 23 | `station5 = id//10` vs Station ID join | NaN | ✗ Station-ID probe #4 |
| 24 | `noaa_station_elevation_by_latlon`: **EXACT** 4-dp-rounded lat/lon equality join, fire coords × station coords | never executed to a rendered result — step cap | ✗ (first geo idea, 16 steps after the winner's, and exact-match not nearest) |
| — | no final answer | "No response from agent" | FAIL (budget exhaustion) |

**First divergence: step 2 (`WX ID` join)** — same as both Delta arms. Never
probed `NWS ID` in 25 steps; its own id-profile op (step 4) explicitly
selected around it, consistent with the row-0 `NWS ID = NaN` trap.

**Repeated-probe audit (the caller's question — were earlier outputs rendered
when the repeat was created?): YES, all of them, at full result fidelity.**
The LATEST render never dropped a block: rendered-op count grows monotonically
3 → 24 blocks (est. context 1.5k → 5.9k tok) and every block keeps its values
row. Verified at the repeat-creation steps:

- At step 14 (3rd WX-ID join), the context contained
  `[noaa_station_id_join_diagnostics] ... 0 779 2916 2960 **0 0** 20107 482106
  16777528 19891205 16777533 19907847` (matches_on_station_id=0,
  matches_on_wx_id=0) AND `[station_elevations_used] ... Output Table: 0 rows`
  AND `[noaa_station_elevation_from_raws] ... Output Table: 0 rows`.
- At step 16 (identical re-probe of step 9), the context contained
  `[noaa_station_elevation_avg] ... avg_elevation_ft / 0 203996.6386302193` —
  the absurd result it then recomputed verbatim.
- At step 22 (NESS probe #3), the context contained the step-17 and step-20
  NESS NaN blocks (`avg_elevation_ft / 0 NaN`).

So on this task the Latest churn is **thrash-despite-evidence, not
render-starvation**: each dead end was re-attempted with its refutation on
screen. The mode-linked behavioral difference that DID materialize: the Delta
anchor edited its join op in place (`stations_with_elev` v1→v2; 9 ops total),
while Latest minted a NEW operator per hypothesis (25 ops, 20 sinks) and spent
its whole step budget inside the id-numerology basin.

## Pair verdicts

**C1 Delta3k > Delta5k — CHRONIC/VARIANCE (rejected method-choice).** Both
arms made the same first divergence (join on `WX ID`; gold key `NWS ID` never
probed by either), and both reached the same rendered knowledge state: 0
matches on WX/Station/NESS, NOAA ids 20107–482106 vs RAWS 8-digit ids,
Latitude/Longitude columns visible, `NWS ID` sample value NaN. From identical
evidence they chose different fallbacks — winner: nearest-RAWS-by-haversine
over the 779 used stations (4793.1, inside tolerance at −0.78%); loser:
unfiltered mean over all 2965 stations (3317.4). No rendered difference
explains the split (the 5k loser's context was a superset), the winner's pass
is itself an off-plan approximation, and the task is chronic-flagged. Not
attributable to the 3k-vs-5k lever.

**C3 Delta3k > Latest3k — CHRONIC/VARIANCE; churn texture confirmed, but the
mechanism is NOT missing rendered history.** The Latest arm churned exactly as
flagged: 25 ops / 20 dead-end sinks / 25 steps / $0.189 / no answer, four
re-probe families (WX ×3, Station ID ×4, NESS ×3, ids-as-elevation ×3
including a difflib-0.925 identical resubmission), and its first geo idea only
at step 24 — 16 steps after the winner's pivot, and exact-coordinate-equality
rather than nearest. But every repeated probe was created WITH its earlier
refutation fully rendered (verified block-by-block above), so the flip cannot
be attributed to an information gap the Delta render closed; and the winner's
decisive action (geo-nearest fallback) was a method choice on facts both arms
had. Failure mode: step-cap exhaustion driven by op-minting +
evidence-disregard — a LATEST-mode behavioral signature on a chronic task, and
the cost gap ($0.093 vs $0.189) is real churn cost, but the accuracy flip is
variance.
