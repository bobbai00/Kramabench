# wildfire-hard-17 — deep dive

Counter-intuitive case: the **3k-solo win** — anchor **Delta3kSchemaOnly** (PASS
4793.1, $0.093) beats both the bigger-budget **Delta5kSchemaOnly** (FAIL 3317.4,
$0.149) and **Latest3kSchemaOnly** (FAIL, no answer, step cap, $0.189; the
levers-report churn poster child: 25 ops / 20 sinks). The twist the walk
exposes: the winner **also missed the gold join** — all three arms fell into the
same row-0-NaN join-key trap, and the winner passed via an off-plan
nearest-station-by-haversine fallback that landed −0.78% from gold, inside
grader tolerance. Both pairs are on `chronic_flippers.json`; both verdicts below
are CHRONIC. Arms are `DataflowSystemGPT52<name>`; traces via
`python3 scripts/extract_walk.py --sut <ARM> --task wildfire-hard-17`.

| arm | result | answer | agent steps | ops | sinks | cost_usd |
|---|---|---|---|---|---|---|
| Delta3kSchemaOnly (mode Y) | PASS | 4793.1 (−0.78%) | 12 | 9 | 5 | $0.093 |
| Delta5kSchemaOnly (mode X1) | FAIL | 3317.4 (−31.3%) | 16 | 12 | 9 | $0.149 |
| Latest3kSchemaOnly (mode X2) | FAIL | none (step cap) | 25 | 25 | 20 (80%) | $0.189 |

## Task

Q: "What is the average elevation (in feet) of the weather stations used for
fire site monitoring in the NOAA dataset?"

D: three files under `data/wildfire/input/`.

- **noaa_wildfires.csv** — 6,658 rows × 37 cols (the gold script reads it as
  `noaa_wildfires_sylvia.csv`; same table). One row per large-wildfire incident
  (ICS-209 derived, Young et al. 2020). Real rows (state,
  `station_verified_in_psa`, latitude, longitude):
  `CA, 45421, 34, -118` · `CA, 42603, 39, -121` · `CA, 45440, 35, -118`
  Relevant columns / quirks:
  - `station_verified_in_psa` — the monitoring-station id per incident; 779
    unique values, all numeric, range **20107–482106** (5–6 digits). The fire
    side of the gold join.
  - `latitude` / `longitude` — fire coordinates, **integer-degree coarse**
    (`34, -118`) despite the data dictionary saying "decimal degrees". This is
    what the winner's geo fallback has to work with (~100 km quantization).
- **noaa_wildfires_variabledescrip.csv** — 37 × 3 data dictionary; **not UTF-8**
  (cp1252) — every arm's first executed step errored on it and reloaded with
  the loader-hint encoding. Row 32 describes `station_verified_in_psa` as
  "Remote Automatic Weather Station (RAWS) ID number for weather data" — it
  names the RAWS *file* but not *which* RAWS id column; no arm ever rendered
  that row (the 37×3 render shows only head rows 0–2).
- **PublicView_RAWS_-3515561676727363726.csv** — 2,965 rows × 34 cols of RAWS
  station metadata. Real rows (Station Name, Station ID, WX ID, NWS ID,
  NESS ID, MesoWest, Elevation, Lat, Lon):
  ```
  YNP PORTABLE    18053677  18053679  NaN     32D3E21C  YNWC1  3633  37.73003 -119.61131
  BOGUE CHITTO    18770452  18770458  227701  32B108FA  BCCM6   126  30.66194  -89.79556
  DEVILS TOWER    17137190  17137193  480606  FA64F588  DVLW4  3872  44.58179 -104.71950
  MENDOCINO PASS  17209642  17209652  041018  32715230  MASC1  5382  39.80742 -122.94511
  ```
  FIVE id-like columns — the task is a join-key discovery problem:
  - `WX ID` — 8-digit internal id, 2,960 distinct, **16777533–19907847**:
    disjoint from the NOAA id space.
  - `Station ID` — 8-digit sibling, 2,916 distinct, **16777528–19891205**:
    also disjoint.
  - `NESS ID` — mostly 8-char hex strings (`32D3E21C`); only 457/2,965 parse
    numeric. Never joins.
  - `MesoWest Station ID` — alphanumeric (`YNWC1`). Never joins.
  - **`NWS ID` — the gold key.** Stored as strings with leading zeros
    (`041018`); 822/2,965 literally empty (27.7%); only 2,127 parse numeric
    (28.3% NaN after coercion); numeric range **1196–661001** — the ONLY id
    column overlapping the NOAA id space (759 of the 779 used ids match).
  - `Elevation` — feet, 2,949 non-null, −229–12120.
  - `Latitude`/`Longitude` — 5-dp precise station coordinates.
  - **The trap: row 0 (YNP PORTABLE) has `NWS ID = NaN`** — and row 0 is the
    only sample row the schema render ever shows.

## Solution

From `solutions/wildfire/wildfire-hard-17.py`, as an operator graph:

```
load noaa_wildfires_sylvia.csv (plain read_csv)
      │
  used = fires['station_verified_in_psa'].unique()          # 779 ids, 20107–482106
      │                                                        (plan 2)
load PublicView_RAWS_….csv (plain read_csv)
      │
  raws['NWS ID'] = pd.to_numeric(raws['NWS ID'], errors='coerce')   # 2127/2965 numeric
      │                                                        (plan 3 — THE key)
  filter raws[NWS ID ∈ used]                                 # 759 of 779 match
      │                                                        (plan 4 — population)
  answer = matched['Elevation'].dropna().mean()  → 1 dp  →  4830.9 ft   (plan 5)
```

Node annotations: no special load params on either file; fire-side key =
`station_verified_in_psa` uniques (station grain, NOT incident-weighted);
station-side key = `NWS ID` coerced numeric (high-null — the coercion is
load-bearing); filter = membership, no dedup needed; final compute = plain mean
over matched stations' elevations, 1 dp.

## What Delta5kSchemaOnly does (mode X1 — FAIL, answer 3317.4)

- **step 0** — load `noaa_wildfires`, `noaa_vars`, `raws_publicview`; vars fails
  UTF-8 → loader hint rendered.
- **step 1** — reload vars `encoding='cp1252'` → 37×3.
- --> **step 2** `noaa_station_elev_avg` — unique `station_verified_in_psa` ids
  renamed to `WX ID`, inner-join RAWS on **`WX ID`**, mean Elevation → NaN.
  **First divergence**: wrong key (gold plan 3 wants `NWS ID`).
- **step 3** `noaa_station_join_debug` — 1×16 wide diagnostic → `n_joined=0`,
  `n_joined_elev_nonnull=0`; ranges 20107–482106 vs 8-digit.
- --> **step 4** — edit `noaa_station_elev_avg`: join on `WX ID`
  **fillna `NESS ID`** → still NaN (wrong key #2).
- **step 5** `noaa_station_join_debug2` → `wx_matches=0`, `ness_matches=0`
  (457 numeric NESS ids).
- **step 6** `noaa_station_id_profile` → 779 distinct, 20107–482106, sample
  `45421, 42603, …`.
- **step 7** `raws_wxid_profile` → 16777533–19907847.
- --> **step 8** `station_elev_from_noaa` — mean of **nonexistent column**
  `station_elevation` → NaN (hallucinated column).
- --> **steps 9–11** `noaa_station_to_rawwx_map` ×3 — three consecutive
  AttributeErrors (`.tolist`/`.to_list` on the DataFrame that `apply(axis=1)`
  returns for an EMPTY join) — churn on a sample-formatting bug.
- **step 12** — 4th version runs → `matched_on_wx=0`, `matched_on_ness=0`.
- --> **step 13** **GIVE-UP**: `noaa_station_elev_direct_avg` = mean `Elevation`
  over **ALL 2,965 RAWS rows** → 3317.3713. **Terminal divergence**: population
  error vs gold plan 4 — every station, not the 779 used ones.
- **step 14** — round 1 dp → 3317.4.
- **step 15** — Final Answer: **3317.4** → FAIL (gold 4830.9, −31.3%).

## What Latest3kSchemaOnly does (mode X2 — FAIL, no answer)

Runs to the 25-step cap, "No response from agent". Churn flag fires: 25 ops,
20 dead-end sinks (80% ≥ 50%, ops ≥ 8). Re-probe families: WX ×3, Station ID
×4, NESS ×3, ids-as-elevation ×3.

- **steps 0–1** — load 3 sources; cp1252 fix. Same as the Delta arms.
- --> **step 2** `station_elevations_used` — used ids `.isin` **`WX ID`** → 0×0.
  **First divergence** (wrong key).
- --> **step 3** — edit → `.isin` **`Station ID`** → 0×0 (wrong key #2; the
  arm's ONLY in-place edit — everything after mints a new operator).
- **step 4** `raws_station_ids_profile` — head(20) of hand-picked columns
  `[Station ID, WX ID, Station Name, Elevation, State, Latitude, Longitude]` —
  **selects around `NWS ID`**, consistent with its NaN sample.
- **step 5** `noaa_station_ids_profile` — head(30) → 5–6-digit ids.
- **step 6** `noaa_station_id_stats` → 779 distinct, 20107–482106.
- **step 7** `noaa_station_ids` → 779×1.
- **step 8** `noaa_station_id_join_diagnostics` → `matches_on_station_id=0`,
  `matches_on_wx_id=0`, plus all four id ranges — the decisive refutation,
  rendered in every later context.
- --> **step 9** `noaa_station_elevation_avg` — mean of
  `station_verified_in_psa` **as if the id were elevation** → 203996.6 (absurd
  unit reinterpretation).
- **step 10** `noaa_station_ids_to_int` — duplicate of step 7.
- --> **step 11** `noaa_station_elevation_from_raws` — `.isin`
  Station-ID-OR-WX-ID → 0×0 (re-probe of steps 2+3).
- **step 12** `noaa_fire_station_ids` — all 6,658 rows ("incident weighting"
  grain probe).
- --> **step 13** — decode `id//10` + read
  `/data/wildfire/input/raws_data/<id>.csv` per station → NaN (files don't
  exist; id-numerology).
- --> **step 14** `noaa_station_elevation_via_raws_wxid` — **WX ID inner join
  AGAIN** (3rd) → 0×0.
- --> **step 15** `noaa_station_elevation_avg_from_raws_stationid` — **Station
  ID join AGAIN** (3rd) → NaN.
- --> **step 16** `noaa_station_elevation_from_incidents` — **semantically
  IDENTICAL to step 9** (difflib 0.925 > 0.92 gate: identical-probe
  resubmission) → 203996.6 again.
- --> **step 17** — NESS `//10` decode join → NaN.
- **step 18** `noaa_station_ids_decoded` — (`id//10`, `id%10`) probe.
- **step 19** `raws_ness_id_to_elev` → 457 numeric NESS ids.
- --> **step 20** — hex-parse `NESS ID` join → NaN (NESS #2).
- --> **step 21** — state-grouped mean of ids-as-elevation (variant of 9/16,
  difflib 0.748) → 245593.9 (absurd #3).
- --> **step 22** — NESS direct + `//10` concat join → NaN (NESS #3).
- --> **step 23** — `station5 = id//10` vs `Station ID` join → NaN (Station-ID
  #4).
- --> **step 24** `noaa_station_elevation_by_latlon` — first geo idea, but
  **EXACT 4-dp-rounded lat/lon equality** join of integer-degree fire coords
  against 5-dp station coords (can only match coincidentally), 16 steps after
  the winner's geo pivot — and its result never renders: **step cap**.
- — no final answer → FAIL (budget exhaustion).

Never probes `NWS ID` in 25 steps.

## What Delta3kSchemaOnly does (mode Y — PASS, answer 4793.1)

- **step 0** — load `noaa_wildfires_vars`, `noaa_wildfires`, `raws_publicview`;
  vars fails UTF-8 → loader hint.
- **step 1** — reload vars cp1252 → 37×3.
- **step 2** `noaa_station_ids` — to_numeric, dropna, unique, sort → 779 ids
  (gold plan 2, exactly).
- --> **step 3** `stations_with_elev` — left-join ids on RAWS **`WX ID`**
  (renamed) → 779 rows, Elevation ALL NaN. **First divergence** — the same
  wrong key as both losers.
- --> **step 4** — edit `stations_with_elev` **in place** → join on
  **`Station ID`** → also all-NaN (seen at step 6). Wrong key #2; the
  near-miss is never recovered *on-plan* — `NWS ID` is never joined.
- **step 5** `raws_id_ranges` — the probe **SELECTS**
  `['WX ID','Station ID','MesoWest Station ID','NWS ID','NESS ID',…]` but
  computes ranges only for WX ID / Station ID / Elevation → WX 16777533–19907847
  (2,960 distinct), Station 16777528–19891205 (2,916), Elevation −229–12120.
  (Its fingers touched the gold key and still didn't profile it.)
- **step 6** `avg_station_elevation_ft` from `stations_with_elev` → NaN
  (dead end confirmed).
- **step 7** `noaa_station_id_ranges` → 779 ids, 20107–482106.
- --> **step 8** **PIVOT (off-plan)**: `noaa_station_elev_from_noaa` — per
  unique station id take the fire's `latitude`/`longitude`, find the nearest
  RAWS station by haversine → NameError (`raws_publicview` missing from the
  signature). Near-miss, recovered next step.
- **step 9** — fix signature; `BallTree(metric='haversine')` k=1 → 779×5;
  rendered row 0: `nearest_raws_km 1.98, Elevation_ft 950`.
- **step 10** — edit `avg_station_elevation_ft` in place → mean of
  `Elevation_ft` → **4793.1**.
- **step 11** — Final Answer: **4793.1** → PASS at −0.78% vs gold 4830.9.

Be explicit about what this pass is: the winner **also missed the gold NWS-ID
join** (never probed, never joined it). Its fallback approximates gold plan 3/4
by geography — nearest RAWS station to each fire's integer-degree coordinates
as a proxy for that fire's verified station — over the correct *population*
(the 779 used stations), and the proxy mean lands inside grader tolerance. A
tolerance-assisted approximation, not plan recovery. Mechanically it stays
tight: 9 ops, 2 in-place re-edits, 5 sinks.

## Why Y succeeded but X failed

**First, the shared trap — a render-sampling blind spot worth naming.** The
schema render of `raws_publicview` shows exactly ONE sample row — row 0 — and
row 0's `NWS ID` is NaN. Verified in all three arms' contexts (identical line,
present from step 0 onward in every arm):

> `0  12  YNP PORTABLE  18053679  4/18/2025 3:02:04 PM  32D3E21C  NaN  3633
> NaN  37.73003  -119.61131  CA  …  18053677  YNWC1  …`

(columns: WX ID = 18053679, NESS ID = 32D3E21C, **NWS ID = NaN**,
Elevation = 3633, …, Station ID = 18053677). The gold key's only visible sample
value looked empty in every arm, and **no arm ever probed or joined `NWS ID`**
— the winner's step-5 probe selected the column then skipped it; Latest's
step-4 profile hand-picked columns around it. One rendered sample row presented
a 72%-populated join key as an empty column to all three arms. The first
divergence (join on `WX ID`) is therefore **shared**, and the flip is decided
entirely downstream of it — by fallback quality and by mint-vs-edit behavior.

**C1 — Y vs Delta5kSchemaOnly: identical (indeed richer) evidence, different
fallback → method-choice, not the 3k-vs-5k lever.** At the winner's pivot
(step 8) its context held, with full DELTA history intact:

> `[raws_id_ranges] … 0 WX ID 2961 16777533 19907847 2960 / 1 Station ID 2916
> 16777528 19891205 2916 / …`
> `[noaa_station_id_ranges] … 0 779 20107 482106 779`
> `[stations_with_elev] … 0 20107 NaN …`

plus the `raws_publicview` schema block with `Latitude`/`Longitude` visible and
the row-0 `NWS ID = NaN`. At the loser's give-up (step 13) its context held the
same load-bearing facts and more (~9.7k est tokens vs the winner's ~4.3k):

> `[noaa_station_join_debug] … 0 6658 779 779 20107 482106 float64 int64 2961
> 2960 2949 779 779 2961 2960 0 0`  (tail: n_joined=0, n_joined_elev_nonnull=0)
> `[noaa_station_join_debug2] … 0 779 20107 482106 2961 457 0 0`  (tail:
> wx_matches=0, ness_matches=0)
> `[raws_wxid_profile] … 0 2965 2961 2960 16777533 19907847 …`

plus the same schema block (coords visible, row-0 NWS NaN). Composition differs
only off the critical path (the winner additionally had the Station-ID range +
its 0-join; the loser instead had the NESS 0-join + NOAA id percentiles/samples)
— on every fact the fallback decision needed (**ID spaces disjoint, zero matches
on every key tried, Elevation available, coordinates available**) the two
knowledge states are equivalent. From that same state the winner chose
nearest-RAWS-by-haversine over the 779 used stations (4793.1, −0.78%, in
tolerance) and the loser chose the unfiltered mean over all 2,965 stations
(3317.4, −31.3%). No rendered difference explains the split; the winner's pass
is itself off-plan. **Verdict: CHRONIC/VARIANCE (rejected method-choice)** —
the task is chronic-flagged, and the 5k render budget cannot be credited or
blamed.

**C3 — Y vs Latest3kSchemaOnly: churn texture confirmed, but the mechanism is
NOT missing rendered history → CHRONIC/VARIANCE; the cost gap is real.** The
Latest arm died by op-minting: after its single step-3 edit it minted a NEW
operator per hypothesis (25 ops / 20 sinks) where the Delta anchor edited in
place (`stations_with_elev` v1→v2, `avg_station_elevation_ft` v1→v2; 9 ops).
The decisive audit — was each re-probe created with its refutation on screen?
**Yes, all of them, at full result fidelity.** The LATEST render never dropped
a block (op blocks grow monotonically 3 → 24, est. context 1.8k → 5.9k tok):

- at step 14 (WX join #3) the context contained
  `[noaa_station_id_join_diagnostics] … 0 779 2916 2960 0 0 20107 482106
  16777528 19891205 16777533 19907847` (matches_on_station_id=0,
  matches_on_wx_id=0) AND both zero-row join blocks
  (`[station_elevations_used] … Output Table: 0 rows`,
  `[noaa_station_elevation_from_raws] … Output Table: 0 rows`);
- at step 16 (difflib-0.925 identical resubmission of step 9) the context
  contained `[noaa_station_elevation_avg] … avg_elevation_ft / 0
  203996.6386302193` — the absurd result it recomputed verbatim;
- at step 22 (NESS #3) the context contained the step-17 and step-20 NESS
  refutations (`[noaa_station_elevation_avg_correct] … 0 NaN`,
  `[noaa_station_elevation_mean] … 0 NaN`).

So this is **thrash-despite-evidence, not render-starvation**: the flip cannot
be attributed to an information gap the DELTA render closed, and the winner's
decisive action (geo-nearest fallback, step 8) was a method choice on facts
Latest also had rendered — Latest's own first geo idea arrived at step 24
(16 steps later) as an exact-coordinate-equality join, then hit the step cap.
Failure mode: budget exhaustion driven by op-minting + evidence-disregard — a
LATEST-mode behavioral signature on a chronic task. The **cost** gap ($0.093 vs
$0.189) is real churn cost; the **accuracy** flip is variance.

**Per-arm divergence table**

| arm | first divergence | terminal divergence / death | gold-plan item fallen short | answer |
|---|---|---|---|---|
| Delta3kSchemaOnly (Y) | step 3 `stations_with_elev`: join on `WX ID` | steps 8–10: off-plan geo-nearest fallback (NWS ID never probed) | plan 3 (key) missed; plans 3/4 approximated by geography | 4793.1 PASS (−0.78%, tolerance-assisted) |
| Delta5kSchemaOnly (X1) | step 2 `noaa_station_elev_avg`: join on `WX ID` | step 13 give-up: mean over ALL 2,965 stations | plan 3 (key) missed; plan 4 (population) violated | 3317.4 FAIL (−31.3%) |
| Latest3kSchemaOnly (X2) | step 2 `station_elevations_used`: `.isin` `WX ID` | step 24 step-cap mid-probe, no answer | plan 3 (key) missed; plan 5 never reached | none FAIL |

**Labels: C1 CHRONIC (rejected method-choice) · C3 CHRONIC (variance; churn
texture and cost real).** Neither flip is attributable to the 3k-vs-5k budget
or the Delta-vs-Latest render; the one mode-linked behavior that did
materialize (mint-vs-edit) explains cost and step-burn, not the answer split.
