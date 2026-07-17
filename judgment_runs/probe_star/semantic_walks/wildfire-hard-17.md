# wildfire-hard-17 — semantic walk (PROBE-STAR vintage)

## Task + gold

Q: "What is the average elevation (in feet) of the weather stations used for fire site
monitoring in the NOAA dataset?" — numeric_approximate, gold answer **4830.9**.
Data: `noaa_wildfires.csv` (6658 rows, 37 cols; the gold's `noaa_wildfires_sylvia.csv`;
link column `station_verified_in_psa` is **col 33 of 37**),
`PublicView_RAWS_-3515561676727363726.csv` (2965 rows, 34 cols; BOM; candidate keys
`WX ID`/`NESS ID`/`NWS ID`/`Station ID`; **`NWS ID` is null in row 0 and 822/2965 null**
— the row-0-NaN join-key trap), `noaa_wildfires_variabledescrip.csv` (37-row data
dictionary, cp1252; row 33: `station_verified_in_psa,,Remote Automatic Weather Station
(RAWS) ID number for weather data`).

Arms (all PROBE-STAR raw-probe prompt vintage `acf87127f`, DELTA, code mode):

| arm | knob values | answer | pass | steps | cost |
|---|---|---|---|---|---|
| Delta5kSchemaOnlyProbePrompt (WINNER C1p) | char_limit=5000, stats=off, D1 | **4830.9** | ✓ | 9 | $0.094 |
| DeltaStats1kD2ProbePrompt (WINNER C2p) | char_limit=1000, stats=on, D2 | **4830.9** | ✓ | 15 | $0.105 |
| Delta1kSchemaOnlyProbePrompt (loser both) | char_limit=1000, stats=off, D1 | 3317.4 | ✗ | 4 | $0.026 |

Chronic tag: **wildfire-hard-17 IS in the old `chronic_flippers.json`** — default
verdict CHRONIC/VARIANCE unless accept rules met. Prior vintages: every arm picked
`WX ID` and failed; the probe-pilot Latest3k found `NWS ID` via null-audit → exact gold.

The vintage's key-check clause (code-mode.md line 86, commit `acf87127f`):
> "**Verify the parse before building on it**: … When a table has several candidate key
> columns and you plan to join, compare each candidate's null share and its overlap with
> the other table's values, and join on the key with real overlap — a column can look
> empty in a small sample yet be the correct key, or look populated yet never match."

Written for exactly this trap. Note the trigger condition: "**you plan to join**".

## Gold semantic plan

From `solutions/wildfire/wildfire-hard-17.py`:

1. Load fires CSV (plain read_csv).
2. Load RAWS stations CSV.
3. `used = fires['station_verified_in_psa'].unique()` (779 distinct IDs, null=0).
4. `stations['NWS ID'] → pd.to_numeric(errors='coerce')` — **key = NWS ID** despite
   row-0 NaN and 822/2965 nulls (WX ID looks fully populated but never matches:
   its range 16,777,533–19,907,847 vs used-ID range 20,107–482,106).
5. Filter `stations[NWS ID isin used]` (758 stations match).
6. `mean(Elevation.dropna())` = **4830.9 ft**. (Unfiltered all-stations mean = 3317.4 —
   the loser's exact answer.)

## What the loser (Delta1kSchemaOnly) does — 4 steps

- S1: three **head-only** raw probes: fires head-5, RAWS head-5, var_desc head-20.
  --> Violates the probe clause's "from the beginning AND from deeper in the file" beat;
  var_desc row 33 (the line that names the link column) is never read.
- S2: loads both files **and bundles the terminal compute into the same batch**:
  `station_elev_mean` = `pd.to_numeric(rawws['Elevation']).mean(skipna=True)` over ALL
  2965 stations. --> The divergence: no join, no key, misses gold items 3–5 entirely.
  The key-check clause never fires because its trigger ("you plan to join") is never met.
- S3: full-load renders arrive — **including the complete 37-col fires header with
  `station_verified_in_psa`** (block = 2104 chars; the 1k cap trims sample rows to 2,
  not the header/schema) — alongside its own `avg_elevation_ft 3317.3713…`. It
  finalizes **3317.4** anyway, anchoring on the already-computed number.

Evidence rendered at its S2 commit (quoted from `steps[2].inputMessages`):
```
0  start_year,region_ind,incident_number,avrh_mean,wind_med,erc_med,rain_sum,region,state,incident_name...
0  ﻿OBJECTID,Station Name,WX ID,Ob Time,NESS ID,NWS ID,Elevation,Site Description,Latitude,Longitude,St...
1  12,YNP PORTABLE,18053679,4/18/2025 3:02:04 PM,32D3E21C,,3633,...
```
100-char cell-cap truncation (knob-independent — identical in the 5k arm's probes) cuts
the fires header at col ~10; `station_verified_in_psa` (col 33) invisible; the RAWS
row-0 NWS-ID null (`32D3E21C,,3633`) is right there, unremarked. No link-column
evidence existed in ANY arm's context at this step depth.

## What the C1p winner (Delta5kSchemaOnly) does — 9 steps

- S1: protocol-compliant **head+mid+tail** probes of all three files. At 5k the 15-row
  var_desc probe renders ALL rows, including:
  `10  33  station_verified_in_psa,,Remote Automatic Weather Station (RAWS) ID number for weather data`
  — the data dictionary names the link column before any loader exists.
- S2: real loaders + deletes all three probes in the same batch (clause 4 hygiene).
- S3: joins `used_ids` → stations **on `WX ID`** (the trap key): 779 rows, `45421 NaN`.
- S4: verification stats op → `matched_elevation_nonnull = 0`. The trap is caught.
- S5: retry on `Station ID` → still 0/779.
- S6: **the key-check clause fires in full** — `station_id_mapping_probe` computes
  per-candidate overlap for Station ID / WX ID / NWS ID / NESS ID plus two profile ops
  (nonnull/distinct/min/max both sides). Render: `NWS ID  2120  779  758` (top row).
- S7: rejoin on `NWS ID` → 780 rows; stats op renders
  `used 780, matched 759, mean 4830.852437…` → **4830.9**. Matches gold items 3–6.

## What the C2p winner (DeltaStats1kD2) does — 15 steps

- S1–S4: probes fires+vars (head+mid); vars loader hits `UnicodeDecodeError`, recovers
  via the cp1252 loader hint → full 37-row dictionary; probes+loads RAWS.
- The RAWS load render — at the **same 1k cap as the loser** — carries the full header,
  schema line, and a 2540-char Column-stats block (the cap trims sample rows only):
  ```
  - "WX ID" (numeric): null=4, mean=1.761e+07, min=16777533, max=19907847
  - "NWS ID" (str): null=822, distinct=2136, duplicate_values=7
  - "station_verified_in_psa" (numeric): null=0, mean=2.04e+05, min=20107, max=482106
  ```
  The null-share AND the WX-ID range mismatch are in context **before the first join**.
- S5: joins **on `WX ID` anyway** → `avg_elevation_ft NaN`. Stats did not prevent the trap.
- S6: audit op `station_match_debug` → `matched_rows 0`, both ranges quoted
  (`noaa 20107–482106` vs `wxid 16777533–19907847`).
- S7–S9: `Station ID` retry, then a dead-end hunt for elevation inside the fires table.
- S10–S11: `station_match_debug2` — the clause's audit over **all seven `*ID*` columns**
  (nonnull/min/max/matched_rows/matched_distinct). Render top row:
  `NWS ID  2127  1196  661001  759  758`.
- S12: join on `NWS ID` → `4830.9, matched 759, elev_nonnull 759`. Exact gold.

## Per-arm divergence table

| arm | first semantic divergence | gold item missed | recovered? |
|---|---|---|---|
| 1kSchema (loser) | S2: terminal all-stations mean bundled with loads — no join planned | 3+5 (link + key) | never (finalized past the full header render at S3) |
| 5kSchema (winner) | S3: join on WX ID | 4 (key) | S4 verify (0 matched) → S6 overlap audit → NWS ID |
| Stats1kD2 (winner) | S5: join on WX ID (with null=822 + range mismatch already rendered) | 4 (key) | S6 audit (0 matched) → S11 7-candidate audit → NWS ID |

## Why the winners passed and the loser failed

**Does the key-check clause fire in the winners? Yes — but as recovery, not prophylaxis.**
Neither winner chose `NWS ID` up front; both first joined on the trap key `WX ID` —
Stats1kD2 did so with `null=822` and the range mismatch literally rendered in its
context. What the clause changed vs prior vintages (where every arm picked WX ID and
FAILED) is the machinery after the wrong pick: verify-the-parse ops surfaced
`matched = 0`, and both winners then wrote the clause's literal fingerprint — a
multi-candidate null-share/overlap audit — whose compact output (4x4 / 7x6 tables)
named `NWS ID` with 758–759/779 real overlap. The probe protocol converts the trap
from fatal to recoverable, conditional on reaching a join.

**Does the loser fail because a starved 1k render can't carry the audit output? No —
falsified by the C2p winner**, which runs at the same 1k cap and rendered its audit
tables, the full 34-col header, and a 2540-char stats block intact (the
`max_operator_result_char_limit` knob trims sample rows only: 2 vs 4 rows, measured).
The loser failed upstream of any audit: head-only probes (clause-violating) never read
var_desc row 33; it then bundled a terminal no-join mean into its load batch — so the
key-check clause's trigger ("you plan to join") never fired — and at S3 it finalized
even though the full fires header with `station_verified_in_psa` was by then rendered.

**Knob attribution fails at the hinge.** The hinge is plan shape at the loser's S2
(join vs no-join), taken when its rendered evidence (100-char-capped raw probe lines)
was equivalent in kind across arms — the cell-width cap is knob-independent (verified
identical in the 5k arm), and a 5k render of the loser's own head-only probes would
still contain no link-column evidence (var_desc rows 0–19 only). On the other side,
the stats lever demonstrably failed to prevent the same wrong-key choice in its own
winner. The two rays do not independently fix the failure — both winners are rescued
by the same prompt-vintage audit machinery, equally available to the loser. This is
not dual-lever convergence; it is one shared vintage mechanism plus a coin-flip on
whether the first compute op is a join or a global mean.

## Verdict

**CHRONIC/VARIANCE** (chronic-listed; confirmed). The flip is not attributable to
either knob: the loser's failure is a plan-shape coin-flip (terminal no-join mean
committed before any arm-differential evidence, then anchored past corrective
renders), and both winners' saves come from the shared raw-probe vintage's
key-check/verify machinery, which runs comfortably inside a 1k render. Vintage-level
finding worth keeping: the probe protocol did not fix the initial WX-ID pick (0/3
arms that reached a join chose NWS ID first) — it added a recovery ramp
(verify → matched=0 → overlap audit → NWS ID) that both joining arms rode to exact gold.
