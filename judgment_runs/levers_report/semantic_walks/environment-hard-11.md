# Semantic walk — environment-hard-11

## Task + gold answer

Q: "What was the average rainfall (to 2 decimal places) in the one-day period before sampling when water samples from Pleasure Bay Beach failed to meet swimming standards? A sample meets the standard if it contains fewer than or equal to 104 counts of Enterococcus per 100 milliliters of water."
Gold answer: **0.37** (`numeric_exact`). Task is in `chronic_flippers.json` — default verdict CHRONIC/VARIANCE unless accept rules are met.

Arms: Delta5kSchemaOnly **PASS 0.37** · Stats3kD2 **PASS 0.37** · Delta3kSchemaOnly **FAIL 0.40**.

Source file (`pleasure_bay_and_castle_island_beach_datasheet.csv`) structure, from the raw bytes:
```
"Pleasure Bay Beach, South Boston: Bacterial Water Quality",,,,,,,,,     <- title row
,,,,Pleasure Bay @ Broadway,,Pleasure Bay @ Flagpole,,Castle Island Playground,   <- location row
Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus  <- measure row
"August 27, 2024",0.00,0,0,,30,,10,,20
```
"Pleasure Bay Beach" = the union of TWO sampling stations (Broadway + Flagpole); Castle Island Playground is the excluded third. Three columns share the literal name `Enterococcus` — the location identity lives only in the location row.

Numeric ground truth (re-executed on the real data, `.venv/bin/python`):
- Gold melt-grain, per (date, location), Location ≠ Castle Island, Ent > 104: **0.372083 → 0.37** (24 exceedance rows)
- Per-date OR (Broadway>104 | Flagpole>104): **0.372083 → 0.37** (24 dates; Broadway's 22 and Flagpole's 2 exceedances are disjoint, so identical to gold) — exactly both winners
- **Broadway-only** exceedances: **0.404091 → 0.40** (22 dates) — exactly the loser
- Flagpole-only: 0.02 (the 2 dropped low-rain dates are what pull 0.40 down to 0.37)
- Include-Castle OR: 0.36 — nobody's answer

All four variants are numerically distinct at 2dp; each arm's answer is uniquely pinned to its predicate.

## Gold semantic plan

1. Load `data/environment/input/pleasure_bay_and_castle_island_beach_datasheet.csv` with `skiprows=1, header=[0,1]`: skip the title row, use location row + measure row as a two-level header.
2. Flatten the header: forward-fill location names across `Unnamed` gaps in the top level → `Pleasure Bay @ Broadway_Enterococcus`, `Pleasure Bay @ Flagpole_Enterococcus`, `Castle Island Playground_Enterococcus` (+ `_Tag`); id cols = Date, 1/2/3-Day Rain. **Every measurement column gets a location identity.**
3. Melt measurement cols, split `Location`/`Measure`, pivot to tidy rows (Date, rains, Location, Tag, Enterococcus); cast numerics.
4. Location filter: `Location != 'Castle Island Playground'` → keep BOTH Pleasure Bay stations (Broadway AND Flagpole).
5. Exceedance filter: `Enterococcus > 104` (fails the ≤ 104 standard).
6. Mean of `1-Day Rain` over exceedance rows → 0.37 (2 dp).

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (WINNER, C1)

**PASS — Final Answer: 0.37.** 5 agent steps, 1 error step.

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | beach | default `read_csv` → 862×10 all-str; title in col-0 name, location+measure rows as data rows 0–1 | 1 (raw) |
| 1 | beach_tidy | header := row 1 (measure row), slice rows 2:, then label-select `['Date','1-Day Rain','Tag','Enterococcus','Tag','Enterococcus']` → **ERROR** `Length mismatch: Expected axis has 14 elements, new values have 6` (duplicate labels each select all 3 location columns) | 2–3 attempt |
| 2 | beach_tidy | positional fix: `tag_idxs[0], ent_idxs[0], tag_idxs[1], ent_idxs[1]` → keeps **Broadway + Flagpole**, leaves `ent_idxs[2]` (Castle) out; names them `ent_broadway`/`ent_flagpole`; numeric cast → 860×6 | 2, 3, 4 |
| 3 | pb_fail_rain_avg | `fail = (ent_broadway > 104) \| (ent_flagpole > 104)`; `mean(rain_1d)` over fail, round 2 → 0.37 | 5, 6 |
| 4 | TEXT | Final Answer: 0.37 | — |

No semantic divergence: per-date OR is an equivalent formulation of gold's per-location grain (proven identical above). The step-1 stumble on duplicate `Enterococcus` labels is the same trap the loser hit; this arm's recovery kept both PB stations.

Evidence rendered at its decision (step 2–3 context, load block): the location row and measure row, fully in context —
```
0	NaN	NaN	NaN	NaN	Pleasure Bay @ Broadway	NaN	Pleasure Bay @ Flagpole	NaN	Castle Island Playground	NaN
1	Date	1-Day Rain	2-Day Rain	3-Day Rain	Tag	Enterococcus	Tag	Enterococcus	Tag	Enterococcus
```
Render width: **18 data rows** (7 head + tail) vs 10 in the 3k arms. The 8 extra rows are three more 2024 head rows and 2001-era tail rows (Flagpole/Castle all-NaN, plus one Broadway exceedance `178`) — **no structural information beyond the shared head**; the location mapping was already complete in the 10-row render.

## Walk: DataflowSystemGPT52DeltaStats3kD2 (WINNER, C2)

**PASS — Final Answer: 0.37.** 5 agent steps, 0 error steps (only arm with none).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | pleasure_bay | default `read_csv` → 862×10 | 1 (raw) |
| 1 | pleasure_bay_tidy | rename ALL 10 columns positionally in one pass: `col0..col3, tag_broadway, ent_broadway, tag_flagpole, ent_flagpole, tag_castle, ent_castle`; slice off location row; to_datetime; strip `<`/commas; numeric cast → 861×10 | 2, 3 |
| 2 | pb_failures | drop date-NaN row (the measure-header row); `ent_max_pb = max(ent_broadway, ent_flagpole)` (**Castle excluded**); `failed = ent_max_pb > 104` → 860×6 | 4, 5 |
| 3 | avg_rain_before_failed | `mean(rain_1d)` over failed → 0.3720833 | 6 |
| 4 | TEXT | Final Answer: 0.37 | — |

No semantic divergence (max-of-two > 104 ≡ OR ≡ gold). Sidestepped the duplicate-label trap entirely by never using the name `Enterococcus` as a column selector.

Evidence rendered at its decision (step 1 context): same 10-row head/tail as the loser PLUS the stats block:
```
- "Unnamed: 4" (str): null=445, distinct=3, top_5={"<"=415, "Pleasure Bay @ Broadway"=1, "Tag"=1}
- "Unnamed: 6" (str): null=666, distinct=3, top_5={"<"=194, "Pleasure Bay @ Flagpole"=1, "Tag"=1}
- "Unnamed: 8" (str): null=791, distinct=3, top_5={"<"=69, "Castle Island Playground"=1, "Tag"=1}
Output Table profile:
- headers: 9 of 10 columns are unnamed ("Unnamed: 1", ...)
```
The stats lines restate the column↔location mapping (and the `<` value quirk) a second way; they plausibly aided the clean one-pass positional rename (0 errors). But they add no fact absent from the 10-row render the loser had — the location row itself states the same mapping.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (LOSER, both pairs)

**FAIL — Final Answer: 0.40** (gold 0.37; = Broadway-only variant exactly). 6 agent steps, 2 error steps.

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | pb_data | default `read_csv` → 862×10 | 1 (raw) |
| 1 | pb_clean | find header row by `Date` in col 0; single-level header from measure row; **location row dropped, no location mapping**; per-`Enterococcus`-named col strip `<` → **ERROR** `AttributeError: 'DataFrame' object has no attribute 'str'` (3 duplicate `Enterococcus` labels → `df2[c]` is a DataFrame) | falls short of 2 |
| 2 | pb_clean | adds `isinstance(s, pd.DataFrame)` branch, still hits the **same AttributeError** (line 26) | falls short of 2 |
| 3 | pb_clean | **THE HINGE.** Dedups header names → `Enterococcus`, `Enterococcus_1`, `Enterococcus_2` (its own code enumerates all 3 measurement columns); then keeps **only the first**, naming it `PleasureBay_Broadway_Enterococcus`; `keep = ['Date', '1-Day Rain', 'PleasureBay_Broadway_Enterococcus']` → Flagpole (still Pleasure Bay per plan item 4) and Castle both dropped → 860×3 | falls short of 3, 4 |
| 4 | pb_fail_rain_avg | Broadway-only > 104; mean → 0.4 | 5, 6 on wrong coverage |
| 5 | TEXT | Final Answer: 0.40 | — |

First divergence: step 1 (drops the location row, so the three `Enterococcus` columns lose their station identity — falls short of plan item 2). Decisive, answer-carrying divergence: **step 3** — the keep-first-only recovery, which reduces "Pleasure Bay Beach" from 2 stations to 1.

Evidence rendered at the divergence (step 3 context, verified programmatically — the load block was **never evicted**; `Pleasure Bay @ Flagpole` present in context at every step 1–5, ctx sizes 2.4k→13.0k chars):
```
0	NaN	NaN	NaN	NaN	Pleasure Bay @ Broadway	NaN	Pleasure Bay @ Flagpole	NaN	Castle Island Playground	NaN
1	Date	1-Day Rain	2-Day Rain	3-Day Rain	Tag	Enterococcus	Tag	Enterococcus	Tag	Enterococcus
```
Two proofs the loser CONSUMED this evidence at the hinge step: (a) its dedup code creates `Enterococcus_1`/`Enterococcus_2` — it knew there were three stations' measurement columns; (b) it named its kept column `PleasureBay_Broadway_Enterococcus` — the token "Broadway" exists nowhere in its trace except that rendered location line. The error was an **interpretation choice** ("Pleasure Bay Beach = the Broadway station", or keep-first after two failed iterations), not an information gap.

Mechanism scans: no churn (3-op DAG, ops < 8, single sink); `[ERROR` renders 2 (loser) / 1 (Delta5k) / 0 (stats); step-1→2 resubmission is a near-identical retry (thrash-lite) but recovery succeeded by step 3; failure-mode class = wrong-answer (coverage), not timeout/format/gave-up.

## Pair verdicts

**C1 — Delta5kSchemaOnly > Delta3kSchemaOnly: CHRONIC/VARIANCE (attribution rejected).**
The decision-relevant fact (two Pleasure Bay stations; which `Enterococcus` column is which) was fully rendered in the loser's context at its divergence step and demonstrably consumed by its code ("Broadway"). The winner's lever — 8 extra rendered rows — carries zero additional structural information (2001 tail rows with Flagpole/Castle NaN + three more 2024 rows). Winner and loser hit the SAME duplicate-label trap and differ only in the recovery interpretation (keep both PB stations vs keep first) on identical evidence — a coin-flip on a task already in `chronic_flippers.json`.

**C2 — DeltaStats3kD2 > Delta3kSchemaOnly: CHRONIC/VARIANCE (attribution rejected).**
The stats lines (top_5 pinning Unnamed:4/6/8 to Broadway/Flagpole/Castle, null counts, unnamed-headers profile) restate a mapping the loser already had rendered and used. Stats plausibly bought error-avoidance ergonomics (only arm with a clean one-pass rename, 0 errors), but the error loop is not what flipped the answer — the C1 winner passed after the identical stumble without stats. The flip-carrying choice (Broadway-only coverage) is not addressed by any stats line.

**Dual-lever convergence: DISPROVEN for this task.** The pattern requires the same MISSING fact fixed two ways; here the fact was present (and used) in the loser's render at decision time. Two winners over the same loser on a chronic-flipper task, with the loser's divergence being an evidence-independent interpretation choice, is the chronic-variance signature — not lever causation.
