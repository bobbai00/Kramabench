# Deep-dive: environment-hard-11 (PROBE-STAR vintage) — CHRONIC, keep-first interpretation coin

## Task
Q: What was the average rainfall (to 2 decimal places) in the one-day period before sampling when water samples from Pleasure Bay Beach failed to meet swimming standards? A sample meets the standard if it contains ≤ 104 counts of Enterococcus per 100 mL.

D:
- `data/environment/input/pleasure_bay_and_castle_island_beach_datasheet.csv` — a wide, multi-station sheet with a title row, then a station-name row, then a sub-header row, then data. Real top rows:
  ```
  "Pleasure Bay Beach, South Boston: Bacterial Water Quality",,,,,,,,,
  ,,,,Pleasure Bay @ Broadway,,Pleasure Bay @ Flagpole,,Castle Island Playground,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus
  "August 27, 2024",0.00,0,0,,30,,10,,20
  "August 20, 2024",0.01,0.12,0.15,,10,,10,,60
  ```
  Loading at the sub-header row yields `Date, 1-Day Rain, 2-Day Rain, 3-Day Rain, Tag, Enterococcus, Tag.1, Enterococcus.1, Tag.2, Enterococcus.2` — **three Enterococcus stations**, each sitting under a station name carried in the row above (verified via ffill of that row):
  - `Enterococcus`   → Pleasure Bay @ Broadway
  - `Enterococcus.1` → Pleasure Bay @ Flagpole
  - `Enterococcus.2` → Castle Island Playground
  Quirks: station names live in the UNNAMED row above the sub-header (the ffill target — pandas suffixes the flat load as `.1`/`.2`); "Pleasure Bay Beach" = the TWO Pleasure Bay stations (Broadway + Flagpole), while Castle Island Playground is a DIFFERENT beach that must be excluded; `1-Day Rain` is the pre-sampling rainfall to average; `Tag` columns hold the `<` below-detection marker.

## Solution
solutions/environment/environment-hard-11.py — prints `0.37`:

```
load(skiprows=1, header=[0,1]) → ffill station names → flatten "Station_Measure"
   → melt to tidy rows: Location / Enterococcus / 1-Day Rain
   → filter(Location != 'Castle Island Playground')   ← keep BOTH Pleasure Bay stations (Broadway + Flagpole)
   → filter(Enterococcus > 104)                       ← failed standard
   → mean('1-Day Rain') = 0.37
```
- load spec: `skiprows=1, header=[0,1]`; ffill the unnamed station cells so both Pleasure Bay columns retain their station name.
- predicate: `Location != 'Castle Island Playground'` (load-bearing — G2), then `Enterococcus > 104`.
- grain: one row per (date × station); mean over all surviving `1-Day Rain`.
- output: **0.37**.

## What Delta5kSchemaOnly (mode X, loser) does
- step 0 `raw_preview`: preview 8 lines → sees title / station-name / sub-header rows.
- step 1 `beach_data`: `read_csv(skiprows=2)` → 860×10, header incl. `Enterococcus`, `Enterococcus.1`, `Enterococcus.2`. G1-equivalent load.
- --> step 2 `pb_failures`: `out = df[["Date","1-Day Rain","Enterococcus"]]`; filter `Enterococcus > 104` → **22 rows**. <— DIVERGES at G2: keep-first collapses "Pleasure Bay Beach" to Broadway alone, silently dropping every `Enterococcus.1` (Flagpole) exceedance.
- step 3 `avg_rain_pb_failures`: `mean(1-Day Rain)`, round 2 → **`0.40` FAIL** (computed on the wrong row set).

## What Latest5kSchemaOnly (mode Y, winner) does
- step 0 `raw_preview`: preview 25 lines → sees title / station-name / sub-header rows.
- step 1 `pleasure_bay`: `read_csv(header=2)` → 860×10, same three-station header.
- step 2 `pb_failures`: rename `Enterococcus→pb_broadway`, `Enterococcus.1→pb_flagpole`; `fails = (broadway>104) | (flagpole>104)` → **24 rows** — both Pleasure Bay stations, excludes `Enterococcus.2` (Castle Island). Matches gold G2+G3 and yields gold's 0.37.
- step 3 `avg_rainfall_failures`: `mean(rain_1d)`, round 2 → **`0.37` PASS**.
- No near-misses to recover from; first divergence from the solution: none.

## Why Y succeeded but X failed
Rendered evidence is identical. Both arms saw:
- the station-name row `,,,,Pleasure Bay @ Broadway,,Pleasure Bay @ Flagpole,,Castle Island Playground,` in their raw preview (winner window 25 lines, loser 8 — both include sheet rows 1–2), and
- the loaded table at 860×10 with `Enterococcus`, `Enterococcus.1`, `Enterococcus.2` all present in the header.

So the loser had every fact it needed on screen: three Enterococcus columns and their station labels. The winner mapped the two Pleasure Bay stations (Broadway OR Flagpole) and dropped Castle Island; the loser under-read to the single first `Enterococcus` column. No latest-vs-delta rendering difference supplied the winner a signal the loser lacked — the schema-only render was informationally identical on this decision.

Verdict: **C3p (Latest5kSchemaOnly > Delta5kSchemaOnly): REJECTED — keep-first interpretation coin on a chronic task.** The divergence (Broadway|Flagpole ≡ gold vs keep-first Broadway) is a free interpretive pick made against identical evidence, orthogonal to `context_mode`; the winner's correct multi-station reading is not attributable to `latest` rendering. It replicates the prior vintage exactly (winners read Broadway|Flagpole → 0.37; losers keep-first → 0.40). Task ∈ chronic_flippers.json → **CHRONIC**, defaults to variance.

Artifacts: `system_scratch/DataflowSystemGPT52{Latest5kSchemaOnly,Delta5kSchemaOnly}ProbePrompt/environment-hard-11/`; gold `solutions/environment/environment-hard-11.py` (0.37, repro with `.venv/bin/python`).
