# Semantic walk: environment-hard-11 (PROBE-STAR vintage, raw-probe prompt)

## Task + gold answer

**Question:** What was the average rainfall (to 2 decimal places) in the one-day period before sampling when water samples from Pleasure Bay Beach failed to meet swimming standards? A sample meets the standard if it contains ≤ 104 counts of Enterococcus per 100 mL.

**Gold answer:** `0.37` (numeric_exact). Gold prints `0.37`.

**Pair (C3p, B-only):** winner `Latest5kSchemaOnlyProbePrompt` (0.37, PASS) vs loser `Delta5kSchemaOnlyProbePrompt` (0.40, FAIL). `environment-hard-11` ∈ `chronic_flippers.json`.

**Config diff (validity gate):** one knob — `context_mode: latest` vs `delta` (both `column_stats:false, data_level:1`, both 5k/3k char limits). Passes one-knob gate.

## Data shape (the trap)

`pleasure_bay_and_castle_island_beach_datasheet.csv` has a title row, then a **station-name row**, then a sub-header row:

```
"Pleasure Bay Beach, South Boston: Bacterial Water Quality",,,,,,,,,
,,,,Pleasure Bay @ Broadway,,Pleasure Bay @ Flagpole,,Castle Island Playground,
Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus
```

Loading at the sub-header row yields columns `Date, 1-Day Rain, 2-Day Rain, 3-Day Rain, Tag, Enterococcus, Tag.1, Enterococcus.1, Tag.2, Enterococcus.2` — **three Enterococcus stations**: `Enterococcus`=Broadway, `Enterococcus.1`=Flagpole, `Enterococcus.2`=Castle Island Playground. "Pleasure Bay Beach" = the two Pleasure Bay stations (Broadway + Flagpole); Castle Island Playground is a different beach.

## Gold semantic plan

Source: `solutions/environment/environment-hard-11.py`

| # | Plan item |
|---|---|
| G1 | Load with `skiprows=1, header=[0,1]`; forward-fill station names; melt the three station×measure columns into tidy `Location`/`Enterococcus`/rain rows |
| G2 | **Filter `Location != 'Castle Island Playground'`** — keep only the two Pleasure Bay stations (Broadway + Flagpole) |
| G3 | Filter `Enterococcus > 104` (failed standard) |
| G4 | `mean('1-Day Rain')` → **0.37** |

The load-bearing step is **G2: include BOTH Pleasure Bay stations, exclude Castle Island Playground.**

## Walk: Latest5kSchemaOnly (WINNER — PASS)

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_preview` | preview 25 lines → sees title/station-name/sub-header rows | — |
| 1 | `pleasure_bay` | `read_csv(header=2)` → 860×10, cols incl. `Enterococcus`, `Enterococcus.1`, `Enterococcus.2` | G1 ✓ (flat form) |
| 2 | `pb_failures` | rename `Enterococcus→pb_broadway`, `Enterococcus.1→pb_flagpole`; `fails = (broadway>104) \| (flagpole>104)` → **24 rows** | **G2 ✓ + G3 ✓** — Broadway OR Flagpole, excludes `Enterococcus.2` (Castle Island) |
| 3 | `avg_rainfall_failures` | `mean(rain_1d)`, round 2 → `0.37` | G4 ✓ |

**First divergence:** none. `Broadway | Flagpole` (both Pleasure Bay stations, Castle Island dropped) is exactly gold's `Location != 'Castle Island Playground'`.

## Walk: Delta5kSchemaOnly (LOSER — FAIL)

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_preview` | preview 8 lines → sees title/station-name/sub-header rows (same window) | — |
| 1 | `beach_data` | `read_csv(skiprows=2)` → 860×10, same cols incl. `Enterococcus.1`, `Enterococcus.2` | G1 ✓ (equivalent load) |
| 2 | `pb_failures` | `out = df[["Date","1-Day Rain","Enterococcus"]]`; filter `Enterococcus > 104` → **22 rows** | **DIVERGES at G2:** keeps only the FIRST Enterococcus (Broadway); drops `Enterococcus.1` (Flagpole) entirely |
| 3 | `avg_rain_pb_failures` | `mean(1-Day Rain)`, round 2 → `0.40` | G4 executed on the wrong row set |

**First semantic divergence:** step 2, op `pb_failures`, plan item G2 — keep-first collapses "Pleasure Bay Beach" to Broadway alone, silently discarding every Flagpole exceedance. 22 rows vs the winner's 24 → 0.40 vs 0.37.

## Evidence at the decision — identical render, different read

Both arms saw the SAME differential evidence:
- **Both** raw previews contain the station-name row `,,,,Pleasure Bay @ Broadway,,Pleasure Bay @ Flagpole,,Castle Island Playground,` and the sub-header row (winner read 25 lines, loser 8 — both windows include rows 1–2).
- **Both** loaded tables render 860×10 with `Enterococcus`, `Enterococcus.1`, `Enterococcus.2` all visible in the header.

So the loser had every fact it needed on screen: three Enterococcus columns and their station labels. The winner mapped the two Pleasure Bay stations (Broadway + Flagpole) and excluded Castle Island; the loser under-read to the single first `Enterococcus` column. No latest-vs-delta rendering difference supplied the winner a signal the loser lacked — the schema-only render was informationally identical on this decision.

## Verdict

**C3p (Latest5kSchemaOnly > Delta5kSchemaOnly): REJECTED — method-choice / interpretation coin flip on a chronic task.** The divergence (Broadway|Flagpole vs keep-first Broadway) is a free interpretive pick made against **identical rendered evidence** — both arms saw the station-name row and all three Enterococcus columns. It is orthogonal to `context_mode`; the winner's correct multi-station reading is not attributable to `latest` rendering. Confirms the historical pattern (winners Broadway|Flagpole ≡ gold 0.37; losers keep-first → 0.40). Chronic flipper → defaults to variance.

Artifacts: `system_scratch/{DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt,DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt}/environment-hard-11/`; gold `solutions/environment/environment-hard-11.py` (0.37, repro with `.venv/bin/python`).
