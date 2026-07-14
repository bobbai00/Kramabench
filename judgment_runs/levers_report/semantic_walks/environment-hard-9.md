# environment-hard-9 — semantic trace walk

Pairs judged: **C1 Delta3k > Delta5k** (chronic*), **C3 Delta3k > Latest3k** (chronic*).
Anchor/winner: `DataflowSystemGPT52Delta3kSchemaOnly` (PASS in both cohorts).
`environment-hard-9` **is on the 23-task chronic-flipper list** (noise floor applies).

| arm | f1 | steps | cost_usd | in/out/cached tok | verdict |
|---|---|---|---|---|---|
| Delta3kSchemaOnly (anchor) | **1.0** | 14 | 0.1088 | 146752 / 3999 / 129536 | PASS |
| Delta5kSchemaOnly (C1 loser) | 0.0 | 5 | 0.0379 | 37802 / 1651 / 32640 | FAIL |
| Latest3kSchemaOnly (C3 loser) | 0.36 | 6 | 0.0402 | 40799 / 1905 / 36736 | FAIL |

## Task + gold answer

Query: *"Which Boston Harbor beaches met swimming standards 100% of the time between 2020 and 2024 (inclusive)? A sample meets the standard if it contains fewer than 104 counts of Enterococcus per 100 milliliters of water."*
Gold answer (`list_exact`, set-compared): **`['Pleasure Bay Beach', 'Castle Island Beach', 'City Point Beach']`**.

## Gold semantic plan

From `solutions/environment/environment-hard-9.py`. Per beach CSV in the fixed 8-file list:

1. **Load spec (critical):** `read_csv(fp, skiprows=1, header=[0,1])` — skip row 0 (the "…: Bacterial Water Quality" banner), take the **next two rows as a multi-index header** (row 1 = station North/Middle/South, row 2 = Date / 1-Day Rain / 2-Day Rain / 3-Day Rain / Tag / Enterococcus …).
2. **Flatten headers:** forward-fill the station level across each Tag/Enterococcus pair; join levels → e.g. `North_Enterococcus`, `Middle_Enterococcus`, `South_Enterococcus`, plus flat `Date`, `1-Day Rain`, …
3. **Melt to long** on the Tag/Enterococcus location columns; split `Variable` into `Location` and `Measure`; **pivot** to one row per (Date, Location) with `Tag` + `Enterococcus`; cast `Enterococcus` numeric. Grain = **one row per station-reading per date** (3 stations × dates per beach).
4. **Year filter:** `Year = Date.split(', ')[-1]` → keep `2020 ≤ Year ≤ 2024`.
5. **Exceedance test:** `ex = df[Enterococcus > 104]`; a beach passes iff it has **zero** exceedances across all stations/dates (i.e. every sample < 104).
6. **File→beach-name mapping (critical):** the file `pleasure_bay_and_castle_island_beach_datasheet.csv` is **one file covering two beaches** → a clean pass on that file must emit **both** "Pleasure Bay Beach" **and** "Castle Island Beach". Final output is the display-name list; gold = {Pleasure Bay, Castle Island, City Point}.

Two hazards decide this task: (a) picking the **actual Enterococcus columns** (not the denser Rain columns), and (b) **splitting the combined Pleasure-Bay-and-Castle-Island file** into two beach names.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (anchor, PASS → `['Castle Island Beach','City Point Beach','Pleasure Bay Beach']`)

| step | action (op) | semantics | vs gold plan |
|---|---|---|---|
| 0 | `boston_harbor_beaches_txt`, `datasheets_2020_2024` | load beach-name txt; **default** `read_csv` concat of 8 files → 10972×20 (multi-header NOT handled) | load present, but headers unflattened |
| 1 | `harbor_beaches`, `datasheet_long`, `enterococcus_samples` | melt-by-column approach; classify header text → **`enterococcus_samples` = 0 rows** | plan 3 attempted, **fails loudly (empty)** |
| 2 | `enterococcus_samples` (retry, pivot_table) | different pivot; **still 0 rows** | still failing, empty |
| 3 | `samples_from_raw_datasheets` | **re-load each file raw** (`header=None, dtype=str`), tab-join each row → 10980×3; renders row-2 labels `Date\t1-Day Rain\t2-Day Rain\t3-Day Rain\tTag\tEnterococcus\tTag\tEnterococcus\tTag\tEnterococcus` | pivot to a layout it can index |
| 4 | `enterococcus_samples` (raw parse) | split raw_row on `\t`; field[0]=Date; **enterococcus = max of fields at fixed indices [5,7,9]** (the 3 Enterococcus cols) → 10955×4 `enterococcus_max` | **plan 1–3 correct**: right cols, max-across-stations = "any station exceeds" |
| 5 | `samples_2020_2024` | `2020 ≤ year ≤ 2024` → 2010 rows | plan 4 ✓ |
| 6 | `beaches_100pct_meeting_standard` | group by beach, keep `n_fail==0` (all `<104`) → **2 rows** (City Point; Pleasure Bay and Castle Island) | plan 5 ✓ |
| 7–10 | `harbor_beaches_norm`, `beaches_100pct_harbor` (2 tries) | strip " Beach" suffix, join datasheet↔harbor names; first tries → only **City Point Beach** (combined file didn't match) | plan 6 partial |
| 11 | `beaches_100pct_harbor` (final) | **detect `beach_name_norm == 'pleasure bay and castle island'` and split into `Pleasure Bay` + `Castle Island`** → 3 rows | **plan 6 ✓ (the decisive step)** |
| 12 | TEXT | Final Answer: Castle Island, City Point, Pleasure Bay | = gold set |

**First divergence:** none that survives — the arm diverges from the plan at steps 1–2 (empty melt) but **recovers**. The pass hinges on two self-generated insights: the fixed-index Enterococcus parse (step 4) and the combined-file split (step 11).
**Evidence that sustained it (quoted, rendered before step 4):**
`[samples_from_raw_datasheets] Output 10980x3: … 2  constitution_beach_datasheet.csv  constitution  Date\t1-Day Rain\t2-Day Rain\t3-Day Rain\tTag\tEnterococcus\tTag\tEnterococcus\tTag\tEnterococcus`
— i.e. the winner **rendered the raw column layout it had chosen to load**, then hard-coded indices [5,7,9]. This evidence was produced by its own method choice (raw string load after two failed melts), not surfaced by the render config.

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (C1 loser, FAIL → `['Pleasure Bay Beach, South Boston']`)

| step | action (op) | semantics | vs gold plan |
|---|---|---|---|
| 0 | `load_beach_csvs`, `load_beach_txt` | **default** `read_csv` concat → 10972×20 | headers unflattened |
| 1 | `beach_long_2020_2024` | **one monolithic op**: for each `'…Bacterial Water Quality'` column find the 'date' header row, extract Tag/Ent pairs, melt, filter years → **only 652 rows**; beach = `header.split(':')[0]` = `"Pleasure Bay Beach, South Boston"` | plan 1–4 attempted; **silently drops 7 of 8 beaches** |
| 2 | `beaches_meeting_100pct_standard_2020_2024` | group by beach, keep `n_fail==0` → **1 row** `Pleasure Bay Beach, South Boston` | plan 5 on corrupt input |
| 3 | TEXT | Final Answer: `['Pleasure Bay Beach, South Boston']` | wrong |

**First divergence (the hinge): step 1.** The monolithic extractor uses the beach-title column as its date source, which only holds dates for the file whose banner landed in column position 0 after the concat; the other files' dates fall in mis-aligned columns and are dropped by `dropna`. Result: 652 rows, essentially one beach, with an un-normalized name (`", South Boston"` suffix never stripped, so no join to harbor names either).
**Rendered evidence at decision time (before step 2):**
`[beach_long_2020_2024] Output 652x6: … 0  Pleasure Bay Beach, South Boston  2024-08-27… None  30  pleasure_bay_and_castle_island_beach_datasheet.csv  2024`
— 652 rows and a comma-suffixed name are both red flags (winner's correct parse had 2010 rows), but the arm accepted them and answered in 5 steps. **No 0-row failure fired**, so nothing forced iteration.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly (C3 loser, FAIL → 8 beaches)

| step | action (op) | semantics | vs gold plan |
|---|---|---|---|
| 0 | `beach_readme`, `beach_data_raw` | txt; **default** `read_csv` concat → 10972×20 | headers unflattened |
| 1 | `beach_data_tidy` | melt to 73106 rows + filename→beach map (`pleasure_bay_and_castle_island → None`) | **dead-end op, unused downstream** |
| 2 | `beach_samples_2020_2024` | per beach-title col treat as Date, filter years (→ correct **2010** rows); **enterococcus = the `Unnamed:*` column with the MOST numeric non-nulls on those rows** | plan 4 ✓ but **wrong measure column** |
| 3 | `beaches_meeting_100pct` | group, keep `pct_meeting == 1.0` → **8 rows** | plan 5 on wrong column |
| 4 | TEXT | Final Answer: 8 beaches (Carson, City Point, Constitution, M Street, Malibu, Pleasure Bay, Tenean, Wollaston) | wrong (gold = 3) |

**First divergence (the hinge): step 2, the enterococcus-column heuristic.** Verified on `constitution_beach_datasheet.csv`: numeric non-null counts are 1-Day Rain **1880**, 2-Day Rain 1880, 3-Day Rain 1880, Enterococcus 1877 / 1877 / 1875. The "max numeric density" rule therefore selects a **Rain** column, not Enterococcus. Rain (inches) is always < 104, so **every beach reads as 100% compliant** → 8/9 beaches returned (Castle Island lost only because the combined file mapped to "Pleasure Bay" alone).
**Rendered evidence at decision time (before step 3):**
`[beach_samples_2020_2024] Output 2010x5: … infer the Enterococcus-count column among 'Unnamed:*' by maximizing numeric non-null values on those date rows …`
— the row count (2010) looked right and the arm never questioned that 8/9 beaches passing was implausible; answered in 6 steps. All rendering here is **schema-only**, identical in kind to the winner's — no value distribution was rendered that would have exposed rain-vs-enterococcus in any arm.

## Pair verdicts

**Both losers fail for the same structural reason:** their first serious parse produced a *plausible, non-empty* result (Delta5k 652 rows/1 beach; Latest3k 2010 rows/8 beaches) that they accepted without verification. The winner's first two parses produced a *glaring* 0-row output, which forced it to keep iterating until it (a) loaded raw rows and read the true Enterococcus column positions and (b) split the combined Pleasure-Bay-and-Castle-Island file. The extra iteration did earn the pass — but it was triggered by the winner's **own code failing loudly**, and sustained by evidence the winner **generated for itself** (the raw column layout at step 3). Nothing about the render lever (3k-vs-5k cap; Delta-vs-Latest mode) caused the losers' first-step parsers to fail silently or the winner's to fail loudly. The divergence sits at steps 1–2, well before any compaction pressure differentiates the arms' contexts (all three are schema-only and saw the same 10972×20 raw table). Per the skill rule "reject method-choice divergence that predates the arms' first rendered difference" — and independently, the task is a **chronic flipper** (single flip → default variance).

- **C1 — Delta3k > Delta5k: REJECTED-method-choice** (concurring CHRONIC-VARIANCE).
  Divergence = Delta5k step 1, a monolithic single-op extractor that mis-aligns dates and silently keeps only 1 of 8 beaches (652 rows). Delta3k's melt fails to 0 rows (steps 1–2), forcing the recovery to a fixed-index raw parse. Loud-vs-silent first-parse failure, not the 3k/5k token cap.
- **C3 — Delta3k > Latest3k: REJECTED-method-choice** (concurring CHRONIC-VARIANCE).
  Divergence = Latest3k step 2, a "densest Unnamed column = enterococcus" heuristic that picks a Rain column (1880 > 1877 numeric non-nulls, verified), making ~all beaches read compliant (8 returned). Independent of Delta-vs-Latest rendering; both arms saw identical schema-only observations.

**Net:** on environment-hard-9 the pass is a fortunate consequence of the anchor's parser breaking early and forcing exploration — a method/luck artifact on a chronic-variance task — not an effect attributable to the compaction render config. No flip attributed to the lever in either pair.
