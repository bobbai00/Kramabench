# environment-hard-9 — deep dive (counter-intuitive: 3k-solo win)

**Counter-intuitive case:** the LEAST render budget wins. `Delta3kSchemaOnly` (f1=1.0) beats
both the larger-budget `Delta5kSchemaOnly` (f1=0.0) and the full-history `Latest3kSchemaOnly`
(f1=0.36). The lesson below is that the winner was not better-*informed* — its own parser
crashed early and *loudly*, and that self-generated failure is what forced the correct method.

| arm | f1 | steps | cost_usd | in/out/cached tok | verdict |
|---|---|---|---|---|---|
| **Delta3kSchemaOnly** (mode Y, winner) | **1.0** | 14 | 0.1088 | 146752 / 3999 / 129536 | PASS |
| Delta5kSchemaOnly (mode X₁, C1 loser) | 0.0 | 5 | 0.0379 | 37802 / 1651 / 32640 | FAIL |
| Latest3kSchemaOnly (mode X₂, C3 loser) | 0.36 | 6 | 0.0402 | 40799 / 1905 / 36736 | FAIL |

**Label:** C1 (Delta3k > Delta5k) = **REJECTED-method-choice** (concurring **CHRONIC-VARIANCE**);
C3 (Delta3k > Latest3k) = **REJECTED-method-choice** (concurring **CHRONIC-VARIANCE**).
`environment-hard-9` is on the 23-task chronic-flipper list → a single flip defaults to variance.

## Task

Q: *"Which Boston Harbor beaches met swimming standards 100% of the time between 2020 and 2024
(inclusive)? A sample meets the standard if it contains fewer than 104 counts of Enterococcus
per 100 milliliters of water."*
Gold answer (`list_exact`, set-compared): **`['Pleasure Bay Beach', 'Castle Island Beach', 'City Point Beach']`**.

D: 8 beach datasheet CSVs (fixed list in the gold) + one beach-name index txt. All datasheets
share a 3-header-row layout with a fatal quirk: **variable width** (stations are interleaved
`Tag`/`Enterococcus` pairs, so column count changes file-to-file), and one file secretly holds
**two** beaches.

- `constitution_beach_datasheet.csv` — 3 stations. Real rows (first 4 lines):
  ```
  "Constitution Beach, East Boston: Bacterial Water Quality",,,,,,,,,   <- row0: banner in col0 only
  ,,,,North,,Middle,,South,                                            <- row1: STATION level (sits above each Tag col)
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus  <- row2: measure labels
  "September 1, 2024",0,0,0,<,10,<,10,<,10
  ```
  Semantics: `Date` = "Month DD, YYYY" string. `1/2/3-Day Rain` = rainfall **inches** (small
  floats, e.g. 0.00–0.82). `Tag` = detection flag (`<` = below detection limit, else blank).
  `Enterococcus` = the measured count (integers 2, 10, 20, 60…). One `Tag`+`Enterococcus` pair
  **per station**; the station name (row1) sits above the `Tag` cell, leaving the `Enterococcus`
  header cell blank — this is why a proper parse needs `header=[0,1]` + forward-fill.

- `pleasure_bay_and_castle_island_beach_datasheet.csv` — **the two-beach file.** Real rows:
  ```
  "Pleasure Bay Beach, South Boston: Bacterial Water Quality",,,,,,,,,   <- banner names ONLY Pleasure Bay
  ,,,,Pleasure Bay @ Broadway,,Pleasure Bay @ Flagpole,,Castle Island Playground,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus
  "August 27, 2024",0.00,0,0,,30,,10,,20
  "August 20, 2024",0.01,0.12,0.15,,10,,10,,60
  ```
  Semantics: 3 stations — **2 are Pleasure Bay, 1 is Castle Island** — inside one file whose
  banner mentions only "Pleasure Bay Beach". A clean pass on this file must emit **both**
  "Pleasure Bay Beach" and "Castle Island Beach". (Verified gold parse: 231 rows in 2020–2024,
  max Enterococcus = 86 < 104, **0 exceedances** → both beaches pass.)

- `city_point_beach_datasheet.csv` — **1 station**, so only 6 columns:
  ```
  "City Point Beach, South Boston: Bacterial Water Quality",,,,,
  ,,,,City Point at Farragut Road,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus
  "August 27, 2024",0,0,0,<,10
  ```
- Width varies across the 8 files by station count: **6 cols** (1 station: city_point, m_street,
  malibu, tenean), **8 cols** (2 stations: carson), **10 cols** (3 stations: constitution,
  pleasure_bay), **12 cols** (4 stations: wollaston). Enterococcus is always at odd indices
  5/7/9/11; rain always at 1/2/3; Date at 0. A naive default-`read_csv` + `concat` of files with
  different widths mis-aligns every column past the first station.

- `boston-harbor-beaches.txt` — 9 display names, one per line: Constitution, **Castle Island**,
  **Pleasure Bay**, City Point, M Street, Carson, Malibu, Tenean, Wollaston. **9 names, 8 files**
  — the mismatch is exactly the combined Pleasure-Bay-and-Castle-Island file.

**The "densest Unnamed column" trap (verified on constitution).** Under default `read_csv` the
banner becomes the header and the rest become `Unnamed: 1…9`. Numeric non-null counts:

| Unnamed col | real measure | numeric non-nulls |
|---|---|---|
| Unnamed: 1 | 1-Day Rain | **1880** |
| Unnamed: 2 | 2-Day Rain | 1880 |
| Unnamed: 3 | 3-Day Rain | 1880 |
| Unnamed: 5 | Enterococcus (North) | 1877 |
| Unnamed: 7 | Enterococcus (Middle) | 1877 |
| Unnamed: 9 | Enterococcus (South) | 1875 |

The **Rain** columns are denser than Enterococcus (1880 > 1877) because a station occasionally
skips a reading but rain is logged every day. So "pick the densest Unnamed column as
Enterococcus" selects a **rain** column — and rain (inches) is always < 104, so every beach reads
as 100% compliant.

## Solution

Per beach CSV in the fixed 8-file list (gold: `solutions/environment/environment-hard-9.py`):

```
read_csv(fp, skiprows=1, header=[0,1])          # skip banner row0; rows 1+2 = 2-level header
        │                                        #   L0 = station (North/…/Broadway/…), L1 = Date|Rain|Tag|Enterococcus
        ▼
flatten headers (ffill station across Unnamed, join levels)
        │   -> North_Enterococcus, Middle_Enterococcus, …  + flat Date, 1-Day Rain, …
        ▼
melt(id=[Date,Rain…], value=Tag/Enterococcus cols) → Variable/Value
        │
        ▼
split Variable → Location(station) + Measure(Tag|Enterococcus)
        │
        ▼
pivot(index=id+Location, cols=Measure) → 1 row per (Date, station); cast Enterococcus numeric
        │   grain = one station-reading per date  (3 stations × dates per beach)
        ▼
filter  Year = Date.split(', ')[-1];  keep 2020 ≤ Year ≤ 2024
        │
        ▼
exceedance test  ex = df[Enterococcus > 104];  beach PASSES iff len(ex) == 0
        │
        ▼
file→beach mapping  (combined file ⇒ emit BOTH "Pleasure Bay Beach" AND "Castle Island Beach")
        │
        ▼
answer = display-name list  = {Pleasure Bay, Castle Island, City Point}
```

Two hazards decide the task: **(a)** select the *actual Enterococcus* columns (indices 5/7/9),
not the denser Rain columns; **(b)** *split* the combined Pleasure-Bay-and-Castle-Island file into
two beach names.

## What Delta5kSchemaOnly does (mode X₁ — monolithic extractor, silent mis-alignment)

- **step 0** `load_beach_csvs`, `load_beach_txt` — default `read_csv` + concat of 8 files →
  10972×20; multi-header never flattened, station level lost.
- **step 1** `beach_long_2020_2024` — **one monolithic op**: for each `"…Bacterial Water Quality"`
  column find its 'date' header row, pull Tag/Enterococcus pairs, melt, filter years →
  **only 652 rows**; beach name = `header.split(':')[0]` = `"Pleasure Bay Beach, South Boston"`.
  **--> DIVERGENCE (the hinge).** It uses the beach-*title* column as its date source; after the
  naive concat, that column holds dates only for the file whose banner landed in position 0. Every
  other file's dates fall in mis-aligned columns and are dropped by `dropna` → the op **silently
  keeps ~1 of 8 beaches**, with the raw comma-suffixed name never normalized.
- **step 2** `beaches_meeting_100pct_standard_2020_2024` — group by beach, keep `n_fail==0` →
  **1 row** (`Pleasure Bay Beach, South Boston`).
- **step 3** TEXT — Final Answer `['Pleasure Bay Beach, South Boston']`. Wrong (1 beach, bad name).
  Answered in **5 steps**; no 0-row failure ever fired, so nothing forced iteration.

## What Latest3kSchemaOnly does (mode X₂ — densest-Unnamed heuristic)

- **step 0** `beach_readme`, `beach_data_raw` — txt; default `read_csv` + concat → 10972×20.
- **step 1** `beach_data_tidy` — melt to 73106 rows + filename→beach map (combined file → `None`).
  Dead-end op, unused downstream.
- **step 2** `beach_samples_2020_2024` — treat each beach-title column as Date, filter years
  (→ correct **2010 rows**); **enterococcus = the `Unnamed:*` column with the MOST numeric
  non-nulls on those rows.** **--> DIVERGENCE (the hinge).** Verified: densest Unnamed on
  constitution = `Unnamed: 1` = **1-Day Rain** (1880 > Enterococcus's 1877). The rule picks a
  **Rain** column, which is always < 104.
- **step 3** `beaches_meeting_100pct` — group, keep `pct_meeting == 1.0` → **8 rows** (rain-benign
  ⇒ everything "compliant").
- **step 4** TEXT — Final Answer = 8 beaches (Carson, City Point, Constitution, M Street, Malibu,
  Pleasure Bay, Tenean, Wollaston). Wrong (gold = 3). Castle Island is missing only because the
  combined file mapped to "Pleasure Bay" alone. Answered in **6 steps**; the plausible 2010-row
  count was never questioned, nor was the implausible 8/9-pass rate.

## What Delta3kSchemaOnly does (mode Y — winner; recovered from two loud near-misses)

- **step 0** `boston_harbor_beaches_txt`, `datasheets_2020_2024` — load names txt; default
  `read_csv` concat → 10972×20 (headers unflattened, same start as both losers).
- **step 1** `harbor_beaches`, `datasheet_long`, `enterococcus_samples` — melt-by-column, classify
  header text → **`enterococcus_samples` = 0 rows.** *Near-miss #1 — fails LOUDLY (empty).*
- **step 2** `enterococcus_samples` (retry, `pivot_table`) — different pivot → **still 0 rows.**
  *Near-miss #2 — still empty.* (The two crashes are what force the method change.)
- **step 3** `samples_from_raw_datasheets` — **re-load each file raw** (`header=None, dtype=str`),
  tab-join each row → 10980×3. Renders the row-2 label string
  `Date\t1-Day Rain\t2-Day Rain\t3-Day Rain\tTag\tEnterococcus\tTag\tEnterococcus\tTag\tEnterococcus`.
- **step 4** `enterococcus_samples` (raw parse) — split raw_row on `\t`; `field[0]`=Date;
  **enterococcus = max of fields at fixed indices [5,7,9]** → 10955×4 `enterococcus_max`.
  Right columns; max-across-stations = "any station exceeds". (Re-derives gold plan 1–3.)
- **step 5** `samples_2020_2024` — `2020 ≤ year ≤ 2024` → **2010 rows**. (plan 4 ✓)
- **step 6** `beaches_100pct_meeting_standard` — group by beach, keep `n_fail==0` → **2 rows**
  (City Point; the combined Pleasure-Bay-and-Castle-Island file). (plan 5 ✓)
- **steps 7–10** `harbor_beaches_norm`, `beaches_100pct_harbor` (2 tries) — strip " Beach" suffix,
  join datasheet↔harbor names; first tries yield only **City Point** (combined file didn't match).
- **step 11** `beaches_100pct_harbor` (final) — **detect `beach_name_norm == 'pleasure bay and
  castle island'` and split into `Pleasure Bay` + `Castle Island`** → **3 rows.** (plan 6 ✓ — the
  decisive move.)
- **step 12** TEXT — Final Answer `['Castle Island Beach', 'City Point Beach', 'Pleasure Bay
  Beach']` = gold set. **14 steps, f1=1.0.**

**First divergence:** none survives. The winner diverges from the plan at steps 1–2 (empty melt)
but **recovers**, and the pass hinges on two *self-generated* insights: the fixed-index
Enterococcus parse (step 4) and the combined-file split (step 11).

## Why Y succeeded but X failed

**The rescue evidence was self-generated by the winner's own loud failure — not provided by the
render config.** All three arms began identically (default `read_csv` concat → the same 10972×20
raw table) and all three render **schema-only** observations. The arms differ only in what caused
their *first serious parse* to succeed or fail — and that is a method choice, not a lever.

- **The losers accepted a plausible, non-empty first parse without verification.** Delta5k's
  monolithic extractor emitted 652 rows / one comma-suffixed beach; its context at the decision
  step rendered exactly that — `[beach_long_2020_2024] Output 652x6: … Pleasure Bay Beach, South
  Boston …`. A 652-row count and a `", South Boston"` suffix are both red flags (the winner's
  correct parse had **2010** rows and clean names), yet nothing crashed, so it answered in 5 steps.
  Latest3k's context rendered `[beach_samples_2020_2024] Output 2010x5: … infer the
  Enterococcus-count column among 'Unnamed:*' by maximizing numeric non-null values …` — a
  *correct-looking* 2010-row count that masked a wrong measure column; it never questioned 8/9
  beaches passing.

- **The winner's first two parses returned 0 rows, which forced iteration.** Only after two loud
  empty outputs did it switch to a raw string load — and that choice **rendered its own rescue
  evidence**: `[samples_from_raw_datasheets] Output 10980x3: … Date\t1-Day Rain\t2-Day Rain\t3-Day
  Rain\tTag\tEnterococcus\tTag\tEnterococcus\tTag\tEnterococcus`. Seeing the literal interleaved
  layout is what let it hard-code Enterococcus indices `[5,7,9]`. Nothing in the 3k cap or the
  Delta mode surfaced that string — the winner's method produced it.

- **The evidence at each divergence step was identical *in kind* across arms.** The Rain-vs-
  Enterococcus distinction (constitution: rain 1880 > enterococcus 1877 non-nulls) is a **value
  distribution** that *no arm rendered* — schema-only observations cannot expose it, so this is not
  something 5k vs 3k or Delta vs Latest could have fixed. The divergences sit at **steps 1–2**,
  before any compaction pressure differentiates the three contexts.

Per the skill rule — *"reject method-choice divergence that predates the arms' first rendered
difference"* — neither flip is attributable to the render lever. Independently, the task is a
**chronic flipper** (single flip → default variance).

**Counter-intuitive takeaway:** the least-budget arm won not because it saw more, but because its
parser happened to fail *early and loudly*, forcing exploration that regenerated the layout it
needed. **A silent, plausible mis-parse is worse than a crash:** Delta5k's mis-aligned 652 rows and
Latest3k's rain-column heuristic both looked fine and were accepted; the winner's 0-row explosions
looked broken and got fixed. This is a method/luck artifact on a chronic-variance task, not a
compaction-lever effect. No flip attributed to the lever in either pair.
