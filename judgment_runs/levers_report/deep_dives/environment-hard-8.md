# environment-hard-8 — deep dive (CHRONIC / VARIANCE)

Counter-intuitive case: the anchor **Delta3kSchemaOnly** lost to all three rays
(Delta5k, DeltaStats3k, Latest3k). This is *not* an evidence gap — the walk
found a **keep-first-station coding choice** made on evidence the loser had
rendered to it. `environment-hard-8` is on `chronic_flippers.json`.

## Task
Q: What percentage of samples (to 2 decimal places) that failed to meet the
swimming standard at Boston Harbor beaches had rainfall within 24 hours prior to
sampling? A sample meets the standard if it contains fewer than or equal to 104
counts of Enterococcus per 100 milliliters of water.
(Gold answer: **54.03**. Failure = Enterococcus > 104; "had rain" = 1-Day Rain > 0.)

D: 8 `*_beach_datasheet.csv` files + `boston-harbor-beaches.txt` (a 9-line beach
list). Every datasheet has the **same 3-row preamble and a variable number of
monitoring stations**, which is the whole task:

- **Row 0** = a title cell: `"<Beach>, <Neighborhood>: Bacterial Water Quality"`,
  rest of the row empty.
- **Row 1** = station names, sitting only above each `Tag` position (cols 4, 6,
  8, 10 …). Empty everywhere else.
- **Row 2** = the real header: `Date, 1-Day Rain, 2-Day Rain, 3-Day Rain`, then a
  repeated **`Tag, Enterococcus` pair per station**.
- **Rows 3+** = data. `Tag` holds a `<` marker (below-detection); the adjacent
  `Enterococcus` is the count. Rain is inches; "within 24 h" = `1-Day Rain > 0`.

The station count varies file to file — REAL first rows:

```
malibu (1 station, 6 cols):
  "Malibu Beach, Dorchester: Bacterial Water Quality",,,,,
  ,,,,Malibu Beach,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus
  "September 1, 2024",0,0,0,,41

carson (2 stations, 8 cols):
  "Carson Beach, South Boston: Bacterial Water Quality",,,,,,,
  ,,,,I Street,,McCormack Bathhouse,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus
  "August 27, 2024",0,0,0,,61,,41

constitution (3 stations, 10 cols):
  "Constitution Beach, East Boston: Bacterial Water Quality",,,,,,,,,
  ,,,,North,,Middle,,South,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus
  "September 1, 2024",0,0,0,<,10,<,10,<,10

wollaston (4 stations, 12 cols):
  "Wollaston Beach, Quincy: Bacterial Water Quality",,,,,,,,,,,
  ,,,,Milton Road,,Channing Street,,Sachem Street,,Rice Road,
  Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus
  "August 31, 2024",0,0,0,,135,,62,,41,,20
```

**Load-bearing quirk:** a beach with N stations has N `Enterococcus` columns, one
row of readings per date. The answer requires stacking **every station's**
Enterococcus. `read_csv` with the default header mangles all 8 files into one
20-col frame whose column names are the *title cells* — the two structural rows
(station names + real header) become data rows.

## Solution

`solutions/environment/environment-hard-8.py` processes each of the 8 CSVs
independently, then accumulates:

```
per beach CSV (×8):
  load(fp, skiprows=1, header=[0,1])              # skip title row; next 2 rows = MultiIndex header
        │
        ▼
  flatten header: ffill station name into level-0, join levels
        │   → Date, 1-Day Rain, 2-Day Rain, 3-Day Rain, {Station}_Tag, {Station}_Enterococcus × N
        ▼
  melt(id=[Date,1/2/3-Day Rain], value=EVERY *_Tag / *_Enterococcus col)   ← N stations → N rows/date
        │
        ▼
  split Variable→(Location, Measure) → pivot Measure→cols → cast Enterococcus numeric
        │
        ▼
accumulate across all 8 beaches:
  n_ex       += len(df[Enterococcus > 104])        → 2071   (per-beach [428,26,30,34,147,149,345,912])
  n_samples  += len(that[1-Day Rain > 0])          → 1119   (per-beach [288,15,15,19,79,67,160,476])
        │
        ▼
  answer = n_samples / n_ex * 100 = 1119/2071*100 = 54.03
```

Correct melt-all → **21,986 long rows → 2,071 exceedances → 54.03**. Taking only
the *first* station per beach collapses to ~1,128 rows → 79 exceedances → **53.16**.

## What Delta3kSchemaOnly does  (anchor — FAIL, 53.16; 12 steps, $0.1018)

- **0** `beach_datasheets_raw`: `glob(*_datasheet.csv)` → `read_csv(p)` default
  header → `concat` → 10972×20 mangled wide frame (title cells as column names;
  the 2 structural rows demoted to data).
- **1** `beach_long`: pull the "Bacterial Water Quality" title columns, stringify
  → 1 raw col per beach (10964×3). Discards rain and all per-station structure.
- **2** `beach_records`: state-machine positional parser — walk raw cell values,
  assign date→enterococcus→rain triples. No station awareness. → 3648×4.
- **3** `failed_samples`: `ent > 104` → **0 rows**.
- **4** `failed_rain_pct`: → percent 0.
- **5** `beach_records` (re-edit): same parser + `<`-handling → still 0.
- **6** `enterococcus_stats`: probe → `n=3648, ent_max=31, n_over_104=0`. Confirms
  the positional parse is garbage.
- **7** `raw_inspect_one` (reads carson `header=None`, head 40) + `beach_records_fixed`
  (fixed `Unnamed:1/2` offsets) + `failed_rain_pct_fixed` → still 0 exceedances.
  **This render shows the multi-station structure (see below).**
- --> **8** `beach_records_fixed2`: locates the `Date` header row in a 12-col
  window, computes `ent_candidates` = **all** Enterococcus col indices, then keeps
  `i_ent = ent_candidates[0]` — **the FIRST Enterococcus column only** → 1128 rows.
  **THE DIVERGENCE**: single-station melt, where gold melts every station. It had
  the full candidate list in hand and threw stations 2–4 away.
- **9** `failed_rain_pct_fixed2`: `ent > 104` → **79** failed, 42 rain → **53.16**.
- **10** TEXT: Final Answer **53.16**. WRONG.

(Steps 0–7 are a *separate*, shared failure — a fully broken sequential parser
producing 0 exceedances. The anchor revisits "how do I parse this" four times but
only reaches the melt question at step 8, and answers it wrong once.)

## What Latest3kSchemaOnly does  (winner — PASS, 54.03; 12 steps, $0.0868)

Same mode-independent path as the anchor, opposite melt outcome.

- **0** `load_all_beach_csvs`: glob + `read_csv` default header + concat → 10972×20
  (**same mangled load as the anchor**).
- **1** `normalize_beach_datasheets`: pull "Bacterial Water Quality" cols; sequential
  is_date / number cell parse → 10964×4 (**same broken sequential approach** as
  anchor steps 1–2; enterococcus all NaN).
- **2** `samples_with_rain_flag`: `ent > 104` + hunt for a "rain within 24h" column
  → **0 rows**.
- **3** `failed_samples`: non-null date + `ent > 104` → **0 rows**.
- **4** `inspect_raw_carson`: raw carson head(30) probe.
- **5** `inspect_clean_stats`: probe → `rows=10964, ent_max=NaN, gt104=0` (**same
  0-exceedance dead-end** as anchor step 6).
- **6** `reload_all_beach_csvs_raw`: reload `header=None, dtype=str` → 10980×14.
  Recovery pivot. **Render shows the same `I Street … McCormack Bathhouse`
  multi-station structure the anchor saw at its step 7.**
- ~~**7**~~ `parse_samples_from_raw`: locate header row; keep cols `[0,1,5,7,9,11]`;
  **melt Enterococcus replicates 5/7/9/11 (melt-ALL)** — but KeyError on integer col
  names. *Near-miss, recovered next step.*
- **8** `parse_samples_from_raw` (re-edit): identical logic, string col names →
  **21,986 rows. Melt-ALL.** RECOVERED.
- **9** `failed_sample_pct_rain24h`: `ent > 104` → **2,071**, `rain_1day > 0` →
  **1,119** → **54.03**.
- **10** TEXT: Final Answer **54.03**. CORRECT.

(Delta5kSchemaOnly and DeltaStats3kD2 also won, the *same* way — landing on the
identical 21,986 → 2,071 → 54.03 intermediate via their own code paths. The
melt-all move is not gated on any one config knob.)

## Why Latest3kSchemaOnly succeeded but Delta3kSchemaOnly failed

**The evidence was identical.** Both arms ran the same mangled glob-concat load
(10972×20), the same broken sequential-cell parser, the same probe confirming
0 exceedances, then the same pivot to a raw `header=None` reload — and both were
**rendered the same multi-station structure** before their melt decision:

- Anchor, step-7 `raw_inspect_one` (carson, in its own context):
  > `| 1  NaN NaN NaN NaN I Street NaN McCormack Bathhouse NaN | 2  Date 1-Day Rain 2-Day Rain 3-Day Rain Tag Enterococcus Tag Enterococcus`
- Latest, step-6 `reload_all_beach_csvs_raw`:
  > `| 1  NaN NaN NaN NaN I Street NaN McCormack Bathhouse NaN …`

Two `Tag Enterococcus` pairs and both station names (`I Street`, `McCormack
Bathhouse`) are plainly visible in **both** contexts.

On that identical evidence the two arms wrote opposite melt code:

- **Latest (step 8)** melted all four replicate positions —
  `value_vars=['ent_a_raw','ent_b_raw','ent_c_raw','ent_d_raw']` over columns
  `5/7/9/11`, its comment enumerating `5 entA, 7 entB, 9 entC, 11 entD` → 21,986 → 54.03.
- **Delta (step 8)** built the full list and then discarded it:
  ```python
  # pick first enterococcus column after date
  ent_candidates = [i for i, v in enumerate(header_norm) if v == 'enterococcus']
  ent_candidates = [i for i in ent_candidates if i > i_date]
  i_ent = ent_candidates[0] if ent_candidates else None   # ← first station only
  ```
  → 1,128 → 53.16.

**Smoking gun (a):** the multi-station fact was in the loser's context — rendered
as `Tag Enterococcus Tag Enterococcus` at step 7, and its own code computed
`ent_candidates` = *all* Enterococcus indices before keeping `[0]`. Not
evidence-starved (fails the skill's accept rule; matches the reject rule for
loser-had-the-evidence).

**Smoking gun (b):** Latest3k walked the **same 12 steps** — same mangled load,
same broken sequential parser, same 0-exceedance dead-end, same raw-reload
recovery render — and chose melt-all where the anchor chose melt-first. Same
evidence, same path, opposite choice: the coin-flip signature.

No config lever separates the arms. The three winners span Delta5k (window),
DeltaStats3k (window+stats) and Latest3k (mode) — three *different* knobs; there is
no single field all winners hold and the anchor lacks. Genuine lever convergence
(cf. legal-hard-15, where a lever *surfaced* a fact the loser lacked) requires the
winners to share the knob that supplies the fact. Here they don't, and the fact was
equally rendered to the loser.

**Verdict: CHRONIC / VARIANCE.** A hard multi-station-melt parse that rolls
independently per run; the anchor coded the wrong melt once on evidence it had.
This is the anti-pattern to legal-hard-15 — there the lever gave the loser a fact
it never saw; here the loser saw the fact and picked the first station anyway.
