# environment-hard-8 — semantic trace walk (triple-convergence candidate)

## Task + gold answer

**Query:** What percentage of samples (to 2 dp) that failed the swimming standard
at Boston Harbor beaches had rainfall within 24 h prior to sampling? A sample
meets the standard if Enterococcus ≤ 104 counts / 100 mL. (So a *failure* =
Enterococcus > 104; "had rain" = 1-Day Rain > 0.)

**Gold answer:** `54.03`

**Arms (all vs the same anchor):**

| Arm | Role | Steps | cost_usd | Answer | Verdict |
|---|---|---|---|---|---|
| Delta5kSchemaOnly (C1) | winner | 6 | 0.0532 | 54.03 | PASS |
| DeltaStats3kD2 (C2) | winner | 5 | 0.0496 | 54.03 | PASS |
| Latest3kSchemaOnly (C3) | winner | 12 | 0.0868 | 54.03 | PASS |
| **Delta3kSchemaOnly (anchor)** | **loser (all 3)** | **12** | **0.1018** | **53.16** | **FAIL** |

environment-hard-8 **IS on `chronic_flippers.json`** (with env-hard 7/9/10/11/13).

## Gold semantic plan

The gold (`solutions/environment/environment-hard-8.py`) processes **each of the
8 beach CSVs independently** with a specific header dance:

1. **Load spec (per file):** `pd.read_csv(fp, skiprows=1, header=[0,1])` — skip
   the title row, read the *next two rows* as a 2-level MultiIndex header.
2. **Flatten header:** forward-fill the empty first-level cells (the station
   names like "I Street", "McCormack Bathhouse"), then join levels →
   canonical columns `Date, 1-Day Rain, 2-Day Rain, 3-Day Rain, Tag, Enterococcus`
   (one `Tag`/`Enterococcus` pair *per station*).
3. **Melt all station measurement columns:** `location_cols` = every column
   containing `Tag` or `Enterococcus`; melt them long so **every (date, station)
   pair becomes its own record**. This is the crux — a beach with N stations
   yields N rows per date.
4. **Split** the melted `Variable` into `Location` + `Measure`, pivot back so
   `Enterococcus` and `Tag` are columns, cast `Enterococcus` numeric.
5. **Filter + accumulate across all beaches:** `ex_df = df[Enterococcus > 104]`
   (total failures `n_ex`), then `ex_df[ex_df['1-Day Rain'] > 0]` (failures with
   rain `n_samples`).
6. **Final:** `n_samples / n_ex * 100`.

Gold per-beach exceedance counts (subtask 6): `[428,26,30,34,147,149,345,912]`
→ **n_ex = 2071**. With-rain (subtask 7): `[288,15,15,19,79,67,160,476]`
→ **n_samples = 1119**. `1119 / 2071 * 100 = 54.03`.

**The load-bearing fact:** the datasheets are multi-station. Column counts seen
in the traces: wollaston 12 cols (4 stations: Milton/Channing/Sachem/Rice),
constitution & pleasure-bay 10 cols (3 stations), carson 8 cols (2 stations:
I Street/McCormack), the other four 6 cols (1 station). **You must stack the
Enterococcus column of EVERY station.** Correct melt → 21,986 long rows →
2,071 exceedances → 54.03. Taking only the *first* station's Enterococcus per
beach collapses to ~1,128 rows → 79 exceedances → the wrong 53.16.

## Walk: Delta3kSchemaOnly (ANCHOR — FAIL, 53.16)

Glob-concats all CSVs with default header into one mangled 20-col wide frame,
then thrashes a sequential-cell parser, recovers to a header-locating parser,
but grabs only the **first** Enterococcus column per beach.

| Step | Action (op) | Semantics | Matches gold? |
|---|---|---|---|
| 0 | `beach_datasheets_raw` | `glob(*_datasheet.csv)` → `read_csv(p)` default header → `concat` (20 cols) | ✗ mangles the 2-row header; loses per-file shape |
| 1 | `beach_long` | pull "Bacterial Water Quality" title columns; stringify → 1 raw col/beach | ✗ discards rain + all-but-title structure |
| 2 | `beach_records` | state-machine: walk raw values, assign date→ent→rain triples | ✗ positional parse, no station awareness |
| 3 | `failed_samples` | `ent > 104` → **0 rows** | ✗ parse broken |
| 4 | `failed_rain_pct` | → percent = 0 | ✗ |
| 5 | `beach_records` (re-edit) | same parser + `<`-handling | ✗ still 0 exceedances |
| 6 | `enterococcus_stats` | probe: **ent max = 31, n_over_104 = 0** | ✗ confirms parse garbage |
| 7 | `raw_inspect_one` + `beach_records_fixed` + `_pct_fixed` | inspect carson raw (header=None); re-parse via `Unnamed:1/2` fixed offsets | ✗ still 0 exceedances |
| 8 | `beach_records_fixed2` | locate `Date` header row in a col-window; **`i_ent = ent_candidates[0]`** — FIRST Enterococcus only → 1,128 rows | ✗ **single-station melt** |
| 9 | `failed_rain_pct_fixed2` | `ent>104` → **79 failed**, 42 rain | ✗ undercounts (should be 2071) |
| 10 | TEXT | Final Answer **53.16** | ✗ |

**First divergence: step 8 (`beach_records_fixed2`).** After finally seeing the
raw structure at step 7, its recovery code deliberately kept only
`ent_candidates[0]` — the first Enterococcus column — collapsing every
multi-station beach to a single station. This is the exact melt decision every
winner gets right. Note the earlier churn (steps 1–7) is a *different* failure
(a fully broken positional parser producing 0 exceedances); it revisits the
"how do I parse this" question 4 times but never the melt-all-stations question,
which it only reaches at step 8 and answers wrong once.

**Evidence at decision time (step 7 render, in the anchor's own context):**

> `[raw_inspect_one] … 2  Date  1-Day Rain  2-Day Rain  3-Day Rain  Tag  Enterococcus  Tag  Enterococcus`

The carson header row it rendered shows **`Tag Enterococcus Tag Enterococcus`**
— two station pairs, plainly visible. Row 1 above it rendered
`I Street … McCormack Bathhouse`. **The multi-station fact was in the anchor's
context.** Its `beach_records_fixed2` even computes `ent_candidates` = *all*
Enterococcus indices, then throws away everything past `[0]`. Not
evidence-starved — a coding choice on evidence it had.

## Walk: Delta5kSchemaOnly (C1 — PASS, 54.03)

| Step | Action (op) | Semantics | Matches gold? |
|---|---|---|---|
| 0 | `beach_datasheets` | glob + concat (20 cols) + `source_file` | same mangled load as anchor |
| 1 | `beach_samples_long` | per source_file: find header row; collect **`entero_cols` as a LIST** (append every 'enterococcus'); iterate `pair_ct = max(len(ent_cols),len(tag_cols))` **stations**; emit (beach, station, date, rain_1d, ent) | ✓ **melts ALL stations** |
| 2 | `failed_samples` | `ent > 104` → **2,071 rows** | ✓ matches gold n_ex |
| 3 | `failed_samples_rain_pct` | `rain_1d > 0`.mean()*100 | ✓ |
| 4 | TEXT | Final Answer **54.03** | ✓ |

**First divergence: none against gold.** Got the melt right on the first
processing step (21,986 → 2,071 → 54.03), no churn.

**Evidence:** `[beach_samples_long] Output 21986x6 … Carson Beach … I Street …`
then `[failed_samples] Output 2071x6`. The 21,986 / 2,071 intermediates are the
tell it stacked all stations.

## Walk: DeltaStats3kD2 (C2 — PASS, 54.03)

| Step | Action (op) | Semantics | Matches gold? |
|---|---|---|---|
| 0 | 9 separate loads: txt + **each beach CSV individually** | `read_csv(p)` per file — preserves per-file shape | ✓ different, better strategy |
| 1 | `harbor_long` | per file: rename col0→Date, drop header rows; rain = `Unnamed:1`; **ent_cols = `Unnamed:5/7/9/11`** (all station enterococcus cols); melt each → long | ✓ **melts ALL stations** |
| 2 | `failed_with_rain` | `ent>104`, `rain_1d>0`.mean()*100 | ✓ |
| 3 | TEXT | Final Answer **54.03** | ✓ |

**First divergence: none.** Reached 21,986 long rows → 54.03 in one processing
step.

**Evidence (step-0 render, this arm's richer per-file view):** loading each file
separately surfaced the station structure directly in the observations, e.g.

> `[wollaston_beach] Output 1906x12 … 0 NaN NaN NaN NaN Milton Road NaN Channing Street NaN Sachem Street NaN Rice Road NaN | 1 Date 1-Day Rain 2-Day Rain 3-Day Rain Tag Enterococcus Tag Enterococcus Tag Enterococcus Tag Enterococcus`
> `[carson_beach] Output 1134x8 … I Street NaN McCormack Bathhouse … Tag Enterococcus Tag Enterococcus`

Each file's true column count (12 / 10 / 8 / 6) and repeated `Tag Enterococcus`
pairs were rendered, motivating `Unnamed:5/7/9/11`. But note: the *same* pairs
were also rendered to the anchor at its step 7.

## Walk: Latest3kSchemaOnly (C3 — PASS, 54.03)

| Step | Action (op) | Semantics | Matches gold? |
|---|---|---|---|
| 0 | `load_all_beach_csvs` | glob + concat default header (20 cols) | same mangled load as anchor |
| 1 | `normalize_beach_datasheets` | pull "Bacterial Water Quality" cols; sequential date/number cell parse | ✗ same broken approach as anchor step 1–2 |
| 2 | `samples_with_rain_flag` | `ent>104` + detect a "rain within 24h" column | ✗ **0 rows** |
| 3 | `failed_samples` | non-null date + `ent>104` | ✗ **0 rows** |
| 4 | `inspect_raw_carson` | raw carson head(30) | probe |
| 5 | `inspect_clean_stats` | probe: **ent_max NaN, gt104 = 0** | ✗ confirms broken (same as anchor step 6) |
| 6 | `reload_all_beach_csvs_raw` | reload `header=None, dtype=str` (14 cols) | recovery pivot |
| 7 | `parse_samples_from_raw` | locate header row; keep cols `[0,1,5,7,9,11]`; **melt ent replicates 5/7/9/11** | ✓ melt-all — but KeyError (int col names) |
| 8 | `parse_samples_from_raw` (re-edit) | same, string col names → **21,986 rows** | ✓ **melts ALL stations** |
| 9 | `failed_sample_pct_rain24h` | `ent>104` → **2,071**, rain>0 → **1,119** | ✓ |
| 10 | TEXT | Final Answer **54.03** | ✓ |

**First divergence: steps 1–3 diverge from gold identically to the anchor**
(broken sequential parse, 0 exceedances). **C3 then recovers at step 7–8 by
melting columns 5/7/9/11 — the exact decision the anchor gets wrong at its
step 8.**

**Evidence (step-6 render, C3's recovery pivot):**

> `[reload_all_beach_csvs_raw] Output 10980x14 … 1 NaN NaN NaN NaN I Street NaN McCormack Bathhouse NaN NaN …`

This is the **same** `I Street … McCormack Bathhouse` multi-station render the
anchor saw at its step 7. On this same evidence C3 melted all four replicate
columns; the anchor took only the first.

## Pair verdicts

The task asks: do all three winners fix the **same missing fact** (→ strong
dual/triple-lever convergence) or is the anchor's fail a **chronic roll**?

**The missing fact is unambiguously the same across all three winners:** *melt
the Enterococcus column of every station, not just the first.* All three land on
the identical intermediate — 21,986 long rows → 2,071 exceedances → 1,119 with
rain → 54.03 — via three different code paths. The anchor collapses to the first
station (`ent_candidates[0]`) → 1,128 rows → 79 exceedances → 53.16.

**But this convergence is NOT lever-attributable. Four reasons it reads as
variance:**

1. **No shared lever separates winners from anchor.** The winners span
   Delta5k-schema (C1), DeltaStats3k (C2) and Latest3k (C3); the anchor is
   Delta3k-schema. C1 differs by *window* (5k), C2 by *window+stats*, C3 by
   *mode* (Latest). Three *different* knobs — there is no single field the three
   winners hold and the anchor lacks. Genuine lever convergence (cf.
   legal-hard-15) needs the winners to share the knob that supplies the fact.
2. **The anchor was NOT evidence-starved.** Its own step-7 render showed
   `Tag Enterococcus Tag Enterococcus` for carson and its code computed the full
   `ent_candidates` list — then discarded all but `[0]`. The missing fact was in
   its context; it chose melt-first anyway. Per the skill's reject rule, the
   loser's error is not explained by an absence of evidence.
3. **C3 is the smoking gun for a coin-flip.** C3 traversed the *same* mangled
   glob-concat, the *same* broken sequential parser, the *same* 0-exceedance
   dead end, and the *same* raw-reload render (`I Street … McCormack`) as the
   anchor — then melted all stations where the anchor melted one. Same evidence,
   same path, opposite melt choice = the evidence-starved-coin-flip signature,
   pointing at variance, not a lever.
4. **Churn didn't cause the loss.** C3 (winner) and the anchor (loser) both ran
   **12 steps** and churned comparably; C2/stats ran 5. The anchor's 12-op churn
   is real but C3 shows equal churn is survivable — the differentiator is purely
   the one melt decision, made on comparable renders.

- **C1 vs anchor (Delta5k winner):** **CHRONIC / VARIANCE.** C1 wrote correct
  melt-all code first try; anchor melted first-only. Same class of load, no lever
  supplies the fact. Chronic-listed task.
- **C2 vs anchor (DeltaStats3k winner):** **CHRONIC / VARIANCE.** C2's
  separate-file loads surfaced per-file station structure cleanly, but that was
  an agent strategy choice, not forced by the stats/window knob — and the same
  `Tag Enterococcus Tag Enterococcus` pairs were rendered to the anchor too.
- **C3 vs anchor (Latest3k winner):** **CHRONIC / VARIANCE (strongest signal).**
  Identical broken path and identical recovery render as the anchor, opposite
  melt outcome. Textbook coin-flip.

**Overall: NOT a triple-lever convergence. CHRONIC / VARIANCE on all three
pairs.** The three winners do converge on the same correct fact (melt all
stations), but that fact was equally rendered/available to the anchor, no config
lever separates the arms, and C3 succeeded on byte-identical evidence and path to
the anchor's failure. environment-hard-8 is a hard multi-station-melt parse that
rolls independently per run; the anchor drew the wrong melt once. This is the
anti-pattern to legal-hard-15 (where a lever *surfaced* a fact the loser lacked)
— here the loser had the fact and chose wrong.
