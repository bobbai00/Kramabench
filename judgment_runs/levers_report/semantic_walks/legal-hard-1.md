# Semantic walk — legal-hard-1

## Task + gold answer

Q: "Report the average number of reported identity thefts for all metropolitan areas that are larger than one million in population in 2023. … linearly interpolate … Be sure to robustly match the names of metropolitan areas: use only the city and state portion of the name, ignoring suffixes like 'Metropolitan Statistical Area' or 'MSA' and **normalizing punctuation**. Drop entries where there's no match. Round to 4 decimal places."
Gold answer: **12964.8727**. Task is in `chronic_flippers.json` — default verdict CHRONIC/VARIANCE unless accept rules are met.

Config check (one knob): the arms' task `config.json` differ on `data_level: 2→1` + `column_stats: true→false` — together the single C2 stats-rendering lever. The loser's file also carries schema-vintage default fields (`frontier_decay_config: null`, `enable_inspect_tool: false`, …) — all inert.

Numeric ground truth (re-executed on the real data, `.venv/bin/python`, replicating each arm's exact code):

- Winner's pipeline (punct-collapsing key): join 452→412, >1M filter → 76, key-dedup → **55 metros, mean 12964.8727** — gold exactly.
- Loser's pipeline (hyphen-preserving key): key-dedup 452→401, join → 361, >1M filter → **54 metros, mean 13122.1852** — the loser's answer exactly.
- The 55-vs-54 membership diff is **exactly one MSA**: `Nashville-Davidson--Murfreesboro--Franklin, TN` (4470 reports, pop_2023 = 2,116,525.75). The FTC CSV writes the name with ASCII **double hyphens** (`Davidson--Murfreesboro`); the wiki table uses en-dashes (`Davidson–Murfreesboro`). Loser key: `nashville-davidson--murfreesboro--franklin, tn` vs wiki `nashville-davidson-murfreesboro-franklin, tn` → inner-join drop. Check: 55×12964.8727 − 54×13122.1852 ≈ 4470.
- Grep of every rendered context in BOTH arms for `Nashville` or `Davidson--`: **0 hits in either arm.** The deciding row sat in the render-elided middle everywhere it could have appeared (theft table row ~300 of 452, wiki rank ~36 of 387; all renders show head ~0–13 + tail).

## Gold semantic plan

Source: `solutions/legal/legal-hard-1.py`

| # | Plan item |
|---|---|
| G1 | Concat all state CSVs in `data/legal/input/csn-data-book-2024-csv/CSVs/State MSA Identity Theft data/` with messy-CSV header detection + numeric cleanup → cols `Metropolitan Area`, `# of Reports` (452 clean rows) |
| G2 | Parse `metropolitan_statistics.html`; select the table captioned "The 387 metropolitan statistical areas of the United States" (387×5: name, 2024 estimate, 2020 census, …) |
| G3 | 2023 population = `2020 census + (3/4)·(2024 estimate − 2020 census)` |
| G4 | Normalize names both sides: lowercase, strip `Metropolitan Statistical Area|MSA` suffix, then **strip ALL non-alphanumerics** (`re.sub(r'[^a-z0-9]+','',…)`) — key is punctuation-**run**-insensitive (`--` ≡ `–` ≡ `-` ≡ nothing) |
| G5 | `drop_duplicates` both frames (cross-state duplicate MSA rows carry identical counts) |
| G6 | Inner merge on the key (no-match entries drop, per task) |
| G7 | Filter interpolated 2023 pop > 1,000,000 |
| G8 | Mean of `# of Reports` → 12964.8727 (4 dp) |

## Walk: DataflowSystemGPT52DeltaStats3kD2 (WINNER, C2)

**PASS — Final Answer: 12964.8727.** 10 agent steps, 0 error renders, $0.0776 (cached 101,888/114,948).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | metro_html, idtheft_files | `read_html[0]` (10×1 nav junk — probe); concat all state CSVs + `__source_file` → 764×3 | G2 probe, G1 |
| 1 | metro_html_tables, idtheft_clean | probe all 6 html tables (reveals idx 1 = 387×5); filter junk rows, numeric reports → 452×3 | G2 discovery, G1 |
| 2 | metro_pop_raw | `read_html[1]` → 387×5 (the gold table) | G2 |
| 3 | metro_pop_2023 | `pop_2020 + (pop_2024 − pop_2020)·(3/4)` | G3 |
| 4 | metro_name_norm | key probe, both sides: NFKD, en-dash→`-`, strip suffix words, split city/state; city: **non-alnum runs → single space** + collapse + lower → `"city, ST"` (839 keys rendered) | G4 |
| 5 | theft_pop_join | same norm inline; inner merge theft(452)×pop(387) → 412×5 | G6 |
| 6 | metro_gt1m | `pop_2023_est > 1_000_000` → 76×4 | G7 |
| 7 | metro_gt1m_dedup | `groupby(key).agg(max)` → 55×4 (cross-state dupes collapse) | G5 |
| 8 | avg_reports_gt1m | mean → 12964.8727 | G8 |
| 9 | TEXT | Final Answer: 12964.8727 | — |

**No divergence.** Dedup after join instead of before is order-equivalent (duplicates carry identical counts). The winning property is in step 4/5's normalizer: `re.sub(r'[^A-Za-z0-9 ]+',' ',city)` collapses ANY punctuation **run** to one space, so `Davidson--Murfreesboro` (CSV) and `Davidson–Murfreesboro` (wiki) unify — same insensitivity class as gold G4.

**Corresponding render (before step 4, norm authoring)** — wiki rows with en-dashes plus the stats-arm profile block:

> `0  New York–Newark–Jersey City, NY-NJ MSA  19940274  20081935  −0.71% …`
> `Column Schema and stats:`
> `- "Metropolitan statistical area" (str): null=0, distinct=387`
> `- "Encompassing combined statistical area" (str): null=112, distinct=160, duplicate_values=115`   ← the 28.9% max-null of the engine profile (112/387)
>
> and on `idtheft_clean`: `- "metro_raw" (str): null=0, distinct=401, duplicate_values=51`

The profile lines carry null/distinct/mean only — **no punctuation or name-variant information**. The one actionable line (`distinct=401, duplicate_values=51` → cross-state dupes) informs G5, which the loser performed anyway without it.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (loser, C2)

**FAIL — Final Answer: 13122.1852** (gold 12964.8727). 9 agent steps, 6 steps with error renders, $0.1840 (cached 72,448/113,607).

| step | op | semantics (from code) | plan-item |
|---|---|---|---|
| 0 | msa_id_theft_raw, metropolitan_html | concat state CSVs → 764×2; load html as **raw text** 1×1 | G1; G2 (raw-text method) |
| 1 | msa_id_theft_2023 | filter junk, numeric reports → 452×2 | G1 |
| 2 | metro_populations v1, **msa_id_theft_keys**, metro_big_joined, final_answer | BS4 heuristic (year,pop) regex over all wikitables → **ERROR** (KeyError `metro_key`, 0 rows); theft key norm: en-dash→`-`, strip suffixes, city + first state token, `re.sub(r'[^A-Za-z0-9,\-\s\.]',' ',s)` — **preserves hyphen runs**; merge+filter+mean wiring | **G4 shortfall (latent, decides answer)**; G2/G3 attempt fails |
| 3 | metro_populations v2 | wikitable with header years 2010 AND 2020 → none → 0×0; downstream KeyError | G2 miss |
| 4 | metro_populations v3 | `read_html`, target cols '2010'+'2020' → none → empty (render unchanged) | G2 miss |
| 5 | metro_tables_probe, msa_id_theft_keys | probe all 6 tables (reveals idx 1: 387×5, '2024 estimate'/'2020 census'); theft keys 452→401 dedup | G2 discovery; G5 |
| 6 | metro_populations v4 | target '2024 estimate'+'2020 census' table; `p2020 + (p2024−p2020)·(3/4)`; same norm → 387×4 | G2 ✓, G3 ✓ |
| 7 | metro_big_joined | inner merge on `metro_key`, filter >1M, mean → 54 metros, 13122.185 | G6, G7, G8 (G4 defect bites) |
| 8 | TEXT | Final Answer: 13122.1852 | — |

**First divergence: step 2, `msa_id_theft_keys` norm_key** — "normalizing punctuation" (G4) done only for en-dash→hyphen; the character class `[^A-Za-z0-9,\-\s\.]` keeps `-`, so hyphen **runs** survive verbatim and CSV `--` ≠ wiki `-` for the one MSA where they differ (Nashville, 4470 reports, >1M) → 54-metro mean 13122.1852. The step-2/3/4 population-parse failures (G2) are a transient second divergence, self-recovered by step 6 via the tables probe (metro_populations resubmission similarity 0.273/0.176/0.384 — progressive rewrites, not identical-probe thrash; no churn flag: 8 ops, single sink).

**Evidence at divergence (rendered before step 2)** — schema-only, and nothing showing punctuation variance beyond single hyphens:

> `[msa_id_theft_2023] … 0  Anniston-Oxford, AL Metropolitan Statistical Area  264 … 451  Cheyenne, WY Metropolitan Statistical Area  140`
> `Schema (2 cols): metro_raw (str), reports_2023 (numeric)`   ← no stats block (the C2 knob)
> `[metropolitan_html] 0  <!DOCTYPE html>\n<!-- saved from url=(0059)https://en.wikipedia.org/wiki/Metropolitan_statistical_are…`

The loser had seen NO wiki rows yet, but pre-mapped en-dashes anyway (`–/—/− → '-'`) — it handled the entire mismatch class its (and the winner's) renders ever exhibited. Post-fix render (before step 7) showed both key columns looking consistent (`anniston-oxford, al` … `new york-newark-jersey city, ny-nj`) and the join succeeding at a plausible 54/13122 — no mismatch signal to trigger a normalizer revisit.

## Pair verdicts

**C2 Stats3kD2 > Delta3kSchemaOnly: CHRONIC-VARIANCE** (chronic task; accept rules NOT met; the ATTRIBUTED claim would independently fail as REJECTED-method-choice).

- **The C2 hypothesis is refuted.** The stats arm's profile lines for `metropolitan_statistics.html` (387 rows, `null=112` = 28.9% max-null, str columns) rendered in the winner and not in the loser, but their content — null counts, distinct counts, numeric means — has no channel to the deciding fact (ASCII `--` vs en-dash hyphen runs in one MSA name). The only behavior-relevant stats line (`distinct=401, duplicate_values=51` → dedup) drove nothing differential: the loser deduped keys without it.
- **The deciding fact was rendered in NEITHER arm** (grep over every step's context: 0 hits for `Nashville`/`Davidson--` in both). Winner's evidence does not explain its action (its punct-run-collapsing normalizer exceeded anything its renders demanded — the visible en-dash/hyphen mismatch is equally fixed by the loser's en-dash mapping); loser's absence does not explain its error (no render it lacked contained the variant). Accept rule 5 fails on both prongs.
- **Divergence class: normalizer-robustness code style**, authored at loser step 2 / winner step 4 with no decision-relevant rendered difference between the arms at those points — method-choice luck on a never-rendered value quirk, exactly the coin-flip profile of a chronic flipper.
- Secondary (cost, not accuracy): the loser's step-0 choice to load the html as raw text (made on the identical initial prompt) cost a 4-rewrite recovery loop (6 error renders, output 7092 vs 2636 tokens, $0.184 vs $0.078) before converging on the same G2/G3 semantics — also method-choice, predating any inter-arm rendered difference.
