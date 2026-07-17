# Semantic walk: legal-hard-2 (PROBE-STAR, C1p)

## Task + gold answer

**Question:** Which metropolitan area is the one with the highest rate of identity thefts per 100,000 population — interpolate 2023 population linearly from known censuses; robustly match MSA names (city+state portion only, normalize punctuation, drop non-matches); ignore csv orderings.

**Gold answer:** Miami-Fort Lauderdale-West Palm Beach FL Metropolitan Statistical Area (string_approximate).

**Judge marks:** Delta5kSchemaOnlyProbePrompt `llm_paraphrase=1` (answer "Miami-Fort Lauderdale-West Palm Beach, FL Metropolitan Statistical Area", PASS); Delta1kSchemaOnlyProbePrompt `llm_paraphrase=0` (answer "Philadelphia–Camden–Wilmington, PA-NJ-DE-MD MSA", FAIL). B-only win in `venn_C1p.txt`, **NOT** in `chronic_flippers.json` (exact-match check; legal-hard-22 is, legal-hard-2 is not) → strict accept rules.

**Config diff (validity gate):** exactly one behavioral knob — `max_operator_result_char_limit: 1000` vs `5000` (plus `system_name`). Same raw-probe prompt, delta mode, schema-only, oracle file list. Both arms fresh-run 2026-07-17 in the SAME star (exp `093430`, oracle_flag='oracle'), symmetric recovery: 1k = main + rec1 + rec2 (walked trace = rec2, 10:54, still FAIL, round log score 0.0 matches trace stats to the cent); 5k = main PASS at 11:00 (absent from its 11:32/12:02 recovery rounds).

**Vintage/determinism caveat (recorded up front):** a superseded same-config star ran earlier the same day (exp `014004`, also oracle). In it the 5k arm ALSO failed this task at least twice (present in its 05:26 AND 06:47 rerun-failed rounds; answers unrecoverable — scratch overwritten). The 1k arm failed at every observed checkpoint across both stars (≥5 confirmed attempts, 0 passes). Sibling texture, same star: Latest5k failed its main run then recovered to exactly **Miami** (14:27 round); DeltaStats1kD2 finished FAIL with **"Odessa, TX MSA"** — a different wrong answer at 1k render. So 5k-render arms converge to Miami on some attempt; 1k-render arms never produced Miami in any observed attempt, and two different 1k arms produced two different wrong answers.

## Gold semantic plan

Source: `solutions/legal/legal-hard-2.py`

| # | Plan item |
|---|---|
| G1 | Load all 52 state CSVs in `State MSA Identity Theft data/` with junk-row trimming (title/blank/footer), thousands-comma number cleaning → (name, # of Reports) rows |
| G2 | Parse `metropolitan_statistics.html`, take the table captioned "The 387 metropolitan statistical areas of the United States" (387×5) |
| G3 | Interpolate 2023 pop = `2020 census + (3/4)·(2024 estimate − 2020 census)` |
| G4 | Normalize names both sides: lowercase, strip `Metropolitan Statistical Area|MSA` suffix, strip punctuation, key = city+states |
| G5 | **`drop_duplicates()` both frames — full-row dedup.** The FTC data book lists every cross-state MSA once under EACH member state's file with an IDENTICAL total |
| G6 | Inner merge on the key (drops non-matches, incl. Micropolitan rows whose suffix never matches) |
| G7 | rate = reports / pop2023 × 100k; argmax → **Miami-Fort Lauderdale-West Palm Beach, FL** |

Data facts (verified with `.venv/bin/python` against the CSVs): 52 files → 452 named data rows → 401 unique names; **43 multi-copy (cross-state) MSAs, and every copy carries the IDENTICAL total** (`nunique==1` for all 43) — repeated totals, NOT per-state portions. `Philadelphia-Camden-Wilmington, PA-NJ-DE-MD` = 4 copies × 28,438 (Delaware/Maryland/NewJersey/Pennsylvania.csv); Miami = 1 copy, 55,457. Answer arithmetic, exact repro of the loser's own pipeline: groupby-name-**SUM** → Philadelphia 113,752 / 6,309,079.25 × 100k = **1802.9889** (its rendered value, to the digit); replace the sum with `drop_duplicates` in the same pipeline → **Miami 869.4944** = the winner's exact rendered rate. One decision decides the task. The sum-pipeline's top-3 are ALL multiplied cross-state names — Philadelphia 1803 (4×), Washington DC-VA-MD-WV 1231 (4×), Memphis TN-MS-AR 1231 (3×) — with true-#1 Miami at rank 4; 9 of its top-12 are multi-copy artifacts.

## Walk: DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt (WINNER — PASS)

**Final answer:** Miami-Fort Lauderdale-West Palm Beach, FL MSA — correct. 10 agent steps, 12 ops, 152,402 in / 136,960 cached, $0.0924, 70.0s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_msa_html`, `raw_idtheft_files` | raw-line probe of html head+tail; glob listing of the 52 CSVs | probes (G1/G2 recon) |
| 1 | `raw_idtheft_sample`, `msa_pop_tables_preview`, `msa_pop_table_extract` | raw lines of CA/FL/NY CSVs (75 rows); BeautifulSoup table survey (finds caption "The 387 metropolitan statistical areas…", 388 tr); `read_html` biggest table → 387×5 | G2 ✓ |
| 2 | `idtheft_all` | glob+concat all 52, `dtype=str`, `source_file` col → 764×3 (junk in-band) | G1 ✓ (clean deferred) |
| 3 | `idtheft_clean` | junk blacklist (`'Metropolitan Area'` header rows, `Source:|Metropolitan Areas are defined`), de-comma → int, keep names containing `Metropolitan Statistical Area|MSA` → 417×2 | G1 ✓; Micropolitan-drop ≡ G6's join-drop; **no dedup and NO aggregation** (argmax-safe; see verdict) |
| 4 | `msa_pop_2023` | `pop_2020 + (pop_2024−pop_2020)·3/4` → 387×4 | G3 ✓ (gold's exact formula) |
| 5–6 | `msa_name_keyed`, `idtheft_name_keyed` | en/em-dash→hyphen, strip MSA/Metropolitan/Micropolitan/µSA suffixes, city + first state token, lowercase, strip punct | G4 ✓ |
| 7 | `idtheft_pop_join` | inner merge on `msa_key` → 410×6; rate = reports/pop2023×100k | G6/G7 ✓ (dup rows survive as identical-rate copies) |
| 8 | `top_rate_msa` | sort desc, head(1) → 1×2 = Miami 869.494 | G7 ✓ |
| 9 | (text) | Final Answer: Miami-Fort Lauderdale-West Palm Beach, FL MSA | PASS |

**First semantic divergence:** none load-bearing. It skips G5 (no dedup) but never introduces a cross-file aggregation, so the duplicated cross-state rows survive only as identical-rate copies and the argmax is invariant (exact repro: 417 → 410 joined → Miami).

**Rendered evidence at the clean-op decision (step-3 input, `idtheft_all` DELTA block, 2,241 chars — fits whole in the 5k budget):** 15 head + 15 tail of 764 rows on screen, including two cross-state areas filed under a member state and the per-state junk skeleton:

```
5   Columbus, GA-AL Metropolitan Statistical Area    1,302  Alabama.csv
12  LaGrange, GA-AL Micropolitan Statistical Area    453    Alabama.csv
...
754 Metropolitan Areas are defined by the Office of Management and Budget...  Wisconsin.csv
756 Source: Consumer Sentinel Network Data Book 2024...   Wisconsin.csv
758 Metropolitan Area   # of Reports   Wyoming.csv
```

Its clean-op blacklist and Micropolitan filter are read literally off this window. Still in DELTA history from step 2: the 39-rendered-row raw sample (3,210 chars) with `"New York-Newark-Jersey City, NY-NJ Metropolitan Statistical Area","67,819"` inside **NewYork.csv** — a cross-state MSA carrying a full metro-scale total under one member state's file. At the terminal step its argmax table rendered exactly `Miami-Fort Lauderdale-West Palm Beach, FL … 869.494`.

## Walk: DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt (LOSER — FAIL)

**Final answer:** Philadelphia–Camden–Wilmington, PA-NJ-DE-MD MSA — the 4×-double-counted artifact. 7 agent steps, 8 ops, 62,433 in / 55,424 cached, $0.0520, 49.0s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_msa_html`, `raw_msa_csv` | html raw lines; head-preview (5 lines) of first 8 CSVs | probes |
| 1 | `msa_pop_html`, `identity_theft_raw` | `read_html` pick by colname heuristic → 387×5; glob+concat 52 CSVs with `skiprows=2` → 660×3 | G2 ✓, G1 ✓ |
| 2 | `msa_pop_2023` | same 3/4 interpolation → 387×4 | G3 ✓ |
| 3 | `identity_theft_clean` | dropna, drop `Source:` rows, de-comma → numeric, then **`groupby('Metropolitan Area').sum()`** → 401×2 | **DIVERGES at G5: resolves the multi-file repetition by SUMMING the identical repeated totals** (portions hypothesis) instead of dedup — Philadelphia 28,438×4 → 113,752 |
| 4 | `msa_name_xwalk` | NFKD + dash fix + suffix strips + punct strip; inner merge → 361×4 | G4/G6 ✓ |
| 5 | `msa_identity_theft_rate` | rate = reports/pop2023×100k, sort desc → 361×5, top = Philadelphia 1802.99 | G7 executed on inflated counts |
| 6 | (text) | Final Answer: Philadelphia–Camden–Wilmington, PA-NJ-DE-MD MSA | FAIL |

**First semantic divergence: step 3, op `identity_theft_clean`, plan item G5** — the one line `df.groupby('Metropolitan Area', as_index=False)['reports'].sum()`. Everything else in its pipeline is gold-equivalent (its dedup-instead-of-sum counterfactual returns Miami 869.494 exactly).

**Rendered evidence — the absence, quoted:** at the step-3 decision its `identity_theft_raw` block was 537 chars — **3 head + 2 tail rows of 660**:

```
0  Anniston-Oxford, AL Metropolitan Statistical Area   264    Alabama.csv
1  Auburn-Opelika, AL Metropolitan Statistical Area    451    Alabama.csv
2  Birmingham, AL Metropolitan Statistical Area        3,968  Alabama.csv
...
658  NaN  NaN  Wyoming.csv
659  Source: Consumer Sentinel Network Data Book 2024…  Wyoming.csv
```

All three visible data rows are single-state AL areas. **The first cross-state row, `Columbus, GA-AL … 1,302 … Alabama.csv`, sits at row 3 — one row below the 1k window cut** (in the winner's uncleaned 764-row equivalent it sat at row 5, inside the 15-row window). Its step-0 file probe rendered 2 of 8 rows with ~100-char previews (Alabama "Anniston-Oxford, AL Me…", Delaware "Dover, DE Metropolitan…") — both same-state rows; Delaware.csv's Philadelphia line was in the un-rendered part. **No render this arm ever saw contained a cross-state name under a member-state file, a repeated name, or a repeated total.** At the terminal decision (step-6 input) its sorted rate table rendered 1 head + 2 tail rows: Philadelphia 1802.99 / reports 113,752 on top and rates 85/79 at the bottom — the diagnostic pattern at ranks 1–4 (three multiplied multi-state names — PA-NJ-DE-MD, DC-VA-MD-WV, TN-MS-AR — then Miami 869) started at row 1, again below the fold. The winner's terminal-table equivalents rendered 18 rows.

## Pair verdict

**C1p Delta5k > Delta1k: ATTRIBUTED (render-starvation shaped), with an explicit determinism caveat.** One knob (1k vs 5k render). The divergence is a single semantic decision — sum-vs-dedup on the FTC data book's repeated cross-state totals, the same duplicate mechanism as the levers report's attributed legal-hard-15 (same dataset family, same char-limit lever). The winner's pipeline is explained by its screen: its junk blacklist and Micropolitan filter quote its 30-row window, and having cross-state areas visible under member-state files (`Columbus, GA-AL` under Alabama.csv at the decision step; `New York-Newark-Jersey City, NY-NJ … 67,819` under NewYork.csv one step earlier) it treated rows as what they visibly are — per-file (MSA, total) listings — and never invented a cross-file aggregation. The loser's error is explained by absence, and the absence is structural, not stochastic: at 1k the 660-row table renders 3+2 rows and the nearest disconfirming row was literally one row below the cut, at BOTH hinge points (raw table at step 3; sorted leaderboard at step 6, where ranks 1–3 all being multiplied hyphen-state names was hidden while only rank-0 rendered). Starved, it guessed portions-to-sum semantics — a policy for a repetition it never saw — and nothing later rendered could refute the guess. Supporting evidence in lieu of same-arm answer diversity: the loser failed every observed attempt (≥5 across two same-day same-config stars — consistently wrong, i.e. the signal is deterministically outside its window), a sibling 1k arm (Stats1kD2) failed with a DIFFERENT wrong answer (Odessa, TX — 1k arms scatter), and both 5k-render arms (Delta5k, Latest5k) converged to Miami. **Caveats, per strict rules:** (a) neither arm ever rendered a literal duplicate PAIR, so the winner's screen shows repetition is coming, not that copies are identical totals — the winner's correctness runs through "don't invent an aggregation you can't see" rather than a positively evidenced dedup (weaker than legal-hard-15's C2, where evidence and action shared a step); (b) the 5k arm failed this task twice in the superseded morning star before passing in the walked one (answers unrecoverable), so the lever raises the hit rate rather than guaranteeing the pass — the attribution is that the 1k window structurally hides the decision's evidence, not that the 5k window mechanically forces the right choice.

Artifacts: `system_scratch/{DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt,DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt}/legal-hard-2/`; repro via `.venv/bin/python` against `data/legal/input/`; rerun record from `logs/kb-rerunfail-*ProbePrompt-*/legal-hard-2.log` and `logs/exp-probestar{,2}-nohup.log`.
