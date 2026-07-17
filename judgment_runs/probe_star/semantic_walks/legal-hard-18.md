# Semantic walk: legal-hard-18 (PROBE-STAR vintage, C3p A-only pair)

## Task + gold answer

**Question:** If the 2007 reports were distributed exactly like the 2024 ones, how many identity theft reports in 2007 would concern people ages 40 or older (rounded to the nearest thousand)?

**Gold answer:** `91000` (numeric_exact). NOT on `chronic_flippers.json` → strict accept rules.

**Pair (C3p, delta>latest @5k schema-only):** winner `DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt` (91000, PASS) vs loser `DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt` (520000, FAIL).

**Config diff (validity gate):** exactly one knob — `context_mode: delta` vs `latest`; both `column_stats:false, data_level:1`, 5000/3000 char limits, probe prompt. PASSES.

## Gold semantic plan

Source: `solutions/legal/legal-hard-18.py` (dirty CSN docket CSVs: title+blank preamble rows, comma-formatted numbers, `%` strings, footnote rows, non-UTF8 categories file).

| # | Plan item |
|---|---|
| G1 | Load `2024_CSN_Report_Count.csv` `skiprows=2`, dropna, strip commas → int |
| G2 | `total_2007` = reports where Year==2007 → **1,070,447** |
| G3 | Load `2024_CSN_Report_Categories.csv` `skiprows=2, encoding=unicode_escape`, dropna → Identity Theft `Percentage` → **0.1754** |
| G4 | Load `2024_CSN_Identity_Theft_Reports_by_Age.csv` `skiprows=2`, dropna, strip commas → per-bracket share of total |
| G5 | Sum shares for brackets 40+ (`40-49 … 80 and Over`) → **0.4861** |
| G6 | `round(total_2007 × id_theft_pct × share_40plus, -3)` = round(91,275) → **91000** |

Three files, three multiplicative factors. The trap is dropping one.

## Walk — Delta 5k schema-only (WINNER, 91000)

| step | action | semantics | gold item |
|---|---|---|---|
| 0 | 3 probes | raw head/mid/tail of ALL THREE files | recon G1/G3/G4 |
| 1 | 3 loaders + delete 3 probes | `report_count`, `age_2024`, `categories_2024`, all `skiprows=2` + comma-strip | G1, G3, G4 ✓ |
| 1→2 | obs | count ✓ 24x2, age ✓ 8x2; **categories UnicodeDecodeError** + loader hint "try encoding=cp1252" | G3 blocked |
| 2 | edit `categories_2024` | reload `encoding='cp1252'` → renders `Identity Theft 1135291 17.54` | G3 ✓ (recovered) |
| 3 | `age40plus_share_2024` | isin(40-49…80 and Over) sum / total → `0.4860709158093` | G5 ✓ |
| 4 | `identity_theft_2024_total` | Category=='Identity Theft' → `1135291` | G3 ✓ |
| 5 | final op | `1070447 × (1135291/6471708=0.17542370576670022) × 0.4860709158093 = 91275.26 → 91000` | G6 ✓ |
| 6 | answer | **91000** | PASS |

No semantic divergence from gold (count-ratio IT share instead of the `%` column is algebraically the gold 0.1754).

## Walk — Latest 5k schema-only (LOSER, 520000)

| step | action | semantics | gold item |
|---|---|---|---|
| 0 | 3 probes | raw head/mid/tail of ALL THREE files (incl. `raw_categories`) | recon ✓ |
| **1** | **2 loaders** + delete 3 probes | `age_2024`, `counts_by_year_raw` — **NO categories loader ever created** | **← diverges: G3 dropped** |
| 2 | `age_2024_clean` | drop footnotes, numeric | G4 ✓ |
| 3 | `reports_2007` | Year==2007 → `1070447` | G1/G2 ✓ |
| 4 | `age40plus_share_2024` | 40+ share → `0.4860709158093` | G5 ✓ |
| **5** | final op | `est = 1070447 × 0.4860709 = 520313 → 520000` — **two-factor product, IT share missing** | **G6 short one factor** |
| 6 | answer | **520000** | FAIL |

First semantic divergence: **step 1** — the plan omits G3; step 5 merely consummates it (its own Summary says "estimate 2007 identity theft reports age 40+ as total_2007 * share_40plus", i.e. it treats ALL 2007 reports as identity theft).

## Evidence at the decision — parity at the fork

**At the fork (step 1 input), the two arms' rendered evidence is semantically IDENTICAL.** Both current renders contain each arm's own `raw_categories` probe with the load-bearing line verbatim:

- Loser (Latest current-DAG block, step-1 input): `### Operator `raw_categories`` … `4  2,Identity Theft,"1,135,291",17.54%`
- Winner (Delta event log, step-1 input): `- operator raw_categories added` … `4  2,Identity Theft,"1,135,291",17.54%`

Nothing in the Delta winner's rendered history was absent from Latest's current-DAG render at the decisive step — the probe was still a live operator in both. The loser saw the Identity Theft share on screen and created loaders for only two of the three files it had just probed ("to verify header/format **before loading**", per its own probe summary — a dropped plan hop, not an evidence gap).

**After the fork the renders DO split** (both arms deleted the probes at step 1): from step 2 on, Latest's current-DAG render contains zero occurrences of `Identity Theft`/`17.54`/the categories table, while Delta's history retains them through the answer step (6 × `Identity Theft`, 2 × `17.54` in the winner's final input). But this erasure post-dates the divergence — and even the loser's final-compute context still held the raw makings of the missing factor: the age-file footnote `Of the 1,135,291 total identity theft reports in 2024…` and `2024  6,471,708` in `counts_by_year_raw` — exactly the two numbers the winner's final op divides (0.175423…). The loser had sufficient rendered evidence to catch the omission at step 5 and did not.

**Mechanism scans:** no probe resubmission (deleted once, never recreated, either arm); no churn (5–6 final ops, sink-share low); errors — winner 1 real error (categories UnicodeDecodeError, fixed in one edit via the loader hint; re-rendered 5× in delta history), loser 0. Failure class: wrong-answer via missing plan hop. Cost aside: winner $0.0668 vs loser $0.0398 — the passing arm paid ~68% more (extra file + encoding recovery), same react step count.

## Verdict

**REJECTED — method-choice (plan-hop omission at evidence parity).** The flip is real and the failure is fully explained (520000 = 1,070,447 × 0.4861: the G3 identity-theft factor was never loaded), but it cannot be attributed to `context_mode`: at the divergence step both arms rendered the identical `Identity Theft … 17.54%` probe line, and the loser's error was committed with that evidence present. Latest-mode's probe-deletion erasure (no categories trace rendered after step 2) removed the strongest recovery cue and is worth logging as a latest-mode liability motif — but the divergence predates the arms' first semantic rendered difference, and residual evidence for self-correction (footnote 1,135,291 + total 6,471,708) was still rendered in the loser's final input and went unused. Strict rules → not an attributed lever flip.

Artifacts: `system_scratch/{DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt,DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt}/legal-hard-18/`; gold `solutions/legal/legal-hard-18.py`.
