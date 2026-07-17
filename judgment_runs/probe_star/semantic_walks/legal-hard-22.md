# legal-hard-22 — semantic walk (PROBE-STAR vintage)

## Task + gold

Q: "What is the proportion (round to 4 decimal places) of all reports who reported
identity theft with Back Account (Theft Type) and New Accounts (Theft Subtype)?"
— numeric_exact, gold answer **0.0555**. (Question typo "Back Account" = Bank Account;
no arm stumbled on it.)

Data (`data/legal/input/`, FTC CSN Data Book 2024, block-structured multi-table CSVs):

- `2024_CSN_Report_Type.csv` — title line, blank line, then the **Report Type block**
  (`Fraud,"2,600,678",40.2% of total reports` / `Identity Theft,"1,135,291",17.5% of
  total reports` / `Other,"2,759,963",42.6% of total reports`), followed by unrelated
  sub-tables in the same file (Top 10 Categories, Identity Theft Types, …). A naive
  `header=2` load yields a 43x6 frame where `# of Reports` is numeric only for the
  first block's 3 rows (all other blocks put non-numerics in that position).
- `2024_CSN_Identity_Theft_Reports_by_Type.csv` — one table (Theft Type / Theft
  Subtype / # of Reports / % Difference), quoted thousands-separators, **cp1252**
  (0x96 en-dashes → UnicodeDecodeError under UTF-8).

Arms (all PROBE-STAR raw-probe prompt, code mode; one-knob parity confirmed from
`config.json`: C1p `max_operator_result_char_limit` 5000↔1000; C2p `column_stats`
on↔off; C3p `context_mode` delta↔latest):

| arm | knob values | answer | pass |
|---|---|---|---|
| Delta5kSchemaOnlyProbePrompt (WINNER C1p, C3p) | delta, 5k, stats=off | **0.0555** | ✓ |
| DeltaStats1kD2ProbePrompt (WINNER C2p) | delta, 1k, stats=on | **0.0555** | ✓ |
| Delta1kSchemaOnlyProbePrompt (loser C1p, C2p) | delta, 1k, stats=off | 0.0097 | ✗ |
| Latest5kSchemaOnlyProbePrompt (loser C3p) | latest, 5k, stats=off | 0.0097 | ✗ |

Chronic tag: **legal-hard-22 IS in the OLD-vintage `chronic_flippers.json`**
(advisory) — default verdict CHRONIC/VARIANCE unless the accept rules are met.

## Gold semantic plan

From `solutions/legal/legal-hard-22.py` (line-block parser; net semantics):

1. `2024_CSN_Report_Type.csv`, Report Type block → the **Identity Theft row's
   `# of Reports` = 1,135,291** (gold's `sub_table1.values[0][1]`; its parser consumes
   the Fraud row as the columns row, so values[0] IS the Identity Theft row). The
   denominator is the identity-theft report population — **not** the all-reports total.
2. `2024_CSN_Identity_Theft_Reports_by_Type.csv` → row (Bank Account, New Accounts)
   `# of Reports` = **62,982** (gold's `sub_table2.values[1][2]`).
3. Answer = 62982 / 1135291 = 0.055476 → **0.0555**.

The single load-bearing semantic choice is item 1: the **denominator**. "proportion of
all reports who reported identity theft with X" is ambiguous English — gold reads it as
"proportion of [reports that reported identity theft]" (÷1,135,291), the rival literal
reading is "proportion of [all reports]" (÷ 2,600,678+1,135,291+2,759,963 = 6,495,932
→ 0.0097).

## Walks (all four arms, aligned)

All four arms run the SAME dataflow shape — the probe prompt works identically
everywhere:

- **step 0**: raw head/mid/tail line-probe of both files (2 ops). All four see the
  Report Type block header + `Fraud` + `Identity Theft` rows as raw lines.
- **step 1**: pandas load both files, `header=2`/`skiprows=2`. delta5k, stats1k,
  delta1k hit `UnicodeDecodeError: 0x96` on file2 → the rendered loader hint
  ("try encoding=\"cp1252\" first") fixes it in exactly one step. latest5k pre-empted
  with `encoding_errors='replace'` and never errored.
- **numerator op** (step 2–4 depending on arm): filter `Theft Type=='Bank Account' &
  Theft Subtype=='New Accounts'` → **62,982 in all four arms** (gold item 2 ✓
  everywhere; comma-stripping correct everywhere).
- **denominator op = THE HINGE** (first and only semantic divergence, per arm):

| arm | step | op | code semantics | gold item 1 |
|---|---|---|---|---|
| delta5k | 3 | `all_reports_total` | `df.loc[df['Report Type'].str.lower().eq('identity theft'), '# of Reports'].sum()` → 1,135,291 | ✓ |
| stats1k | 3 | `identity_theft_total` | filter `== 'identity theft'`, take `iloc[0]` → 1,135,291 | ✓ |
| delta1k | 4 | `report_total` | **sum the entire `# of Reports` column** (digit-extract drops junk-block rows) → 6,495,932 | ✗ |
| latest5k | 3 | `total_reports` | **"sum across types to compute total reports"** → 6,495,932 | ✗ |

- **final op**: numerator/denominator. Winners → 0.0555; losers → 0.009696 → **both
  losers submit the byte-identical wrong answer 0.0097**. Losers' code is clean and
  their sum lands exactly on Fraud+IdT+Other — only the question interpretation
  differs from gold.

## Evidence at decision time (the acceptance test)

What was RENDERED in each arm's context at its denominator step:

| arm | IdT row (`1,135,291` + `17.5% of total reports`) | `Other 2,759,963` row | `6,495,932` anywhere | stats block | choice |
|---|---|---|---|---|---|
| delta5k (W) | visible — load obs rendered all 3 block rows | visible (load obs) | never rendered | none | filter ✓ |
| stats1k (W) | visible — probe line + 2-row truncated load obs | **never visible** (probe head-5 stops at IdT; load obs truncated after row 1) | never | `"Report Type" (str): null=5, distinct=19, duplicate_values=19` + `empty rows: 5 of 43` — junk-structure warnings, **no denominator fact** | filter ✓ |
| delta1k (L) | visible — probe + 2-row load obs | **visible** — raw probe line `5  Other,"2,759,963",42.6% of total reports,,,` | never | none | sum ✗ |
| latest5k (L) | visible — load obs rendered all 3 block rows (`0 Fraud… | 1 Identity Theft… | 2 Other 2,759,963 42.6%…`) | visible (load obs) | never | none | sum ✗ |

Reading of the table: **evidence visibility is fully uncorrelated with the choice.**
The decisive fact (the Identity Theft row with its count) was rendered in ALL four
contexts; no context contained the losers' 6,495,932. The leanest-evidence arm
(stats1k — never even saw the `Other` row or 42.6%) chose the gold denominator; the
richest (latest5k — 3-row render identical to winner delta5k's) summed; delta1k had
MORE denominator-relevant rows visible than winner stats1k and still summed. Both
accept conditions fail in all three pairs: the winners' extra evidence does not explain
their action, and there is no loser-side absence to explain the error.

## Mechanism scans

- Identical-probe repetition: none (no resubmissions in any arm; 6–8 steps each).
- Churn flag: no (5–7 ops, single sink, no re-edit loops).
- `[ERROR` in rendered context: exactly one per delta arm (the file2 UTF-8 decode),
  recovered in one step via the loader hint, upstream of and orthogonal to the hinge;
  latest5k had zero errors and still lost — errors anti-correlate with outcome here.
- Failure class: wrong-answer (clean execution, wrong denominator semantics); no
  timeout / gave-up / format issues.

## Verdicts

- **C1p (delta 5k > 1k): CHRONIC/VARIANCE.** Hinge = delta1k step 4 `report_total`
  (column sum) vs delta5k step 3 filter. The budget lever did not change the visible
  decisive facts — delta1k saw the IdT row AND the Other row (raw probe); nothing the
  5k render added bears on the denominator reading.
- **C2p (stats > schema @1k): CHRONIC/VARIANCE.** The stats block rendered only
  junk-structure warnings (distinct=19, 5 empty rows) — no denominator fact; the stats
  winner decided with strictly LESS row evidence than the schema-only loser.
- **C3p (delta > latest @5k): CHRONIC/VARIANCE.** Both contexts rendered the identical
  3-row Report Type block; opposite choices.

**One coin, not three mechanisms — but a coin nonetheless.** Both losers share a single
failure: the literal "all reports" reading (÷6,495,932 → byte-identical 0.0097), a
question-interpretation fork on ambiguous English (same family as archeology-hard-7's
L2-vs-L∞). The choice splits 2–2 across four arms, uncorrelated with budget, stats, or
context mode, on a task already in the old chronic set. What looked like dual-lever
convergence (1k loses to both 5k and stats) is two of the star's cells reusing the same
single delta1k tails — do not build a lever story on any of these three cells.
