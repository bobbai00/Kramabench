# legal-easy-9 — semantic walk (PROBE-STAR vintage)

## Task + gold

Q: "Between 2002 and 2024 inclusive, which year saw the greatest relative increase in
total reports (Fraud, Identity Theft and Other) compared to the previous year?" —
four-digit year, gold answer **2002**.
Data: `2024_CSN_Report_Count.csv` (31 raw lines: title + blank + `Year,# of Reports`
header, then 2001–2024 with quoted-comma counts, then 4 footer lines — blank / DNC-note
/ blank / source). Dirty-profile facts: footer rows load as NaN/str junk (28 rows,
`Year (str)`), counts need `thousands=','`.

Arms (all PROBE-STAR raw-probe prompt, DELTA context; one-knob diffs confirmed from
`config.json` — C1p pair `max_operator_result_char_limit` 1000→5000 only; C2p pair the
stats bundle `column_stats` off→on + `data_level` 1→2):

| arm | knob values | answer | pass | steps | cost_usd |
|---|---|---|---|---|---|
| Delta5kSchemaOnlyProbePrompt (WINNER C1p) | 5k chars, stats=off, D1 | **2002** | ✓ | 5 | 0.0231 |
| DeltaStats1kD2ProbePrompt (WINNER C2p) | 1k chars, stats=on, D2 | **2002** | ✓ | 6 | 0.0226 |
| Delta1kSchemaOnlyProbePrompt (loser both pairs) | 1k chars, stats=off, D1 | 2020 | ✗ | 6 | 0.0202 |

Chronic tag: **NOT on the old-vintage chronic list** (legal chronics: legal-easy-19,
legal-hard-1, legal-hard-22) — accept rules applied strictly.

## Gold semantic plan

From `solutions/legal/legal-easy-9.py`:

1. Load `2024_CSN_Report_Count.csv`, `skiprows=2` (title+blank), header `Year,# of Reports`.
2. Clean: `dropna()` (kills 4 footer rows) → `Year` → int → `# of Reports` strip commas → int.
3. `diff` = YoY difference, `rel_diff = diff / shift(1)` — computed over the **FULL
   2001–2024 series**: 2001 is the base row that gives 2002 its +69.46%.
4. Answer = `Year` at `rel_diff.idxmax()` → **2002** (0.6946; runner-up 2020 at 0.4818).

The single load-bearing choice is item 3's **order**: restrict-to-[2002,2024] must
happen AFTER the shift. Filter-first makes 2002's prev NaN → silently drops it →
max flips to 2020.

## Walk: C1p — Delta5kSchemaOnlyProbePrompt (PASS)

| step | action | semantics | vs gold |
|---|---|---|---|
| 0 | +`raw_csn` | probe raw lines 0–7 + 13–17 (head + middle) | recon |
| 1 | +`csn_reports` +`relative_increase` −`raw_csn` | load `skiprows=2, thousands=','` but `dtype={'Year':'int64'}` (footer NaN bug); compute = `sort_values('Year')` → `pct_change()` → **then** `[(Year>=2002)&(Year<=2024)]` → `idxmax` | G1 partial (dtype bug); **G3+G4 correct at first write** |
| 2 | ~`csn_reports` | drop dtype; `to_numeric(coerce)` + `dropna` + `astype('int64')` → 24 rows 2001–2024 | G1+G2 ✓ |
| 3 | Final Answer 2002 | rendered `2002 0.694592…` | ✓ |

One transient `[ERROR] … ValueError: cannot convert float NaN to integer`, self-fixed
next step. **No semantic divergence.**

## Walk: C2p — DeltaStats1kD2ProbePrompt (PASS)

| step | action | semantics | vs gold |
|---|---|---|---|
| 0 | +`raw_csn_preview` | probe first 60 → all 31 lines | recon |
| 1 | +`csn_reports` | `skiprows=2`, rename, dropna Year, strip `,`/`"`, dropna both, sort → 24 rows 2001–2024 | G1+G2 ✓ |
| 2 | +`csn_yoy_relative_increase` | `sort` → `shift(1)` → `(cur−prev)/prev` → **then** filter [2002,2024] | **G3+G4 ✓** |
| 3 | +`csn_max_relative_increase_year` | max → 2002 | G4 ✓ |
| 4 | Final Answer 2002 | | ✓ |

**No semantic divergence.**

## Walk: loser — Delta1kSchemaOnlyProbePrompt (FAIL, 2020)

| step | action | semantics | vs gold |
|---|---|---|---|
| 0 | +`raw_preview` | probe first 20 lines | recon |
| 1 | +`reports` | `skiprows=2, thousands=','` → 28 rows incl. footer | G1 ✓ |
| 2 | +`reports_clean` | Year coerce/notna/int; **`df = df[(df['Year'] >= 2002) & (df['Year'] <= 2024)]` BEFORE any diff** → 23 rows starting 2002 | **← DIVERGENCE: violates G3 order — evicts the 2001 base row** |
| 3 | +`relative_increase` | `shift(1)` inside the pre-filtered frame; `df[df['prev_reports'].notna()]` silently drops 2002; `idxmax` → 2020 (0.4818) | compounds step 2 |
| 4 | Final Answer 2020 | | ✗ |

Failure-mode class: silent wrong-answer (no errors, no probe repetition, no churn —
4-op DAG, 6 steps).

## Evidence at decision time (the attribution test)

**Loser, at the hinge (writing `reports_clean`, agent step 4 input):** its rendered
context contained the base-year fact TWICE —

> ```
> 3	2001,"325,519"
> 4	2002,"551,622"
> ```
(raw_preview render, rows 0–9 + 11–19 of 20 shown; 1k cap elided only row 10) and

> ```
> 	Year	# of Reports
> 0	2001	325519
> 1	2002	551622
> ```
(reports 28x2 render, rows 0–7 + 23–27 shown — head years AND footer junk both visible).

**C1p, when it wrote the correct compute (agent step 2 input):** ONLY the probe render —
> `3	3	2001,"325,519"` — i.e. **less** rendered evidence than the loser had at its hinge.

**C2p, when it wrote the correct compute (agent step 3 input):** probe (all 31 lines,
head+tail) + `csn_reports` 24x2 render (`0 2001 325519` first row) + stats block
> `- "Year" (numeric): null=0, mean=2012, min=2001, max=2024`
— the stats lever does deliver an explicit `min=2001`, but it only duplicates what the
loser's own renders already displayed verbatim.

Both winners' op **summaries** even phrase the plan loser-style — C1p: "filter years
2002–2024, compute year-over-year relative increase…"; C2p: "filter to years 2002–2024
and compute year-over-year relative increase… (pct_change)" — while their **code** does
shift-then-filter. Prose plan identical across all three arms; only implementation
order differs.

## Mechanism scans + fleet coin-flip evidence

- No identical-probe repetition, no `[ERROR` in loser trace, no churn flag (all DAGs
  3–4 ops). Costs statistically identical (loser cheapest at $0.0202).
- **Fleet sweep (106 scratch arms): 14 answer 2020, 92 answer 2002.** The 2020 branch
  appears at every information level: `DeltaStats3kD2Explore` (stats=on, 3k, D2 — both
  levers ABOVE the loser) fails with the **byte-identical mechanism**
  (`out = out[(out['year'] >= 2002) & (out['year'] <= 2024)]` upstream of a
  shift-then-filter final op), as do `DeltaWin6kCompressLean`,
  `LatestSchemaConvergeLineage`, 6 GPT54 arms, GPT5Mini Rows3/Rows5, Haiku45Annot2.
- The loser's identical-config-except-prompt sibling `Delta1kSchemaOnly` (old prompt
  vintage) **passes** with 2002. Within GPT52 dataflow arms the wrong branch fires
  4/53 (~8%) — a coin-flip base rate uncorrelated with either lever.

## Verdict

**REJECTED-method-choice — BOTH pairs (C1p 5k>1k and C2p stats>schema@1k). Not
dual-lever convergence; the dual-winner structure is coincidental variance.**

Accept-rule test fails on the loser side: the winners' evidence is consistent with
their action, but the loser's error is NOT explained by any evidence absence — the
2001 base row was rendered twice in its context at the divergence step, and C1p made
the correct ordering choice with strictly LESS rendered evidence than the loser had.
The extra facts the levers deliver (5k: longer table middles; stats: `min=2001` line)
are redundant with what the loser already saw; and arms holding both levers higher
(`DeltaStats3kD2Explore`) commit the same filter-before-shift error. The divergence is
a pure implementation-order coin flip on a shared, identically-phrased plan.
legal-easy-9 is not on the old chronic list but exhibits the classic flip-prone
profile this vintage — recommend tagging it as a new-vintage variance candidate rather
than crediting either lever.
