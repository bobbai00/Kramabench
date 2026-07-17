# Semantic walk: legal-easy-19

## Task + gold answer

**Question:** What is the proportion (round to 3 decimal places) of fraud reporters who lost between $1-$500 in 2024?

**Gold answer:** 0.523 (numeric_exact).

**Judge marks:** Delta1kSchemaOnlyProbePrompt `success=1` (0.523, PASS), Delta5kSchemaOnlyProbePrompt `success=0` (0.827, FAIL), DeltaStats1kD2ProbePrompt `success=0` (0.199, FAIL), Latest5kSchemaOnlyProbePrompt `success=1` (0.523, PASS). Task **IS** in old-vintage `chronic_flippers.json` (advisory for this vintage) and is `*`-starred in all three probe-star venns. Pairs: C1p A-only (A=1k, B=5k), C2p A-only (A=1k schema, B=stats1k D2), C3p B-only (A=delta5k, B=latest5k). This task sits in THREE exclusive cells this vintage.

**Config diffs (validity gate):** C1p — exactly one knob, `max_operator_result_char_limit` 1000 vs 5000 (both delta, schema-only, D1). C2p — the stats lever bundle `column_stats: true` + `data_level: 2` vs `false`/`1` (both 1k delta). C3p — exactly one knob, `context_mode` latest vs delta (both 5k schema-only D1). All other agent_settings byte-identical (probe-star vintage: `static_compaction: false`, frontier/fold/retirement null, `tool_dialect: qwen-xml`).

**Vintage:** all four final traces are same-day 2026-07-17 (1k 10:33, 5k 12:20, stats1k 13:44, latest5k 15:22); the venns were regenerated 15:27, after the last trace — the venn verdicts are computed on exactly these traces. Recovery symmetry: every arm's final state received the standard 2 `rerun-failed` rounds (1k cleared after round 1 so round 2 excluded it). The 1k and 5k arms additionally have a discarded earlier same-day run + its own 2 rounds; those recorded answers are used below as rerun-instability evidence. (Extraction note: the first walk extraction raced the live batch mid-rewrite; re-extracted after venn generation.)

## Gold semantic plan

Source: `solutions/legal/legal-easy-19.py`

| # | Plan item |
|---|---|
| G1 | Custom-parse `data/legal/input/csn-data-book-2024-csv/CSVs/2024_CSN_Fraud_Reports_by_Amount_Lost.csv` — a 37-line multi-block CSV (strip commas-in-quotes, blank-line block separation) → 4 scalar attributes + 2 sub-tables |
| G2 | **Denominator = scalar attribute `Reports with $ Loss` = 987,520** (the with-$-loss population) — NOT `Number of Fraud Reports` = 2,600,678, NOT any bucket total |
| G3 | Numerator: from sub-table `Reported Fraud Losses in $1 - $1,000 Range`, sum the five buckets `$1 - $100`(243,174) + `$101 - $200`(114,336) + `$201 - $300`(67,064) + `$301 - $400`(44,982) + `$401 - $500`(46,752) = **516,308** |
| G4 | 516308 / 987520 = 0.522833 → round 3 dp → **0.523** |

Data facts (verified against the file): the file co-locates THREE "total"-candidates within 8 lines — `Number of Fraud Reports,"2,600,678"`, `Reports with $ Loss,"987,520",38% of the total` (the file itself annotates the 38% relationship: 987520/2600678 = 0.3797), and the `$1 - $1,000` bucket of the first sub-table (624,110). Identity: the 11 buckets of the `$1 - $10,000 +` sub-table sum EXACTLY to 987,520 — the with-loss population is closed under that table. Answer↔denominator map (numerator 516,308 in all cases): **0.523** = /987520 (gold), **0.827** = /624110 ($1-$1,000 sub-table total), **0.199** = /2600678 (all fraud reports), 0.632 = 624110/987520 (right denominator, $1-$1,000-bucket numerator). Every wrong answer observed today is one of these attractors. The dirty-CSV structure itself (multi-block, dup/empty/unnamed per prior profiles) was NOT the failure mode: all four arms parsed the blocks correctly and produced the gold numerator 516,308.

## Walk: DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt (WINNER C1p+C2p — PASS)

**Final answer:** 0.523. 5 agent steps (6 counted), 38,232 input tokens, $0.0265.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_preview` (probe) | read first 6 raw lines | G1 recon ✓ |
| 1 | `raw_preview_mid` (probe) | read head 10 + mid 10 + tail 10 raw lines (30 of 37) | G1 recon ✓ |
| 2 | `fraud_amount_lost` + delete both probes | csv.reader; keep rows with `$` in col0 AND numeric col1 → 24x2 (3 `$`-scalars + all 21 bucket rows; the `$`-filter drops `Number of Fraud Reports`) | G1 ✓ (probe-beat: full retire) |
| 3 | `loss_1_500_proportion` | total = row `Reports with $ Loss` (987,520); num = the five ≤$500 buckets (516,308); round(num/total,3) | **G2 ✓ G3 ✓ G4 ✓** |
| 4 | (text) | Final Answer: 0.523 | ✓ |

**First semantic divergence:** none.

**Rendered evidence at the denominator decision (step 3 input, 5,286 chars total — everything ever rendered was in-window):** its own materialized table leads with the gold scalar, and the probe blocks (still in delta history) show both candidates:

```
[fraud_amount_lost]  0  Reports with $ Loss  987520
                     1  Total $ Loss  12537194708 ... 3  $1 - $1,000  624110 ...
[raw_preview]        2  Number of Fraud Reports,"2,600,678",
                     3  Reports with $ Loss,"987,520",38% of the total
```

Note: both wrong attractors (2,600,678 and 624,110) were ALSO visible at this decision. The winner's `$`-filter incidentally excluded `Number of Fraud Reports` from its working table — an accidental de-distraction, not an evidence gap.

## Walk: DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt (loser C1p + loser C3p — FAIL 0.827)

**Final answer:** 0.827 = 516308/624110. 6 agent steps (7 counted), 50,802 input tokens, $0.0354.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_preview` (probe, never deleted) | head 8 + mid 6 + tail 8 raw lines | G1 recon ✓ |
| 1 | `fraud_amount_lost` | `pd.read_csv(header=None)` → full 37x3 raw frame | G1 ✓ |
| 2 | `loss_1_500_reports` | locate `Reported Fraud Losses in $1 - $1,000 Range` section; sum five ≤$500 buckets AND **`total_1_1000` = sum of that sub-table** → the proportion base is framed as the $1-$1,000 population | **DIVERGES at G2**: never extracts `Reports with $ Loss`; G3 ✓ (516,308). Fails with `KeyError: 0` (stringified column names) |
| 3 | `loss_1_500_reports` (fix) | same semantics via `df.columns[0]` → 516308, 624110 | G2 divergence carried |
| 4 | `proportion_1_500` | 516308/624110 = 0.827271 | G4 executed on wrong base |
| 5 | (text) | Final Answer: 0.827 | wrong |

**First semantic divergence:** step 2, op `loss_1_500_reports`, plan item G2 — the returned pair `{total_1_500, total_1_1000}` hard-codes the reference population as the $1-$1,000 sub-table. The step-3 KeyError fix is orthogonal (schema artifact, same semantics).

**Rendered evidence at the decision (step 2 input, 4,467 chars) — a STRICT SUPERSET of the winner's:** the full 37-row frame rendered with zero truncation at 5k, gold's denominator line on screen twice:

```
[fraud_amount_lost]  2  Number of Fraud Reports   2,600,678   NaN
                     3  Reports with $ Loss       987,520     38% of the total
                     ...rows 9-19 ($1-$10,000+ buckets)... rows 23-32 ($1-$1,000 buckets)...
[raw_preview]        3  Reports with $ Loss,"987,520",38% of the total
```

The loser's error is NOT explained by absence — it saw more than the winner and chose the sub-table total anyway.

## Walk: DataflowSystemGPT52DeltaStats1kD2ProbePrompt (loser C2p — FAIL 0.199)

**Final answer:** 0.199 = 516308/2600678. 5 agent steps (6 counted), 40,068 input tokens, $0.0298.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_preview` (probe, never deleted) | read first 25 raw lines | G1 recon ✓ |
| 1 | `loss_1_1000_table` | csv.reader; state-machine extract of the ten $1-$1,000-range bucket rows → 10x3 | G1/G3-source ✓ |
| 2 | `fraud_report_totals` | scan file for row `Number of Fraud Reports` → 2,600,678 | **DIVERGES at G2**: extracts the all-reports scalar, not `Reports with $ Loss` |
| 3 | `prop_1_500` | five ≤$500 buckets (516,308) / 2600678 = 0.199 | G3 ✓; G4 on wrong base |
| 4 | (text) | Final Answer: 0.199 | wrong |

**First semantic divergence:** step 2, op `fraud_report_totals`, plan item G2 — a literal reading of "proportion of fraud reporters" (all 2.6M) with the gold scalar on screen.

**Rendered evidence at the decision (step 2 input, 5,797 chars):** the probe block shows both scalar candidates adjacent, including the 38% annotation; the D2 stats add nothing about population semantics:

```
[raw_preview]        2  Number of Fraud Reports,"2,600,678",
                     3  Reports with $ Loss,"987,520",38% of the total
   Column Schema and stats: - "line" (str): null=0, distinct=23, duplicate_values=2
   Output Table profile: - duplicate rows: 2 of 25 (8%)
[loss_1_1000_table]  stats: reports (numeric): null=0, mean=62411, min=14019, max=243174, top_10={...}
```

The C2p lever's entire information delta at this decision is bucket mean/min/max and a dup-row count — semantically irrelevant to WHICH total is the denominator. Evidence effectively identical to the winner's; the choice differs.

## Walk: DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt (WINNER C3p — PASS)

**Final answer:** 0.523. 7 agent steps (8 counted), 52,447 input tokens, $0.0425.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_preview` (probe, never deleted — re-renders every step in LATEST) | head 8 + mid 8 + tail 8 raw lines | G1 recon ✓ |
| 1 | `fraud_amounts` | parse the `$1 - $10,000 +` section → tidy 11x3 (label, count, low_usd) | G1 ✓; sum of its counts = 987,520 ≡ G2 by the closure identity |
| 2 | `prop_1_500` v1 | denominator = `fraud_amounts['count'].sum()` (= 987,520 ✓); numerator = the `$1 - $1,000` bucket (624,110) → 0.632; its own summary flags the proxy: "used as the $1-$500 bucket given available data" | G2 ✓; **transient DIVERGENCE at G3** |
| 3 | `fraud_1_1000_detail` | parse the `$1 - $1,000 Range` sub-table | recovery move (ValueError on empty count) |
| 4 | `fraud_1_1000_detail` (fix) | guard empty counts → 10x3 buckets | G3-source ✓ |
| 5 | `prop_1_500` EDIT | numerator = buckets `low_usd <= 401` (= $1-$100…$401-$500 = 516,308) / 987,520 = 0.522833 | **G3 ✓ G4 ✓** — divergence recovered |
| 6 | (text) | Final Answer: 0.523 | ✓ |

**First semantic divergence:** step 2 numerator proxy — recovered at step 5 (near-miss, self-corrected; the probe's line `16 "Reported Fraud Losses in $1 - $1,000 Range"` visible in its persistent block evidences the finer table it went to fetch).

**Rendered evidence at the denominator decision (step 2 input, 2,591 chars):** its own tidy 11-bucket table (fully rendered, sum = the with-loss population) plus the persistent probe block with `3 Reports with $ Loss,"987,520",38% of the total` and `2 Number of Fraud Reports,"2,600,678"`. Same facts as every other arm; it anchored on the with-loss table.

## Run-to-run instability — the seal (recovery-round records, all 2026-07-17)

Recorded attempts on this task today (round logs `logs/kb-rerunfail-*/legal-easy-19.log` + round membership):

| arm | attempt history (initial → recovery rounds) |
|---|---|
| 1k (winner) | run1 initial PASS · run2 initial FAIL → rec1 **0.523** PASS (rec2 not needed) — **pass→fail→pass same day** |
| 5k (loser) | run1 initial FAIL → rec1 "No response from agent" → rec2 NaN · run2 initial FAIL → rec1 **0.199** → rec2 **0.827** — two different wrong attractors in consecutive rounds |
| stats1k (loser) | initial FAIL → rec1 **0.199** → rec2 **0.199** — stable on the all-reports attractor |
| latest5k (winner) | initial FAIL → rec1 **0.827** → rec2 **0.523** PASS — produced the C1p loser's exact final answer one round before passing |

10 recorded outcomes across arms: 0.523 ×3, 0.199 ×3, 0.827 ×2, NaN ×1, no-response ×1. The same three attractors recur within and across arms; final pass/fail per arm is where each arm's 2-round recovery budget happened to stop.

## Pair verdicts

**C1p 1k > 5k: CHRONIC/VARIANCE (attribution rejected).** The one knob (result char limit 1000 vs 5000) changed only per-op row windows, and in the wrong direction for a lever story: the loser's decision context was a strict superset (full 37-row frame; gold's `Reports with $ Loss,"987,520",38% of the total` rendered twice) — the loser's error is not explained by absence, failing the skill's acceptance test. Both arms coin-flip on exactly this decision across their own same-day reruns (1k pass→fail→pass; 5k 0.199→0.827). Contexts are tiny (≤5.3k chars); zero render pressure; no probe-thrash, no churn flag, errors (KeyError/ValueError) orthogonal to the divergence.

**C2p 1k-schema > stats1k-D2: CHRONIC/VARIANCE (attribution rejected).** The stats lever's rendered delta at the loser's G2 decision — bucket mean/min/max, `duplicate rows: 2 of 25` — carries no information about which of three co-rendered totals is "fraud reporters"; both arms saw the same two scalar lines adjacent. The loser is actually the most STABLE arm (0.199 three times running): a consistent alternative (literal) reading of the question, not stats-induced noise — but the winner's own same-day fail→pass flip puts the pair difference inside variance. Nothing here supports "stats hurt".

**C3p latest5k > delta5k: CHRONIC/VARIANCE (attribution rejected).** One knob (latest vs delta), and the winner itself answered 0.827 — the loser's exact final answer, same wrong denominator class — in its previous recovery round before landing 0.523. Both arms' step-0/1 evidence was equivalent (both scalars + both section titles in-window at 5k); the winner's correct denominator came from a method choice (tidy-parse the $1-$10,000+ section, sum it) made on the same rendered facts the loser had. LATEST's persistent probe re-render (probes never deleted) kept the scalars on screen every step, but the delta losers' histories also retained them in-window — no differential.

**Shared loser mechanism (per the three-cell question):** the losers do NOT share a wrong answer (624,110 vs 2,600,678 denominators) — they share the mechanism class: **population mis-selection among three co-rendered "total" candidates on an ambiguous "proportion of fraud reporters" question**. The file's dirtiness never bit: every arm parsed the multi-block CSV correctly and produced the gold numerator 516,308 (the probe-beat's raw previews did their structural job; only the 1k arm actually retired its probes). The task's three-cell appearance this vintage is one interpretation coin-flip surfacing in three venns, consistent with its old-vintage chronic tag — it should stay noise-gated in all three pair aggregates.

Artifacts: `system_scratch/{DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt,DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt,DataflowSystemGPT52DeltaStats1kD2ProbePrompt,DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt}/legal-easy-19/`; rerun rounds `logs/kb-rerunfail-*ProbePrompt-20260717_*/legal-easy-19.log`; walks extracted via `scripts/extract_walk.py`, decision contexts via `extract_walk.op_blocks` over `inputMessages`.
