# legal-easy-19 — deep dive (counter-intuitive: leaner modes won, and the same task flipped three venns)

PROBE-STAR vintage (all arms raw-probe prompt, code mode; knob diffs verified from
`config.json`: C1p = char limit 1000 vs 5000, one knob; C2p = the stats bundle
`column_stats`+`data_level`; C3p = `context_mode` delta vs latest, one knob, both 5k).
One task occupies an exclusive venn cell in **all three probe-star pairs** — A-only in
C1p and C2p (winner = the leanest arm), B-only in C3p (winner = Latest). `legal-easy-19`
is in `chronic_flippers.json` (old vintage, advisory) and `*`-starred in all three venns.
Verdict up front: **all three pairs CHRONIC/VARIANCE — one denominator coin-flip among
three co-rendered totals, resampled per arm; the 10-attempt same-day flip distribution
seals it.** All four arms computed the gold NUMERATOR (516,308) correctly.

| Arm | role | knobs | agent steps | input tok | cost_usd | answer | verdict |
|---|---|---|---|---|---|---|---|
| **Delta1kSchemaOnlyProbePrompt** (Y) | **WINNER C1p + C2p** | delta, 1k, no stats, D1 | 5 | 38,232 | 0.0265 | **0.523** | PASS |
| **Latest5kSchemaOnlyProbePrompt** (Y) | **WINNER C3p** | latest, 5k, no stats, D1 | 7 | 52,447 | 0.0425 | **0.523** | PASS |
| Delta5kSchemaOnlyProbePrompt (X, C1p + C3p) | loser | delta, **5k**, no stats, D1 | 6 | 50,802 | 0.0354 | 0.827 | FAIL |
| DeltaStats1kD2ProbePrompt (X, C2p) | loser | delta, 1k, **stats, D2** | 5 | 40,068 | 0.0298 | 0.199 | FAIL |

## Task
Q: "What is the proportion (round to 3 decimal places) of fraud reporters who lost between $1-$500 in 2024?"

D: one file, `data/legal/input/csn-data-book-2024-csv/CSVs/2024_CSN_Fraud_Reports_by_Amount_Lost.csv`
— a 37-line, CRLF-terminated, **multi-block** CSV (FTC Consumer Sentinel Network Data
Book 2024): a title line, a scalar block, two bucket sub-tables, and footnotes, separated
by blank `,,` lines; every numeric is quoted with thousands commas. Real rows:

Scalar block (file lines 1–6; the three denominator candidates live here and in the next
block's first bucket, all within 8 non-blank lines of each other):

```
Fraud Reports by Amount Lost,,
Number of Fraud Reports,"2,600,678",
Reports with $ Loss,"987,520",38% of the total
Total $ Loss,"$12,537,194,708 ",
Median $ Loss,$497 ,
```

Sub-table 1, `"Reported Fraud Losses in $1 - $10,000 + Range"` (11 buckets; first + last shown):

```
Amount Lost,# of Reports,
"$1 - $1,000","624,110",
"$1,001 - $2,000","97,799",
...
"More than $10,000","124,708",
```

Sub-table 2, `"Reported Fraud Losses in $1 - $1,000 Range"` (10 buckets; the numerator's five):

```
Amount Lost ,# of Reports,
$1 - $100,"243,174",
$101 - $200,"114,336",
$201 - $300,"67,064",
$301 - $400,"44,982",
$401 - $500,"46,752",
```

Semantics + quirks (verified against the file):
- `Number of Fraud Reports` = **2,600,678** — ALL fraud reports, with or without a
  reported dollar loss. Wrong denominator; /2600678 → **0.199**.
- `Reports with $ Loss` = **987,520** — the with-$-loss population = **gold denominator**.
  The file itself annotates the relationship: "38% of the total" (987520/2600678 = 0.3797).
  Closure identity: the 11 buckets of sub-table 1 sum **exactly** to 987,520.
- Sub-table 1's first bucket `$1 - $1,000` = **624,110** — a bucket count, not a
  population total. Wrong denominator; /624110 → **0.827**.
- Numerator = sub-table 2's five ≤$500 buckets: 243,174+114,336+67,064+44,982+46,752 =
  **516,308**. Answer map (numerator fixed): **0.523** = /987520 (gold) · **0.827** =
  /624110 · **0.199** = /2600678 · 0.632 = 624110/987520 (right base, bucket-proxy
  numerator). Every answer observed today is one of these attractors.
- Dirtiness that did NOT bite: CRLF endings, quoted thousands commas, trailing spaces
  (`"$12,537,194,708 "`, header `Amount Lost ` in sub-table 2), repeated header-like rows
  (dup rows 2 of 25 in one arm's probe render), footnote lines. All four arms parsed the
  blocks correctly — the failure surface is purely WHICH total is "fraud reporters".

## Solution
From `solutions/legal/legal-easy-19.py` (custom block parser; commas stripped inside
quotes, so the sub-table key becomes `$1 - $1000`), as an operator graph:

```
parse_blocks(2024_CSN_Fraud_Reports_by_Amount_Lost.csv)    # strip commas-in-quotes,
        │                                                  # split on blank ',,' lines →
        │                                                  # 4 scalar attrs + 2 sub-tables
        ├── scalar 'Reports with $ Loss' ──────────────────────────── denominator = 987,520
        │
        └── sub-table 'Reported Fraud Losses in $1 - $1000 Range'
                │  filter(label ∈ {$1-$100, $101-$200, $201-$300, $301-$400, $401-$500})
                ▼
            sum('# of Reports') = 516,308 ─────────────────────────── numerator
                                                │
                                                ▼
                        516308 / 987520 = 0.522833 → round(3) → 0.523
```

Annotated: **load** = line-level custom parse (no pandas read_csv survives the multi-block
layout unmangled) · **denominator grain** = the with-$-loss population scalar — NOT all
reports (2,600,678), NOT the $1-$1,000 bucket (624,110) · **numerator grain** = five
$1–$500 buckets from the FINER sub-table · **output** = ratio rounded to 3 dp.

## What Delta5kSchemaOnlyProbePrompt does (mode X: C1p loser + C3p loser — 0.827, FAIL)
- **step 0** `raw_preview` (probe, never deleted): head 8 + mid 6 + tail 8 raw lines. Recon ✓.
- **step 1** `fraud_amount_lost`: `pd.read_csv(header=None)` → the full raw **37×3** frame. ✓.
- **--> step 2** `loss_1_500_reports`: locates the `Reported Fraud Losses in $1 - $1,000
  Range` section, sums the five ≤$500 buckets AND returns
  `{total_1_500, total_1_1000 = sum of that sub-table}` — the proportion base is framed
  as the **$1-$1,000 bucket population**, never touching `Reports with $ Loss`.
  **Divergence from the solution's denominator node.** (The step itself dies on
  `KeyError: 0` — stringified column names — an orthogonal schema artifact.)
- **step 3** `loss_1_500_reports` (fix): same semantics via `df.columns[0]` → 516,308 and
  624,110. Numerator ✓; divergence carried.
- **step 4** `proportion_1_500`: 516308/624110 = 0.827271.
- **step 5** Final Answer: **0.827**.

## What DeltaStats1kD2ProbePrompt does (mode X: C2p loser — 0.199, FAIL)
- **step 0** `raw_preview` (probe, never deleted): first 25 raw lines — both scalar
  candidates on screen. Recon ✓.
- **step 1** `loss_1_1000_table`: csv.reader state machine → the ten $1-$1,000-range
  bucket rows, 10×3. Numerator source ✓.
- **--> step 2** `fraud_report_totals`: scans the file for the row
  `Number of Fraud Reports` → **2,600,678**. **Divergence from the solution's denominator
  node:** a literal reading of "proportion of fraud reporters" (all 2.6M reports), taken
  with the gold scalar rendered adjacent in its own probe block.
- **step 3** `prop_1_500`: five buckets (516,308) / 2,600,678 = 0.199. Numerator ✓; base wrong.
- **step 4** Final Answer: **0.199**.

## What Delta1kSchemaOnlyProbePrompt does (mode Y: C1p + C2p WINNER — 0.523, PASS)
- **step 0** `raw_preview` (probe): first 6 raw lines. Recon ✓.
- **step 1** `raw_preview_mid` (probe): head 10 + mid 10 + tail 10 — 30 of the 37 lines. ✓.
- **step 2** `fraud_amount_lost` + **delete both probes** (full probe retire): csv.reader,
  keep rows with `$` in col0 AND numeric col1 → **24×2** = 3 $-scalars + all 21 bucket
  rows. The `$`-filter incidentally drops `Number of Fraud Reports` from the working
  table (an accidental de-distraction — the value stayed visible in delta history). ✓.
- **step 3** `loss_1_500_proportion`: `total = df[amount_range == 'Reports with $ Loss']`
  (987,520); `num = sum of the five ≤$500 buckets` (516,308); `round(num/total, 3)`.
  **Denominator ✓ numerator ✓ rounding ✓.**
- **step 4** Final Answer: **0.523**. No divergence at any step.

## What Latest5kSchemaOnlyProbePrompt does (mode Y: C3p WINNER — 0.523, PASS)
- **step 0** `raw_preview` (probe, never deleted — LATEST re-renders it every step):
  head 8 + mid 8 + tail 8 raw lines. Recon ✓.
- **step 1** `fraud_amounts`: tidy-parse of sub-table 1 → 11×3 (range_label, count,
  low_usd). Its counts sum to 987,520 ≡ the gold denominator by the closure identity. ✓.
- **step 2** `prop_1_500` v1: denominator = `fraud_amounts['count'].sum()` (**= 987,520,
  correct base**); numerator = the `$1 - $1,000` bucket (624,110) → 0.632 — a **transient
  numerator-proxy near-miss**, self-flagged in its own summary: "(used as the $1-$500
  bucket given available data)".
- **step 3** `fraud_1_1000_detail`: goes to fetch the finer sub-table (ValueError on an
  empty count row).
- **step 4** `fraud_1_1000_detail` (fix): guard empties → 10×3 buckets. ✓.
- **step 5** `prop_1_500` EDIT: numerator = buckets `low_usd <= 401` = 516,308; / 987,520
  = 0.522833. **Near-miss recovered** — the probe's persistent line
  `16 "Reported Fraud Losses in $1 - $1,000 Range"` evidenced the finer table it fetched.
- **step 6** Final Answer: **0.523**.

## Why Y succeeded but X failed
**The evidence was effectively identical across all four arms — in the C1p pair the
loser's was a strict superset — so this is method-choice / chronic-variance, not a lever
effect.** The single decision separating 0.523 from 0.827 from 0.199 is which of THREE
co-rendered totals is the denominator, on the ambiguous English "proportion of fraud
reporters". Every arm had both scalar candidates and the bucket total rendered at its
denominator decision:

- **Winner 1k (step-3 input, 5,286 chars — everything ever rendered still in-window):**
  its own table leads with the gold scalar, and BOTH wrong attractors are also visible —
  `[fraud_amount_lost] 0 Reports with $ Loss 987520 ... 3 $1 - $1,000 624110 ...
  18 $401 - $500 46752` plus the probe block
  `2 Number of Fraud Reports,"2,600,678", | 3 Reports with $ Loss,"987,520",38% of the total`.
- **5k loser (step-2 input, 4,467 chars) — a STRICT SUPERSET of the winner's:** the full
  37-row frame with zero truncation, gold's line on screen TWICE —
  `2 Number of Fraud Reports 2,600,678 | 3 Reports with $ Loss 987,520 38% of the total`
  (frame) and `Reports with $ Loss,"987,520",38% of the total` (probe) — and it still
  framed the base as the $1-$1,000 sub-table. Its error is not explained by absence.
- **Stats loser (step-2 input, 5,797 chars):** probe block with both scalars adjacent
  (same two lines as above); the C2p lever's entire information delta is
  `- "line" (str): null=0, distinct=23, duplicate_values=2`, `duplicate rows: 2 of 25
  (8%)`, and bucket stats `reports (numeric): null=0, mean=62411, min=14019, max=243174`
  — semantically irrelevant to WHICH total is the denominator.
- **Latest winner (step-2 input, 2,591 chars — the SMALLEST decision context of the
  four):** its tidy 11-bucket table (whose sum IS the gold base) plus the persistent
  probe block with both scalars. Same facts; it anchored on the with-loss table — then
  needed a numerator recovery the delta winner never needed.

**The noise seal — 10 recorded attempts on this task today (initial runs + `rerun-failed`
rounds, logs `logs/kb-rerunfail-*ProbePrompt-20260717_*/legal-easy-19.log`):**

| arm | attempt history (initial → recovery rounds) |
|---|---|
| 1k (winner) | run1 initial PASS · run2 initial FAIL → rec1 **0.523** PASS — pass→fail→pass same day |
| 5k (loser) | run1 initial FAIL → rec1 no-response → rec2 NaN · run2 initial FAIL → rec1 **0.199** → rec2 **0.827** — two different wrong attractors in consecutive rounds |
| stats1k (loser) | initial FAIL → rec1 **0.199** → rec2 **0.199** — stable on the literal reading |
| latest5k (winner) | initial FAIL → rec1 **0.827** → rec2 **0.523** PASS — produced the C1p loser's exact answer one round before passing |

Distribution: **0.523 ×3 · 0.199 ×3 · 0.827 ×2** (+1 NaN, +1 no-response). The same three
attractors recur within and across arms; each arm's final mark is where its 2-round
recovery budget happened to stop. The C3p winner emitting the C3p loser's exact wrong
answer (0.827) one round earlier is the cleanest possible demonstration that the
denominator choice is a per-sample coin, not a mode property.

## Pair verdicts
- **C1p Delta1kSchemaOnly > Delta5kSchemaOnly: CHRONIC/VARIANCE (attribution rejected).**
  The one knob (1000 vs 5000 chars) moved per-op row windows in the WRONG direction for a
  lever story: the loser's decision context was a strict superset with the gold line
  rendered twice. Both arms coin-flip on this exact decision across their own same-day
  reruns (1k pass→fail→pass; 5k 0.199→0.827). Listed chronic flipper; contexts tiny, zero
  render pressure, KeyError orthogonal.
- **C2p Delta1kSchemaOnly > DeltaStats1kD2: CHRONIC/VARIANCE (attribution rejected).**
  The stats lever's rendered delta (bucket marginals, dup-row counts) carries no
  information about population semantics; both arms saw the same two scalar lines
  adjacent. The loser is actually the most STABLE arm (0.199 three times) — a consistent
  literal reading, not stats-induced noise — but the winner's own fail→pass flip puts the
  pair inside variance. Nothing here supports "stats hurt".
- **C3p Latest5kSchemaOnly > Delta5kSchemaOnly: CHRONIC/VARIANCE (attribution rejected).**
  One knob (latest vs delta), and the winner itself answered 0.827 — the loser's exact
  final answer — in its previous recovery round. LATEST's persistent probe re-render kept
  the scalars on screen every step, but the delta arms' in-window histories retained the
  same lines — no differential. The winner's correct base came from a method choice
  (tidy-parse sub-table 1 and sum it) made on the same rendered facts.
- **Shared-loser mechanism (the three-cell question):** the losers do NOT share a wrong
  answer (624,110 vs 2,600,678 bases) — they share a mechanism class: **population
  mis-selection among three co-rendered totals on an ambiguous "proportion of fraud
  reporters"**. The file's dirtiness never bit (all four arms produced the gold numerator
  516,308; the probe beat did its structural job in every arm — only the 1k winner
  actually retired its probes). One interpretation coin surfacing in three venns:
  keep this task noise-gated in all three pair aggregates.
