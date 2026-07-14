# Deep dive: environment-easy-3 — Latest3kSchemaOnly (268, PASS) vs Delta3kSchemaOnly (267, FAIL)

One of the two NON-chronic flips in the levers grid, and counter-intuitive: the
history-less mode (Latest) beat the history-carrying one (Delta). Verdict up
front: **REJECTED-method-choice** — the flip is an unforced join-key convention
sampled at ~4/49 arm frequency, not a context_mode effect.

## Task
Q: How many beaches had a higher bacterial exceedance rate for water samples collected in 2013 compared to 2012, excluding those with no samples in 2012?

D: `data/environment/input/water-body-testing-2012.csv` (15,377 rows x 11 cols) and `data/environment/input/water-body-testing-2013.csv` (15,388 rows x 11 cols)

Massachusetts beach water-quality testing; one row per bacterial sample. Real rows (file head, both files identical in shape):

```
Community Code,Community,County Code,County Description,Year,Sample Date,Beach Name,Beach Type Description,Organism,Indicator Level,Violation
242,Provincetown,001,Barnstable,2012,2012-06-05 00:00:00,333 Commercial Street,Marine,Enterococci,8,no
242,Provincetown,001,Barnstable,2012,2012-07-16 00:00:00,333 Commercial Street,Marine,Enterococci,8,no
242,Provincetown,001,Barnstable,2013,2013-07-15 00:00:00,333 Commercial Street,Marine,Enterococci,4,no   <- 2013 file
```

Column semantics:
- `Community Code` / `Community` — the town that runs the beach (242 = Provincetown). `County Code` / `County Description` — its county; `County Code` is zero-padded text in the raw CSV (`001`) but parses numeric under default `read_csv`.
- `Year` — redundant with the file split (zero off-year rows in either file).
- `Beach Name` — free-text beach name, **NOT unique statewide**. This is THE quirk that decides the task. Real collision rows from the 2012 file — three different physical "Sandy Beach"es in three towns/counties:

```
 Community Code Community  County Code County Description  Beach Name Violation
            292   Swansea            5            Bristol Sandy Beach        no
             71   Danvers            9              Essex Sandy Beach       yes
             65  Cohasset           21            Norfolk Sandy Beach        no
```

  Likewise `Crystal Lake` (Orleans / Newton / Carver), `Bayview` (Dennis / Dartmouth), `Atlantic Avenue`, `Black Rock`, `Edgewater`, `Flax Pond`, `Follins Pond`, ... — 14 colliding names in 2012 (1171 (community, name) groups vs 1155 bare names = 16 collapsed groups), 12 in 2013 (1176 vs 1162 = 14 collapsed).
- `Violation` — the exceedance flag, uniform lowercase `yes`/`no`, zero NaN; `Organism` + `Indicator Level` — the underlying measurement (not needed; `Violation` already encodes the exceedance).
- Verified FD: `Community Code -> County Code` has 0 violations across both files, so `(Community Code, Beach Name)` is exactly equivalent to gold's triple key on this data.

## Solution
From `solutions/environment/environment-easy-3.py`:

```
load_2012(water-body-testing-2012.csv,     load_2013(water-body-testing-2013.csv,
          default read_csv) ─┐                       default read_csv) ─┐
                             ▼                                          ▼
        lower(Beach Name)  [casing uniform: no-op]      lower(Beach Name)
                             ▼                                          ▼
  beach_id = str(Community Code)+str(County Code)+Beach Name   (same beach_id)
                             ▼                                          ▼
  groupby(beach_id):                                    groupby(beach_id):
    total_samples = count                                 total_samples = count
    num_violations = (Violation.lower()=='yes').sum()     num_violations = ...
    exceedance_rate = viol/samples          → 1171 rows   → 1176 rows
                             └───────────────┬───────────┘
                                             ▼
                inner join on beach_id  → 1138 rows
                [inner = "excluding those with no samples in 2012"]
                                             ▼
                count(rate_2013 > rate_2012) → **268**
```

Key annotations: the grain is the PHYSICAL beach `(community, county, name)`, not
the name string — that is the entire task difficulty. Key arithmetic on the real
data: composite key → 1171 / 1176 groups, inner join 1138, count 268; `Beach
Name` alone → 1155 / 1162 groups, inner join 1124, count **267**. The key choice
alone flips the answer; everything else in either trace is a no-op (exact-repro
verified in the semantic walk).

## What Delta3kSchemaOnly does
(loser: answer 267, 4 agent steps, 33,636 input tokens, 37.4s; trace via `python3 scripts/extract_walk.py --sut DataflowSystemGPT52Delta3kSchemaOnly --task environment-easy-3`)

- step 0 `wb2012`, `wb2013`: `pd.read_csv(...)` both files, default params — G1 ok, code identical to the winner's step 0.
- --> step 1 `exceedance_rates`: per year — filter `Year==year` (no-op), drop `Beach Name` NaN (no-op), flag = `Violation.strip().lower()=='yes'`; **`groupby('Beach Name')`** — single-column key; then inner merge on `Beach Name`, `samples_2012 > 0` guard, `higher_in_2013 = rate_2013 > rate_2012` → 1124x8. **DIVERGES at G3**: keying on the name string pools the 16 (2012) / 14 (2013) cross-community same-named groups into single pseudo-beaches, so the join has 1124 rows instead of 1138. Concretely, the whole off-by-one is `Bayview`: Dennis (CC 75, 0/13 → 1/14) and Dartmouth (CC 72, 0/12 → 1/13) BOTH went higher in 2013 = +2 under the composite key, but the pooled pseudo-beach (0/25 → 2/27) counts +1. Across all colliding names the contributions are composite 8 vs pooled 7 — net −1.
- step 2 `count_higher`: sum of the flag → 1x1 = 267 — correct given step 1.
- step 3 text: `Final Answer: 267` — off by one.

Exact repro on the CSVs reproduces the trace end-to-end (1155/1162 groups, 1124 joined rows as rendered, count 267) and confirms the key choice, not the filters, flips 268 → 267.

## What Latest3kSchemaOnly does
(winner: answer 268, 5 agent steps, 41,650 input tokens, 68.6s; trace via `python3 scripts/extract_walk.py --sut DataflowSystemGPT52Latest3kSchemaOnly --task environment-easy-3`)

- step 0 `wb2012`, `wb2013`: identical loads to the loser.
- step 1 `exceed_rate_2012`: drop `Violation` NaN (no-op, 0 NaNs); **`groupby(['Community Code','Beach Name'], dropna=False)`**; `samples=size`, `exceed=(strip/lower=='yes').sum()`, `rate_2012=exceed/samples` → 1171x5. G3 ok via a gold-EQUIVALENT variant: omits `County Code` (redundant under the CC→County FD) and lowercasing (casing uniform) — identical 1171/1176/1138 cardinalities and identical count on re-execution.
- step 2 `exceed_rate_2013`: same key, same aggregation → 1176x5.
- step 3 `higher_exceed_2013_count`: inner merge on `['Community Code','Beach Name']`; `samples_2012 > 0` guard (no-op after inner join); count `rate_2013 > rate_2012` → 1x1 = **268**.
- step 4 text: `Final Answer: 268`.

No near-misses, no recoveries — a clean 4-op run whose only load-bearing property is the composite key committed at step 1.

## Why Latest3kSchemaOnly succeeded but Delta3kSchemaOnly failed
Both arms commit the key at the SAME decision point — the first processing op,
with exactly one prior action (the two loads). What each had rendered at that
moment (sweep-era traces have empty thoughts; attribution is code + renders only):

Winner (LATEST grammar), `inputMessages` before step 1:

> `### Operator \`wb2012\` (DataLoading)` … `Output Table: 15377 rows, 11 cols` — 4 head rows (all `242 Provincetown … 333 Commercial Street`) + 4 tail rows (all `287 Sturbridge … Yogi Bear Campground`) + `Schema (11 cols): Community (str), Year (numeric), Violation (str), Community Code (numeric), … Beach Name (str)` — and the same block for `wb2013`.

Loser (DELTA grammar), `inputMessages` before its step 1:

> `## Agent Event 2` — Action block echoing the two loader codes, then `- operator wb2012 added` / `result:` … `Output Table: 15377 rows, 11 cols` — **the same 4 head rows + 4 tail rows + the identical `Schema (11 cols): …` line**, and the same for `wb2013`.

**The evidence was informationally identical.** Byte-level diff of the two
decision-time contexts finds only (1) grammar scaffolding (LATEST `### Operator`
blocks + one-line Summary vs DELTA event framing + code echo) and (2) task
file-list order (Latest3k lists `...-2013.csv` first) — which sits in the
initial task message, BEFORE any action, i.e. harness manifest ordering, not a
context-mode effect; the 49-arm crosstab kills it anyway (2013-first → 3x268 /
12x267; 2012-first → 1x268 / 30x267 — both orders overwhelmingly 267). Neither
context rendered any beach-name collision (the 8 visible rows per file cover
exactly two beaches, one community each — no `Sandy Beach`-style duplicate in
sight) and SchemaOnly renders no uniqueness/nunique stats. The winner's render
does not explain its action (nothing motivates compositing the key), and the
loser's render does not explain its error (the same `Community Code` column sat
equally visible in its schema line and sample rows).

So this is **method-choice, not a lever story**: the composite key appears in
only ~4/49 GPT-5.2 arms (Latest3kSchemaOnly, LatestSchemaConverge,
FullSchemaNoStats, DeltaWin3kCompressPromptAware) scattered across latest,
delta-windowed, and full modes; the winner's OWN mode family at 5k/7k budgets
picks `Beach Name` alone → 267, and every stats-enriched Latest arm → 267,
while a Delta-family arm picks the winning key. The ~8% rate is also exactly
why the task is absent from `chronic_flippers.json`: three identical-config
twin pairs simply never sampled the rare branch — "non-chronic" here is
twin-sample sensitivity, not determinism. **Verdict: REJECTED-method-choice.**
