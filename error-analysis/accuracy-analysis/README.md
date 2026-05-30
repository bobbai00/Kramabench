# Accuracy analysis — `DataflowSystemGPT52LatestStatsOnCap0` vs `DataflowSystemGpt52ResultChars1000NoActionDetailCarryMetadata`

Comparison scope: **legal domain only**, 30 tasks, both systems run on the same workload with the same underlying model (`gpt-5.2`). Folders compared:

- **Cap0**: `system_scratch/DataflowSystemGPT52LatestStatsOnCap0/` (this repo)
- **NACM**: `system_scratch/DataflowSystemGpt52ResultChars1000NoActionDetailCarryMetadata/` (in `~/Desktop/Kramabench/`)

Determined by comparing each task's `answer.json` to its `ground_truth.json` under the workload's `answer_type` policy (numeric_exact / numeric_approximate within 1% / string_exact / string_approximate / list_exact / list_approximate as set equality).

---

## Summary scoreboard

| | Cap0 | NACM |
|---|---:|---:|
| Passed | 21 / 30 | 28 / 30 |
| Cap0 won, NACM lost | **0** | — |
| NACM won, Cap0 lost | — | **7** |
| Both passed | 21 | 21 |
| Both failed | 2 | 2 |

NACM strictly dominates Cap0 on legal — there is no task where Cap0 produced a correct answer that NACM missed.

### The 7 tasks NACM passed but Cap0 failed

| Task | Type | GT | Cap0 | NACM |
|---|---|---:|---:|---:|
| `legal-easy-4` | numeric_exact | 2,111,635 | 2,280,292 | 2,111,635 |
| `legal-easy-19` | numeric_exact | 0.523 | 0.439 | 0.523 |
| `legal-hard-1` | numeric_approximate | 12,964.8727 | 13,902.2400 | 12,964.8727 |
| `legal-hard-15` | numeric_exact | 243,377 | 593,524 | 243,377 |
| `legal-hard-18` | numeric_exact | 91,000 | 520,000 | 91,000 |
| `legal-hard-22` | numeric_exact | 0.0555 | 0.0097 | 0.0555 |
| `legal-hard-23` | string_approximate | District of Columbia | Florida | District of Columbia |

### Both-failed tasks (does not affect ranking)
- `legal-easy-21` (numeric_exact): GT=15387, both produced 16589
- `legal-hard-24` (string_exact): GT has no comma before "CA", both put a comma — same wrong answer

---

## Root cause

The model, the codebase, the data lake, and the workflow tooling are all identical between the two systems. What differs is **how the agent context is serialized to the LLM** at every step. The defects compound into a single failure pattern:

> Cap0 does not show the LLM enough information about the data it just produced.

There are two compounding mechanisms.

### Mechanism 1 — Cap0 discards the ReAct history

Every Cap0 LLM input contains:

```
# Recent Agent Events (last 0 of N events)
(... N earlier events omitted to keep context focused on recent work ...)
```

That is the meaning of "Cap0" in the SUT name (the `recentEventsCap` knob was added in commit `447e01a`; Cap0 = 0). The LLM never sees its own prior thoughts, prior tool-call code, or prior tool-result messages — only a structured snapshot of the *current* dataflow.

NACM, by contrast, sends the entire OpenAI message-by-message conversation: every prior assistant turn and every prior tool result is still visible.

**Effect**: when the snapshot is ambiguous, Cap0 has no way to recall *"the filter `[Army, Navy]` already returned (0,0)"* or *"I just inspected and saw a `Fraud & Other` row"*. It re-derives intent each turn from a static snapshot, which leads either to repeated guess loops (cost) or premature commitment to a wrong filter (accuracy).

### Mechanism 2 — Cap0's per-operator snapshot is thinner than NACM's tool-result text

For the same operator output, the two systems render different information to the LLM:

| | **Cap0 dataflow snapshot** | **NACM raw tool-result text** |
|---|---|---|
| Sample rows | head 4 + tail 2 | head 4 + tail 5 (full table if small) |
| String column stats | `null=N, distinct=M`; `top_10={…}` **only when distinct is small** | `null=N, distinct=M, top_10={…}` **always emitted (10 most-common)** |
| Numeric column stats | `null, mean, min, max` | `null, distinct, mean, std, min, p25, median, p75, max` |

The **always-emit `top_10`** behavior is the single most impactful difference. NACM gives the LLM the actual categorical vocabulary of every string column; Cap0 hides it whenever the column has more than ~10 distinct values, which is exactly when the LLM most needs it.

---

## Three representative case studies

Traces for each are duplicated under `traces/<task-id>/cap0/` and `traces/<task-id>/nacm/`.

### Case A — `legal-easy-4`: truncated head hides the right category

**Query**: "How many frauds were reported by FTC over the web between 2022 and 2024 in total?"
**GT**: 2,111,635 — **Cap0**: 2,280,292 ❌ — **NACM**: 2,111,635 ✓

The CSV `2024_CSN_Data_Contributors.csv` has 142 rows. Each year (2022, 2023, 2024) has multiple "FTC - Web Reports" sub-rows:

```
row 3: 2022  FTC - Web Reports (IDT)             796,366    ← identity theft
row 5: 2022  FTC - Web Reports (Fraud & Other)   693,789    ← actual frauds
```

**What Cap0's LLM saw** (from `cap0/react_steps.json` step 4, `inputMessages[0]`):

```
Operator contributors
Result:
  Output table shape: (142, 4)
        Data Contributors    Unnamed: 1    Unnamed: 2    Unnamed: 3
  0     NaN
  1     FTC
  2     Year           Data Contributor      # of Reports    %
  3     2022           FTC - Web Reports (IDT)   796,366     14.98%
  ...   ...
  140   NaN
  141   Source: …
  Column stats:
  - "Data Contributors" (str): null=6, distinct=10, top_10={"2022"=43, "2023"=42, …}
  - "Unnamed: 1" (str): null=11, distinct=45              ← no top_10 because distinct=45
  - "Unnamed: 2" (str): null=11, distinct=128             ← no top_10
```

Cap0 saw `FTC - Web Reports (IDT)` as the only FTC-web category in the head. The actual `(Fraud & Other)` row at index 5 is hidden in `...`. `distinct=45` on the contributor column suppresses `top_10`. The LLM has no signal that another category exists, so it filters on `(IDT)` and sums 2,280,292 — the **identity-theft** total.

**What NACM's LLM saw** (from `nacm/messages.json` step 4 tool result, after an `inspect_raw_contributors` step):

```
Output table shape: (40, 4)
        0       1       2       3
[stats] String,null=3,distinct=8   String,null=7,distinct=23   ...
0   Data Contributors  NaN  NaN  NaN
1   NaN
2   FTC
3   Year     Data Contributor       # of Reports     %
4   2022     FTC - Web Reports (IDT)              796,366    14.98%
5   2022     FTC - Web Reports (Fraud & Other)    693,789    13.05%      ← visible
...
```

NACM took an extra exploratory step (steps 1-2 sampled 6 rows, steps 3-4 read 40 rows with `header=None`) before computing. Combined with NACM's preserved ReAct history (so step 5 could refer back to step 4's output), it filtered on `(Fraud & Other)` and summed correctly to 2,111,635.

**Mechanism**: a combination of (1) head/tail truncation hiding the disambiguating row, (2) `top_10` suppression hiding the categorical alternatives, (3) no agent history allowing exploratory steps to inform later compute steps.

### Case B — `legal-easy-19`: top_10 suppression hides the aggregate row

**Query**: "What is the proportion (round to 3 decimal places) of fraud reporters who lost between $1-$500 in 2024?"
**GT**: 0.523 — **Cap0**: 0.439 ❌ — **NACM**: 0.523 ✓

The CSV `2024_CSN_Fraud_Reports_by_Amount_Lost.csv` contains both **bin rows** and an **aggregate row**:

```
$1 - $100
$101 - $200
$201 - $300
$301 - $400
$401 - $500
…
$1 - $1,000     ← AGGREGATE of $1-$100 through $901-$1,000
$1,001 - $2,000   ← AGGREGATE of $1,001-$1,100 through …
…
```

**What Cap0's LLM saw** (`cap0/react_steps.json` step 5, `inputMessages[0]`):

```
Column stats:
- "Fraud Reports by Amount Lost" (str): null=5, distinct=31    ← no top_10 (distinct=31)
- "Unnamed: 1" (str): null=9, distinct=26                       ← no top_10
- "Unnamed: 2" (str): null=35, distinct=1, top_10={"38% of the total"=1}
```

`distinct=31` is above Cap0's threshold for emitting `top_10`. Cap0's `fraudAmountLostProportion1to500` operator wrote a rule "select bins where label startswith '$' and parsed lower bound between 1 and 500" — which silently picks up the `$1 - $1,000` aggregate. Numerator 1,140,418 / total 2,600,678 = **0.439**.

**What NACM's LLM saw** (`nacm/messages.json` step 2 tool result):

```
Output table shape: (36, 3)
        Fraud Reports by Amount Lost            Unnamed: 1               Unnamed: 2
[stats] str,null=5,distinct=31, top_10={"null"=5,
        "$1 - $1,000"=1, "$1 - $100"=1,
        "$1,001 - $2,000"=1, "$101 - $200"=1,
        "$2,001 - $3,000"=1, "$201 - $300"=1,
        "$3,001 - $4,000"=1, "$301 - $400"=1,
        "$4,001 - $5,000"=1}                    ← TOP_10 EMITTED AT DISTINCT=31
```

NACM's `top_10` listed both `$1 - $1,000` and `$1 - $100` distinctly. The aggregate vs bin split was visible at first read. NACM excluded the aggregate and got 0.523.

**Mechanism**: pure `top_10` suppression. Same data, same model, different rendering.

### Case C — `legal-hard-22`: top_10 suppression hides parent/child structure

**Query**: "What is the proportion (round to 4 decimal places) of all reports who reported identity theft with Bank Account (Theft Type) and New Accounts (Theft Subtype)?"
**GT**: 0.0555 — **Cap0**: 0.0097 ❌ — **NACM**: 0.0555 ✓

The CSV `2024_CSN_Report_Type.csv` contains parent rows (`Fraud`, `Identity Theft`, `Other`) **and** their breakouts. The numerator pipeline is straightforward (Bank Account / New Accounts = 62,982 in both systems). The disagreement is in the denominator.

**Cap0's final pipeline** (`cap0/react_steps.json` step 4, calcProportion operator):

```
denominator_total_all_reports = 6,495,932    ← sum of every "# of Reports" cell
proportion = 62,982 / 6,495,932 = 0.009696
```

That denominator inflates the actual ~6.47M total reports by double-counting parents and children together.

`Report Type` column in the Cap0 snapshot:
```
- "Report Type" (str): null=6, distinct=20    ← no top_10 (distinct=20)
```

With `top_10` suppressed at `distinct=20`, the LLM cannot see that the column mixes top-level types (`Fraud`, `Identity Theft`, `Other`) with their subtype rows. Summing the whole column inflates.

**NACM** (`nacm/messages.json`) computed the correct denominator (~1,135,000 = total Identity Theft reports, which the workload treats as the "all reports" parent) and got 0.0555. Its visible `top_10` showed the parent/child distinction.

**Mechanism**: same as Case B — `top_10` suppression at moderate distinct counts hides the structural shape of the column.

---

## Why this is a *serialization* bug, not a model bug

The same GPT-5.2 model, given NACM's richer column stats and 4+ exploratory steps of preserved history, solves all 7 failure tasks. Given Cap0's truncated snapshot and zero history, the same model commits to wrong filters in 2 LLM calls. The failure is reproducible across very different question types (count totals, proportions of bins, parent-vs-child denominators, string lookups) and all of them share the same underlying signature: **a column with > ~10 distinct values whose `top_10` would have disambiguated the right filter, but the snapshot omits it**.

---

## Suggested fixes (in priority order)

1. **Always emit `top_10`** for string columns regardless of `distinct`. Single highest-leverage change — fixes Cases B and C outright and substantially helps Case A.
2. **Add quartile stats** (`distinct`, `std`, `p25`, `median`, `p75`) to numeric columns in the snapshot. Aligns Cap0's render to NACM's.
3. **Reconsider `recentEventsCap=0`.** Even keeping the last 1-2 events (the last tool-call code + result message) would let the agent run multi-step exploration without burning iterations on repeated guesses.
4. **Optionally widen head/tail rows** when `distinct` is high. Likely unnecessary once #1 is in place.

---

## Files in this folder

```
accuracy-analysis/
├── README.md                          (this file)
└── traces/
    ├── legal-easy-4/
    │   ├── cap0/    (full Cap0 trace: react_steps.json, answer, ground_truth, etc.)
    │   └── nacm/    (full NACM trace: messages.json, answer, ground_truth, etc.)
    ├── legal-easy-19/
    │   ├── cap0/
    │   └── nacm/
    └── legal-hard-22/
        ├── cap0/
        └── nacm/
```

Each task folder contains both systems' raw artifacts so the diffs above are independently verifiable. The key files to read are:

- `cap0/react_steps.json` → look at any agent step's `inputMessages[0].content` to see the dataflow snapshot the LLM saw
- `nacm/messages.json` → walk the OpenAI conversation; tool-result messages contain the raw rendering the LLM saw
- `answer.json` and `ground_truth.json` → the verdict
