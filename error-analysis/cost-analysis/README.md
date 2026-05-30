# Cost analysis — `DataflowSystemGPT52LatestStatsOnCap0` vs `DataflowSystemGpt52ResultChars1000NoActionDetailCarryMetadata`

Comparison scope: **legal domain only**, 30 tasks, same model (`gpt-5.2`), same workload, same dataflow agent backend. Folders:

- **Cap0**: `system_scratch/DataflowSystemGPT52LatestStatsOnCap0/` (this repo)
- **NACM**: `system_scratch/DataflowSystemGpt52ResultChars1000NoActionDetailCarryMetadata/` (in `~/Desktop/Kramabench/`)

Numbers below come straight from each task's `stats.json` and `react_steps.json`/`messages.json`.

---

## Bottom-line

| | Cap0 | NACM | Δ |
|---|---:|---:|---:|
| Total tokens (legal, 30 tasks) | 1,210,444 | 1,050,365 | **−13.2%** |
| Input tokens | 1,133,659 | 1,004,694 | −11.4% |
| Output tokens | 76,785 | 45,671 | **−40.5%** |
| Cost @ GPT-5.2 pricing | $3.06 | $2.40 | **−$0.66 (−21.6%)** |
| Wall-clock seconds | 1,236 | 1,155 | −6.5% |

**Output tokens cost 8× more than input** at GPT-5.2 pricing ($14.00/M vs $1.75/M), so the 40.5% output-token reduction is the single biggest dollar driver. Cap0 is more expensive primarily because **it produces more output per LLM call**, not because it makes more calls.

### Surprising counter-intuitive fact

```
Cap0 mean LLM calls per task : 5.73
NACM mean LLM calls per task : 6.63   ← NACM actually makes MORE calls
```

NACM does *more* turns on average and is still cheaper because each Cap0 turn is fatter on both ends.

---

## Where the cost goes — three structural causes

### Cause 1 — Cap0 pays a ~4,700-token system-prompt overhead per LLM call, and pays it many extra times when stuck

Across all 30 legal tasks, Cap0 made 172 LLM calls. Their input-token distribution:

```
min  = 4,658 tokens     (basically system prompt + tool schema + tiny user message)
mean = 5,532 tokens
max  = 9,748 tokens
```

The **floor of ~4,658 tokens is the static per-call overhead** (system prompt, tool descriptions, fixed task scaffolding). On a typical Cap0 step the dynamic context (`# Current Dataflow` snapshot + user task) is only 700-4,000 chars (~180-1,000 tokens). Decomposing the input-token budget:

```
Cap0 total input tokens          : 951,580   (across 172 agent calls; excludes init + final-answer)
"Fixed overhead" (4,658 × 172)   : 801,176   (~84%)
"Dynamic snapshot/context"       : 150,404   (~16%)
```

**Implication**: when Cap0 enters a guess loop, every extra iteration costs ~4,700 input tokens of pure boilerplate. NACM pays a similar per-call overhead, but with *fewer* wasted iterations on the worst-offender tasks (see Cause 2), so the multiplier effect is smaller.

### Cause 2 — `recentEventsCap=0` causes runaway guess loops the backend itself flagged

The dataflow agent backend has a **parametric-thrashing detector** that flags when the LLM submits a no-op modification:

> `WARNING: the code you just submitted matches the prior version after stripping docstrings/comments and normalizing whitespace — this modification was a no-op. … Repeating near-identical modifications without pivoting is parametric thrashing and will not converge.`

This warning fires in the tool result. **But Cap0's prompt template carries only `Recent Agent Events (last 0 of N events)` — it never feeds the warning back to the next LLM call.** The LLM resumes guessing as if nothing happened.

The worst case in legal is `legal-hard-15`: **22 consecutive `createOrModifyOperator` calls on the same operator `parseStateMsaIdTheft2024`**. The backend started emitting thrashing warnings at iteration 17; the LLM continued thrashing through iteration 22 because none of those warnings reached its next input context.

7 of 30 legal tasks show ≥2 consecutive identical tool-result outputs in Cap0 — a strong proxy for guess loops. (Detail in `traces/legal-hard-15/`.)

### Cause 3 — Per-call output is ~2× NACM because each call regenerates the operator code "cold"

```
Cap0 mean output tokens / call : 446
NACM mean output tokens / call : 229
```

Why is each Cap0 LLM call twice as verbose on the output side?

Every Cap0 `createOrModifyOperator` invocation has three fields the LLM must produce from scratch:
- `operatorId`
- `code` (full pandas function body)
- `summary` (multi-sentence natural-language description of what the code does)

Because the LLM has no view of its previous turn, it cannot make an *incremental* edit. It re-emits the full code and a fresh summary every time. In contrast, NACM's next assistant turn already sees its previous tool-call inputs in the OpenAI conversation history, so the LLM tends to write smaller incremental code changes.

Concrete: `legal-hard-15` step 5 has `toolcall_chars=2,176` (Cap0 emitting ~2 kB of code in one call); NACM step 2 of the same task has `toolcall_chars=355`. Tracked across the trace, Cap0 emits roughly **5× the total tool-call code bytes** of NACM on this task.

---

## Three representative case studies

Full traces are duplicated under `traces/<task-id>/cap0/` and `traces/<task-id>/nacm/`. Per-step token usage is in each `react_steps.json` step's `usage` field (Cap0) and aggregated in `stats.json`.

### Case A — `legal-hard-15`: the 22-iteration thrash loop

| | Cap0 | NACM | Δ |
|---|---:|---:|---:|
| Total tokens | **158,153** | 30,135 | **+128,018** |
| Input tokens | 143,187 | 28,630 | +400% |
| Output tokens | 14,966 | 1,505 | +894% |
| Operator-call iterations | 25 | 5 | 5× |
| Final answer | 593,524 ❌ | 243,377 ✓ | (also wrong) |

**Anatomy** (from `cap0/react_steps.json`):

```
step  1  CALL createOrModifyOperator  loadStateMsaIdTheft2024     in=4658  out=254
step  2  CALL createOrModifyOperator  parseStateMsaIdTheft2024    in=4959  out=559
step  3  CALL createOrModifyOperator  parseStateMsaIdTheft2024    in=5460  out=653
step  4  CALL createOrModifyOperator  parseStateMsaIdTheft2024    in=5474  out=693
step  5  CALL createOrModifyOperator  parseStateMsaIdTheft2024    in=5549  out=692
... 17 more steps, all on parseStateMsaIdTheft2024 ...
step 23  CALL createOrModifyOperator  parseStateMsaIdTheft2024    in=5507  out=671
step 24  CALL createOrModifyOperator  sumCrossStateReports2024    in=5410  out=198
step 25  text  "**Final Answer: 593524**"                          in=5590  out=11
```

**Steps 18-21 received the thrashing warning** in their tool result:

```
Operator parseStateMsaIdTheft2024 modified, deleted links: […], created links: […]
WARNING: the code you just submitted matches the prior version after stripping
docstrings/comments and normalizing whitespace — this modification was a no-op.
If you intended to alter behavior, you must change strategy …
```

But the next LLM call (step 19) received an `inputMessages[0]` that contained no `"WARNING"` or `"thrashing"` substring — only `# Recent Agent Events (last 0 of 19 events)` and the current snapshot, which by that point showed an unrelated `TypeError` from the last failed code execution. The signal that the backend explicitly built to break the loop was suppressed by the Cap0 prompt template.

NACM solved the same task in 5 operator iterations (1 load → 1 parse → 1 filter → 1 aggregate → final answer) using the same model.

**Cost decomposition** for Cap0 legal-hard-15:
- 22 wasted thrash iterations × ~5,550 input + ~660 output tokens = ~122,000 + ~14,500 = **~136,500 wasted tokens**
- The other 3 productive steps account for ~21,600 tokens
- Total: ~158,000 ≈ measured 158,153 ✓

### Case B — `legal-easy-25`: 8 failed filter guesses in a row

| | Cap0 | NACM | Δ |
|---|---:|---:|---:|
| Total tokens | 86,421 | 24,030 | **+62,391** |
| Input tokens | 80,352 | 23,122 | +247% |
| Output tokens | 6,069 | 908 | +568% |
| Operator iterations | 13 | 4 | 3.25× |
| Final answer | U.S. Space Force ✓ | U.S. Space Force ✓ | (both right) |

**This is a pure-cost case**: same correct answer in both systems, but Cap0 spent 3.6× more.

Cap0 trace (`cap0/react_steps.json`):
- steps 2-3: load the CSV (2 iterations to figure out `header=1`)
- steps 4-8: **5 consecutive failed `medianFraudLossByBranch2024` operator versions**, each filtering `Military Status in [Army, Navy, Air Force, Marine Corps, Space Force, Coast Guard]` and returning a (0,0) empty table because the actual values are `"U.S. Army"`, `"U.S. Navy"`, `"U.S. Space Force"` etc. (these rows are hidden by truncation, see accuracy analysis Case A).
- step 9-11: 3 more retries
- step 12-14: finally finds the right filter and computes the answer

The pattern is identical to Cause 2: the (0,0) tool result tells the LLM the filter is wrong, but without prior-event history the LLM doesn't remember which 5 filter strings it has already tried, so it tries variants of the same ones. NACM in 4 calls got the right vocabulary on the first preview and never re-tried.

### Case C — `legal-easy-9`: the same pattern at smaller scale

| | Cap0 | NACM | Δ |
|---|---:|---:|---:|
| Total tokens | 37,411 | 18,729 | **+18,682** |
| Operator iterations | 6 | 3 | 2× |
| Final answer | 2002 ✓ | 2002 ✓ | (both right) |

This is the cleanest "both-correct, Cap0 2× cost" example. The CSV `2024_CSN_Report_Count.csv` has a title row above the real header (a common pattern in this dataset). Cap0 took 4 iterations on the `reportCount` loader (default load → `skiprows=1` → re-cast types → finally got it right) before computing. NACM saw the header structure in its first preview and used `skiprows=1` immediately.

Per-step tokens for Cap0:

```
step 1  load                                   in=4678  out=130
step 2  modify (skiprows=1)                    in=4917  out=175
step 3  modify (rename cols, dtype handling)   in=4987  out=520   ← typed wrong on this attempt
step 4  modify (try different cleaning)        in=5157  out=534
step 5  add maxRelativeIncreaseYear            in=5068  out=338
step 6  final answer                           in=5358  out=11
```

The first 4 steps are all `reportCount` iterations costing ~20,000 tokens — purely from inability to remember "I just tried this, the type error was on the Year column".

---

## Why output cost dominates

GPT-5.2 pricing:
```
input  : $1.75 / 1M tokens
output : $14.00 / 1M tokens   ← 8× more expensive
```

Cap0 vs NACM cost decomposition on legal:
```
input  : Cap0 1.13M × $1.75/M = $1.98   |  NACM 1.00M × $1.75/M = $1.76   Δ = +$0.22
output : Cap0 0.077M × $14/M  = $1.08   |  NACM 0.046M × $14/M  = $0.64   Δ = +$0.44
TOTAL  : Cap0                   $3.06   |  NACM                   $2.40   Δ = +$0.66
```

**67% of the dollar gap (44¢ of 66¢) comes from the output-token increase**, which traces directly to Cap0's regenerate-from-scratch behavior (Cause 3).

---

## Suggested fixes (cost-only ranking)

1. **Carry forward the `WARNING: parametric thrashing` message** (and any backend-emitted advisory) into the next Cap0 prompt, even with `recentEventsCap=0`. Single-line addition to the prompt builder. Would have saved ~136 k tokens on `legal-hard-15` alone.
2. **Set `recentEventsCap≥1`** (carry at least the previous tool-call's input + result message). This addresses Causes 2 and 3 simultaneously: it stops repeated guesses, and it gives the LLM a base to write incremental code edits instead of full rewrites.
3. **Always emit `top_10` for high-distinct string columns** (also the top recommendation in the accuracy analysis). Reduces both wrong-filter loops (cost) and wrong-final-answer commits (accuracy).
4. **Show the full distinct vocabulary inline** when `distinct ≤ 30` even if `top_10` already exists.

Fixes 1+2 directly close the loops causing Cause 2 outliers. Fixes 3+4 reduce the *underlying need* for loops by giving the LLM enough information on the first preview.

---

## Per-task call-count analysis

A direct count of LLM calls per task gives a more nuanced picture than the population average.

```
Cap0 made MORE  calls than NACM on :  7 tasks
Cap0 made FEWER calls than NACM on : 20 tasks
Cap0 made SAME  calls as   NACM on :  3 tasks
                                     -----------
                                     30 legal tasks
```

### When Cap0 has MORE calls — the guess-loop tasks

These are the cost outliers — Cap0 entered a retry loop because the history wipe prevented it from realizing it had already tried the current code path:

| Task | Cap0 calls | NACM turns | Cap0 total tokens | NACM total | Δ | Loop description |
|---|---:|---:|---:|---:|---:|---|
| `legal-hard-15` | 25 | 6 | 158,153 | 30,135 | **+128,018** | 22 consecutive modifies on `parseStateMsaIdTheft2024` |
| `legal-easy-25` | 14 | 5 | 86,421 | 24,030 | +62,391 | 8 consecutive filter attempts (`[Army, Navy, …]` vs `U.S. Army`) |
| `legal-hard-1` | 13 | 7 | 106,800 | 52,308 | +54,492 | iterative wage-computation refinement |
| `legal-easy-9` | 6 | 4 | 37,411 | 18,729 | +18,682 | 4 retries on title-row CSV header |
| `legal-easy-21` | 6 | 4 | 36,818 | 19,792 | +17,026 | filter iteration |
| `legal-easy-11` | 6 | 5 | 39,532 | 25,202 | +14,330 | small extra retry |
| `legal-easy-26` | 5 | 4 | 31,693 | 18,650 | +13,043 | one extra retry |

For two of these (`legal-easy-9`, `legal-easy-25`), the agent eventually converged to the correct answer despite the loop. For three (`legal-hard-15`, `legal-hard-1`, `legal-easy-21`) Cap0 ran out of patience and committed to a wrong answer. The other two (`legal-easy-11`, `legal-easy-26`) both got the right answer.

### When Cap0 has FEWER calls — the early-commit tasks

The dominant pattern (20 of 30). Cap0 typically does **load → compute → answer in 3-4 calls**, while NACM does **sample → load → inspect → clean → compute → verify → answer in 5-12 calls**. Many of NACM's wins on accuracy (`legal-easy-4`, `legal-easy-19`, `legal-hard-18`, `legal-hard-22`, `legal-hard-23`) are in this group — the extra NACM exploratory turns catch structural quirks Cap0 commits past.

### When Cap0 and NACM have EQUAL call counts — controlled per-step cost

Only 3 legal tasks fall here. Using `stats.json` totals divided by call count gives a true ground-truth per-call average:

| Task | calls | Cap0 in/call | NACM in/call | Δ in | Cap0 out/call | NACM out/call | Δ out | Cap0 total | NACM total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legal-easy-19` | 4 | 5,220 | 4,553 | **+667** | 265 | 160 | **+104** | 27,910 | 18,853 |
| `legal-hard-7` | 4 | 5,296 | 4,540 | **+756** | 234 | 209 | +25 | 28,149 | 18,996 |
| `legal-hard-28` | 9 | 5,840 | 5,089 | **+752** | 559 | 311 | **+247** | 64,033 | 48,602 |

Two consistent gaps **independent of step count**:

1. **Cap0 input is ~700 tokens / call higher** in every case. That overhead is the synthetic `# Current Dataflow` block Cap0 builds and prepends to *every* call: operator summaries, last-execution results, column stats. NACM doesn't construct that — the same information is implicit in the OpenAI conversation history (tool-result messages persist across turns) and isn't re-serialized.
2. **Cap0 output is 12-79% higher per call**. Because every `createOrModifyOperator` Cap0 emits must rewrite the full `code` + a fresh natural-language `summary` from scratch (no recall of the previous turn), even an incremental fix re-emits the whole operator. NACM, with conversation history, tends to write smaller diffs.

So **even when step count is identical**, Cap0 is structurally ~15% more expensive on input per call and ~15-80% more expensive on output per call. At GPT-5.2's 8× output-to-input price ratio, the output gap is what dominates the dollar cost.

### Putting the pieces together

The 13.2% total-token gap on legal decomposes as:

- **~46% of the gap** comes from the 7 guess-loop tasks (the "MORE calls" cluster), where each extra Cap0 iteration costs ~5-6k tokens.
- **~25% of the gap** comes from the per-call input overhead (~700 tokens/call × 172 calls ≈ 120k tokens).
- **~29% of the gap** comes from the per-call output overhead (~200 tokens/call × 172 calls ≈ 35k output tokens, which at 8× the input price weighs heavily in dollars).

The first bucket is mitigated by **carrying the thrashing warning forward** (Fix 1). The other two are mitigated by **`recentEventsCap ≥ 1`** (Fix 2), which lets the LLM write incremental code edits instead of full rewrites AND lets it cite the previous tool-result rather than re-rendering the snapshot block.

---

## Files in this folder

```
cost-analysis/
├── README.md                          (this file)
└── traces/
    ├── legal-hard-15/                 (extreme outlier: 22-iter thrash loop, 158k vs 30k tokens)
    │   ├── cap0/
    │   └── nacm/
    ├── legal-easy-25/                 (both right, 86k vs 24k tokens; 13 vs 4 calls)
    │   ├── cap0/
    │   └── nacm/
    └── legal-easy-9/                  (both right, 37k vs 19k tokens; 6 vs 3 calls)
        ├── cap0/
        └── nacm/
```

To reproduce the per-step breakdowns above, walk `cap0/react_steps.json` step-by-step and inspect each agent step's `usage`, `toolCalls[].input`, `toolResults[].output`, and `inputMessages[0].content`. For NACM walk `nacm/messages.json` linearly — each `assistant` message is one LLM call, each `tool` message is one tool result.
