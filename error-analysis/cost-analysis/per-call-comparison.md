# Per-call cost comparison — controlled by step count

Follow-up to `README.md`. This document answers two questions:

1. Why do some tasks have *more* Cap0 LLM calls than NACM (e.g. `legal-easy-9`, `legal-easy-25`)?
2. For tasks where the call count is the *same*, what is the per-step input/output token cost?

Both questions use ground-truth data: per-step `usage` blocks in `cap0/react_steps.json` and per-task totals in `nacm/stats.json` divided by the assistant-turn count from `nacm/messages.json`.

---

## 1. Per-task call-count classification

Across 30 legal tasks:

```
Cap0 made MORE  calls than NACM on :  7 tasks  (guess-loop pattern)
Cap0 made FEWER calls than NACM on : 20 tasks  (early-commit pattern, dominant)
Cap0 made SAME  calls as   NACM on :  3 tasks  (controlled-comparison group)
                                     -----------
                                     30 legal tasks
```

The population averages (`Cap0 5.73 < NACM 6.63` calls/task) come from the dominant **early-commit** group — Cap0 typically wraps up in 3-4 calls while NACM spends 5-12 calls exploring before computing. The **guess-loop** group is the source of the extreme cost outliers.

### "MORE calls" group (7 tasks): always a history-wipe loop

| Task | Cap0 calls | NACM turns | Cap0 total | NACM total | Δ | Loop description |
|---|---:|---:|---:|---:|---:|---|
| `legal-hard-15` | 25 | 6 | 158,153 | 30,135 | **+128,018** | 22 consecutive modifies on `parseStateMsaIdTheft2024`; backend warned "parametric thrashing" but the warning never reached the next LLM input |
| `legal-easy-25` | 14 | 5 | 86,421 | 24,030 | +62,391 | 8 consecutive `medianFraudLossByBranch2024` versions with filter `[Army, Navy, …]` instead of `U.S. Army`/`U.S. Navy`/`U.S. Space Force` — all returned (0,0) empty tables |
| `legal-hard-1` | 13 | 7 | 106,800 | 52,308 | +54,492 | iterative wage-computation refinement |
| `legal-easy-9` | 6 | 4 | 37,411 | 18,729 | +18,682 | 4 retries on title-row CSV (`reportCount` loader) — kept trying `skiprows=1` variants because it didn't remember just trying them |
| `legal-easy-21` | 6 | 4 | 36,818 | 19,792 | +17,026 | filter iteration |
| `legal-easy-11` | 6 | 5 | 39,532 | 25,202 | +14,330 | small extra retry |
| `legal-easy-26` | 5 | 4 | 31,693 | 18,650 | +13,043 | one extra retry |

**Pattern**: In every one of these tasks, Cap0 re-tries variations of the same approach because the previous attempt's failure (or its own previous code) is invisible at the next LLM call. The thrashing-detection warning the backend emits is dropped on the floor.

### "FEWER calls" group (20 tasks): early commit, sometimes wrong

The dominant pattern. Cap0's typical legal-task trace is:

```
step 1 — createOrModifyOperator (load)
step 2 — createOrModifyOperator (compute)
step 3 — final answer
```

NACM's typical trace adds 2-4 exploratory turns:

```
step 1 — createOrModifyOperator (sample head+tail)
step 2 — tool result with column stats showing structure
step 3 — createOrModifyOperator (inspect with header=None)
step 4 — tool result revealing hidden header row
step 5 — createOrModifyOperator (proper load)
step 6 — tool result confirming shape
step 7 — createOrModifyOperator (compute)
step 8 — tool result
step 9 — final answer
```

Cap0 saves 4-6 calls on these tasks **but** also catches none of the structural traps the NACM exploration would have caught. Most of the accuracy regressions (`legal-easy-4`, `legal-hard-18`, `legal-hard-22`, `legal-hard-23`, …) live here.

---

## 2. Same-call-count tasks: controlled per-step cost

Only 3 of the 30 legal tasks have identical Cap0 and NACM call counts. They give a controlled view of per-step cost without confounding by step-count differences.

| Task | calls | Cap0 in/call | NACM in/call | Δ in | Cap0 out/call | NACM out/call | Δ out | Cap0 total | NACM total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `legal-easy-19` | 4 = 4 | 5,220 | 4,553 | **+667 (+14.6%)** | 265 | 160 | **+104 (+65%)** | 27,910 | 18,853 |
| `legal-hard-7` | 4 = 4 | 5,296 | 4,540 | **+756 (+16.7%)** | 234 | 209 | +25 (+12%) | 28,149 | 18,996 |
| `legal-hard-28` | 9 = 9 | 5,840 | 5,089 | **+752 (+14.8%)** | 559 | 311 | **+247 (+79%)** | 64,033 | 48,602 |

Method: `cap0/react_steps.json` stamps `usage.inputTokens` / `usage.outputTokens` per agent step; the per-call values above are the mean across those agent steps. NACM has no per-call usage stamp, so per-call values are `stats.json` totals divided by the number of `assistant`-role messages in `messages.json`.

### What the +700 input-tokens-per-call overhead is

The synthetic block Cap0 prepends to every LLM call (visible in any `cap0/react_steps.json` step at `inputMessages[0].content`):

```
# Recent Agent Events (last 0 of N events)
(... N earlier events omitted to keep context focused on recent work ...)

# Current Dataflow
## Operators

### Operator `op1` (DataLoading)
Summary: <multi-sentence natural-language description of what the code does>
Result:
  Executed operator op1
  Output table shape: (N, M)
  <head 4 + tail 2 rows>
  Column stats:
  - "col1" (str): null=..., distinct=..., top_10={...}
  - "col2" (numeric): null=..., mean=..., min=..., max=...

### Operator `op2` (DataProcessing)
...

## Links
- op1 → op2
```

This block is **rebuilt and re-serialized every turn**. Its size grows as more operators accumulate (the `inputMessages[0]` content in legal-hard-15 grew from 730 chars at step 1 to 3,870 chars at step 19).

NACM doesn't construct this. The dataflow's per-operator state is carried *implicitly* via the conversation history of tool-result messages, which the OpenAI API caches across turns. So:

- Cap0 input/call = static system prompt (~4,500 tok) + snapshot block (~700 tok regenerated each call)
- NACM input/call = static system prompt (~4,500 tok) + cumulative conversation (small early, larger late)

For short tasks (4 calls), NACM's cumulative history hasn't grown much, so its per-call input runs ~600-750 tokens *below* Cap0. For longer tasks (9+ calls), NACM eventually catches up because its history is larger than any single Cap0 snapshot — but Cap0 still loses on the output side.

### What the +12% to +79% output-tokens-per-call overhead is

Every Cap0 `createOrModifyOperator` invocation has three required fields:

- `operatorId` (short string)
- `code` (full pandas function body)
- `summary` (multi-sentence natural-language description)

Because the LLM cannot see its prior turn, **it cannot make an incremental edit**: each call re-emits the full code from scratch + a fresh prose summary. NACM, with conversation history, can write smaller diffs in subsequent turns.

Concrete: `legal-hard-15` shows this in extremis. Cap0 step 5 has `toolcall_chars=2,176` (~500-600 tokens of code generation). NACM step 5 of the same task has `toolcall_chars=355` (a small incremental edit, ~100 tokens).

---

## Dollar breakdown of the +13.2% total-token gap

Using GPT-5.2 pricing ($1.75/M input, $14.00/M output) and the 30-task legal totals:

| Component | Cap0 tokens | NACM tokens | Δ tokens | Δ $ |
|---|---:|---:|---:|---:|
| Input | 1,133,659 | 1,004,694 | +128,965 | +$0.22 |
| Output | 76,785 | 45,671 | +31,114 | **+$0.44** |
| **Total** | 1,210,444 | 1,050,365 | +160,079 | **+$0.66 (+21.6%)** |

Approximately:

```
~46% of the token gap : 7 guess-loop tasks       (history wipe + thrashing warnings dropped)
~25% of the token gap : per-call input overhead  (~700 tok/call × 172 calls ≈ 120k)
~29% of the token gap : per-call output overhead (~200 tok/call × 172 calls ≈ 34k output ≈ 270k input-equivalent at 8× price)
```

The **first bucket** is closed by carrying the backend's thrashing warning forward into the next prompt even when `recentEventsCap=0`.

The **second and third** are both closed by `recentEventsCap ≥ 1`: keeping the previous tool-call's code + tool-result message in context lets the LLM (a) skip the synthetic snapshot block reconstruction and (b) emit incremental code edits instead of full rewrites.

---

## Traces in this folder relevant to per-call comparison

```
traces/
├── legal-easy-19/   (4=4 calls, +667 in/call, +104 out/call — Cap0 wrong on accuracy too)
├── legal-hard-7/    (4=4 calls, +756 in/call, +25 out/call  — both right)
├── legal-hard-28/   (9=9 calls, +752 in/call, +247 out/call — both right)
├── legal-hard-15/   (25 vs 6 calls — guess-loop extreme; from main README)
├── legal-easy-25/   (14 vs 5 calls — guess-loop, both right; from main README)
└── legal-easy-9/    (6 vs 4 calls — guess-loop, both right; from main README)
```

To verify the per-call numbers above directly:

```bash
# Cap0 per-step input tokens (legal-easy-19)
python3 -c "
import json
steps = json.load(open('traces/legal-easy-19/cap0/react_steps.json'))['steps']
for s in steps:
    if s.get('role') == 'agent' and s.get('usage'):
        u = s['usage']
        print(f\"  in={u['inputTokens']:>5}  out={u['outputTokens']:>5}\")
"

# NACM per-call (use stats.json total / assistant-turn count)
python3 -c "
import json
stats = json.load(open('traces/legal-easy-19/nacm/stats.json'))
msgs = json.load(open('traces/legal-easy-19/nacm/messages.json'))
n_calls = sum(1 for m in msgs if m['role'] == 'assistant')
print(f\"  in/call ≈ {stats['input_tokens']/n_calls:.0f}\")
print(f\"  out/call ≈ {stats['output_tokens']/n_calls:.0f}\")
"
```
