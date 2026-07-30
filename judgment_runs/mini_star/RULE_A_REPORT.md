# Rule A — "rich source, lean interior"

**Run:** 2026-07-28 14:25–15:40. 6 arms × 20 hard tasks = 120 runs, single-shot, no retries.
**Service:** `:3002` @ `4af1e98da`, `src_dirty=False` on every task's `config.json`, worktree clean at
the same SHA after the run ⇒ **single vintage start to finish.**
**Arms:** `A1RolePolicyReplicate{1,2,3}` (policy) vs `A0ControlReplicate{1,2,3}` (control). Same model
(gpt-5-mini), same LATEST+code core, same sampling cap. The *only* difference is the per-operator
render policy.

Reproduce: `.venv/bin/python judgment_runs/mini_star/ruleA_analyze.py`

---

## 1. The manipulation landed

Verified two ways: on a shared task (`archeology-hard-7`) and corpus-wide. Per rendered operator block:

| arm | src rows | src stats | int rows | int stats |
|---|---|---|---|---|
| A1 policy | 8.7 | 86% | 6.5 | 0% |
| A0 control | 19.6 | 0% | 11.7 | 0% |
| delta | **−10.9** | **+86pp** | **−5.3** | 0 |

`role_policy_config` populated in A1's `config.json`, `None` in A0's. Both legs bind — the row cap on
sources *and* interiors, plus the stats block on sources only. Not a no-op (unlike the first
`role === "source"` version, which silently matched nothing).

## 2. Result

| arm | accuracy | cache% | reasoning/task | uncached-in | $/task | steps |
|---|---|---|---|---|---|---|
| A1 rich-source/lean-interior | **67.3 ± 8.5** | 83.9 | 6,559 | 10,311 | 0.0217 | 6.8 |
| A0 control (uniform render) | **62.3 ± 7.7** | 84.7 | 6,893 | 9,772 | 0.0210 | 7.0 |

Deltas: accuracy **+5.0 pt**, cost **+3.3%**, input +1,344 tok/task, output +273, reasoning **−334**,
cache −0.7pp, steps −0.2.

Per-rep: A1 `79.0 / 64.0 / 59.0`, A0 `69.0 / 66.5 / 51.5`.

## 3. Verdict: directionally positive, NOT ESTABLISHED

**+5.0 pt sits inside the noise floor.** Pooled rep std is 8.5 and the independently measured
run-level randomness floor is ±4–5 pt (VARIANCE_REPORT.md). Three reps cannot resolve a 5-point
difference against a 20-point rep spread.

Worse for the claim: the entire delta is **two tasks**. Only `astronomy-hard-8` and
`biomedical-hard-8` moved ≥0.5 (both +0.67, both 0.00 → 0.67); nothing regressed. 2 × 0.67 / 20 =
+6.7 pt, i.e. the two flips *are* the headline number. That is ~6 run-level events carrying the
result.

**What is solid:** Rule A is **cheap** — +3.3% cost, and it *reduces* reasoning (−334) and steps
(−0.2) rather than inflating them. Every Rule B memory channel was accuracy-neutral; this is the
first overlay that is directionally positive at near-zero cost. That earns more reps, not adoption.

### Correction to the mid-run read
At 113/120 the preview said +3.8 pt for **+13.8%** cost and concluded NEGATIVE, with the story
"source stats costs more than the rows it saved." Completing `A0ControlReplicate3` moved the control's
own cost up and the true gap to **+3.3%**. The stats block is *largely paid for* by the row capping.
The preview conclusion was an artifact of an incomplete control arm — recorded here because the
partial-data trap has now bitten this project three times (also the `empty=20` probe and the
"control reasons 20% less" claim, both retracted the same day).

### Denominator note
`orchestratorA.sh`'s `overall()` prints `compute_scores.py` at **104-task scope** (15.2% / 12.3% /
11.3% vs 13.3% / 12.8% / 9.9%), where the 84 non-subset tasks are stub zeros. Rescaled by 104/20
those reproduce the subset means above. The report uses subset-scoped numbers; the 104-scoped figures
in `orchA_progress.log` are **not** accuracy.

---

## 4. Next test: A2 — row-capping without stats

Rule A bundles two levers. Row-capping saves bytes; the source stats block spends them and stats has
never paid in any prior arm (C2, C4, C5, C6). Split them:

```
sourceSampleRows: 12, sourceStats: FALSE, sourceStructuralHints: true,
interiorSampleRows: 3, interiorStats: false, leanTerminal: true
```

If A2 keeps the +5 and drops the +3.3%, the gain is layout, not statistics. If A2 loses it, the stats
block is doing the work and is worth its price. Either outcome is decisive; the current bundle is not.

**Also raise the rep count.** To resolve ±5 pt against a ±4–5 pt floor needs ~6–8 reps per arm, not 3.
A1 vs A0 vs A2 at 6 reps = 360 runs ≈ 3.5 h at P5. This is the one overlay where that spend is
justified.

---

## 5. What to combine — and what the traces refuse

### Rule A + one memory channel
Rule A shapes *what* each operator renders; Rule B tested *how much history* it renders. The Rule B
result: no channel bought accuracy, but per-operator **data history** was cheapest (−10.4% cost) and
most stable (std 1.9), and code history was cache-safe (+2.0pp). Combine Rule A's row-capping with
**data history**, which is byte-cheap precisely because Rule A has already capped the rows each
version renders.

**Never thought replay.** Measured: cache 82.8 → 48.4% (−34pp), uncached-in +27,395 tok, cost +31%,
and reasoning only −80 — it does not even buy back the tax it imposes.

### The "important version" question is answered NO — for both code and data

Asked whether an operator's *important version* (error version, peak version, superseded version)
should be retained. Measured across 1,028 LATEST+code runs:

- **Data versions have no history to keep.** Row counts change in **11 operator-lifetimes out of
  thousands** across 517 runs; **0 of 1,028** runs destroyed a good result with a later edit.
  Operator results are effectively write-once. This is *why* B2 data history was cheap and
  accuracy-neutral — it was rendering near-duplicates.
- **Code rebuilds are a symptom, not a retention failure.** 7.0% of runs rewrite a superseded
  version (171 events). Raw signal looks huge (−48 pt when the old version was *evicted*), but
  within-task the median is **+0.0 pt**, the bucket is 97–100% hard, 16.1 steps vs 5.9, and it
  clusters in `wildfire-hard-19` + `astronomy-hard-7/9/11/12` — **the same cluster the error analysis
  hit, for the third time.** Also, "still visible" rebuilds score nearly as badly (−25 pt) as evicted
  ones, so visibility is not the discriminator.

### What agents actually rebuild: source-boundary facts

Reading the rebuilt code, not the counts:

- `astronomy-hard-11`: the same `process(omni2_raw, swarm_raw)` body copied onto **three** operator
  ids; the only diff between versions is *which column it guesses is F10.7 vs daily Ap*.
- `wildfire-hard-21`: `zhvi` → `zhvi_load`, differing by exactly `low_memory=False`.

Neither is a lost version. Both are a durable property of the file. Sizing that surface — runs needing
≥1 non-default read argument:

| fact | runs |
|---|---|
| `encoding=` | 31.4% |
| `skiprows`/`header` | 22.7% |
| `low_memory` / mixed dtypes | 7.0% |
| `sep`/`delimiter` | 7.0% |
| **any** | **47%** |

Against error-version retention at 3.1% and the failure-fact ledger at 6.0%, the fact surface is
8–15× larger.

**So importance belongs on the per-source load recipe + column identity — keyed per source operator,
immutable once discovered, rendered inside the already-stable source block (cache-safe, unlike
replay). Not on versions of anything.**

Gap in the current prototype: `low_memory`/`DtypeWarning` **never raises**, so the exception-only
`failure_ledger.py` misses that 7%. The ledger must absorb load-quality facts, or Rule A's
`sourceStructuralHints` must carry them from the data side — it already fires on 86% of source blocks,
which makes it the cheaper carrier.

---

## 6. Trace health

All six arms: `instant-fails=0`, `empty=0`, `No response from agent=0`, quota errors `=0`.
Steps 6.6–7.7 (max 26), cache 80.2–86.3%, reasoning 5,750–6,598/task. Services flat across the run
(:3001 233 MB, :3002 172–188 MB), no leak, load ≤1.1.
