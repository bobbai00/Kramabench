# Why does C3 (Latest 1k + code) top the table? — it mostly doesn't

Date 2026-07-27. Question: C3 `Latest1kCodeInSnap` scores highest (68.7) at
near-anchor cost ($0.0140 vs $0.0139). Find the mechanism in the traces.

**Answer: the cost finding is real and explainable. The score finding does not
survive scrutiny.** C3 beats the anchor by a solid +5.0 points, but it is
statistically indistinguishable from C1 and C5, and the mechanism I proposed
from the aggregate statistics (code-in-snapshot) was falsified by every trace
dive that examined it.

---

## 1. The cost result — real, and it has a clean explanation

| arm | steps | input tok | cached | cache % | output tok | $/task |
|---|---|---|---|---|---|---|
| anchor Delta1k | 5.0 | 34,066 | 29,184 | 86.4% | 4,383 | 0.0108 |
| C1 Delta5k | 5.0 | 35,639 | 29,696 | 83.8% | 3,890 | 0.0101 |
| **C3 Latest1k+code** | **5.0** | **33,495** | **28,928** | **85.8%** | 4,493 | 0.0111 |
| C6 Latest1k+stats | 5.0 | 32,924 | 28,544 | 85.0% | 4,254 | 0.0104 |

(medians over 5 reps × 104 tasks; $ here is the raw median, not the trimmed
mean in the headline table.)

C3 adds a `Code:` block for every live operator **on top of** the `Result:`
block — verified in the raw prompts, it is additive, not a replacement — and
still ends up with *fewer* input tokens than the anchor. The reason is that the
two knobs move context in opposite directions:

- **`latest` removes the chronological event log.** Delta accumulates
  `# Agent Events`; latest replaces it with a `# Current Dataflow` snapshot of
  live operators only.
- **`code_in_snapshot` spends part of that saving back**, re-presenting each
  live operator's source every turn.

Net: a wash. Measured on final-step contexts (median over ~102 tasks/arm):

| arm | final ctx chars | code defs visible | live ops |
|---|---|---|---|
| anchor delta | 6,790 | 4 | 3 |
| C3 latest+code | **5,422** | 4 | 4 |
| C6 latest, no code | 5,188 | **0** | 3 |
| C1 delta 5k | 9,025 | 4 | 3 |

C3 shows the *same* code coverage as delta in 20% less context. The compression
comes from deduplication — on the 35 tasks per arm that edit at least one
operator, the anchor's final context carries **+3.0 code blocks beyond its live
operator count** (superseded versions still sitting in the event log), while C3
carries **+0.0**. Prompt growth per step is also flat (anchor 564 tok/step, C3
539, C6 493), so latest does not become cache-hostile at these trace lengths
(median 5 steps).

**This is a genuine, reusable finding: latest+code is a lossless compression of
delta's information, not an addition to it.**

---

## 2. The important correction: C6 is not a "latest without code" control

I initially attributed C3's win to code-in-snapshot by contrasting it with C6
(latest, code off). That contrast is invalid:

- **Delta already shows operator code**, inside its `# Agent Events` log.
  Measured `def load|transform|process(` counts in final contexts: anchor 4,
  C3 4, C1 4 — **C6 0**.
- So `enable_code_in_snapshot` does not add code that delta lacks. It
  *restores* under `latest` what delta had all along.
- C6 is therefore the deprived arm — the only config where the agent cannot see
  what its own operators do. That, not "latest", is why C6 is the worst config
  in the table (65.5 clean, worst floor, 44.2% flip rate).

My earlier decomposition ("latest alone −1.5, code +7.1") is void. The real
contrast between anchor and C3 is **how the same information is organized**:
a chronological log with stale versions, versus a deduplicated current-state
snapshot.

---

## 3. Trace dives — three tasks, three falsifications

Every dive was asked to find code-in-snapshot at work. None did.

### legal-easy-4 (C3 0.80, anchor 0.00, C1 1.00)
The dive first corrected the premise: C3 rep0 **fails**, identically to the
anchor (4391927). Passes are reps 1–4.

The cause is the **loader's header offset**, not the lever. Passing reps load
with `skiprows=3` / `header=3`, so real data starts at row 0 and the decisive
row `FTC - Web Reports (Fraud & Other)` lands at row 1 — inside the 1k window.
Failing reps use a bare `pd.read_csv`, so three junk rows consume the window
and that row sits at position 4, elided. Same cap, different offset.
Correlation across 10 traces is perfect: pass ⟺ header realigned. C3's own
failing rep used the bare read.

### biomedical-hard-1 (C3 1.00, anchor 0.20, C6 0.40)
Real failure mode found and it is shared: anchor and C6 map proteomics sample
columns `S001..S153` to metadata rows **by position** instead of via
`meta['idx']`. C3 keys on `idx` and gets gold.

That choice is made at **step 1**, by a richer preview operator that surfaced
`idx,S001,S002,...`. C3 reps 0–3 needed no self-correction at all. One genuine
code-in-snapshot effect does appear in rep 4 — a one-line edit
(`astype(float)` → `pd.to_numeric(..., errors='coerce')`) to code visible in the
snapshot — but it is one rep out of five, not the mechanism of the win.

### environment-hard-10 + legal-hard-29 (C3 1.00/0.80 vs C6 0.20/0.20)
Both decided at step 1–2 by load path, from contexts that are **byte-identical
between arms** (step-1 md5 `aef39bb9` shared by C3 and C6). In legal-hard-29 the
decisive line is `df['state'] = os.path.basename(p).rsplit('.',1)[0]` —
preserving per-file provenance instead of concatenating 52 files and
reconstructing state from the metro name. C3's rep 4 skipped that preview, used
the same heuristic as C6, and produced the same wrong answer.

The dive did find one narrow real effect, in environment-hard-10 reps 0/3: an
operator returned `0 rows, 0 cols` **without erroring**, its source was visible
in the snapshot, and the next step was a verbatim copy with a one-line
forward-fill fix. Without code-in-snapshot that source is shown only on
exceptions — so the lever's genuine niche is **silent failures**, where there is
no error block to carry the code.

### The losses are lever-independent too (legal-hard-16, biomedical-hard-3)
A fourth dive tested two hypotheses for where C3 *loses* — "latest forgets a
one-time fact" and "code crowds out data rows". Both were refuted by direct
counts, and both losses turned out to be the same parse lottery seen in the
wins.

- **legal-hard-16**: C3 answered Maine, gold Delaware. Its parser used
  `L.rsplit(',', 1)`, which splits *inside* a quoted thousands separator —
  `"…PA-NJ-DE-MD…","28,438"` became 438, collapsing Delaware from 0.96 to 0.60
  and handing the max to Maine (whose values are all under 1,000, so no commas).
  The anchor used `pd.read_csv` + `str.replace(',','')` and was immune. Note the
  inversion of the "latest forgets" story: the corruption markers appear **4
  times in the failing C3 context and 0 times in the anchor's**. The loser saw
  the quirk and misread it. Both arms coin-flip on this task (C3 1/5, anchor
  3/5, Fisher p≈0.5), and an anchor rep that used the same `rsplit` also failed.
- **biomedical-hard-3**: C3 stopped after 4 steps with a plausible wrong answer
  (a NaN-matching fallback silently returned mmc1 row 0). The anchor's cleaner
  merge returned **0 rows**, which forced 15 more probing steps and eventually
  gold. This is the "loud failure is a feature" pattern from
  NEVER_SOLVED_CEILING.md, not a context-knob effect.

That dive independently confirmed the §2 correction: "delta renders code too,
inside the action record — `enable_code_in_snapshot=false` does not remove code
from delta contexts."

---

## 4. What the process metrics say (3 judged reps, ±pop std)

| arm | M10 useful | M10 wrong_param | M7 |
|---|---|---|---|
| anchor | 0.751 ± 0.003 | 0.187 ± 0.005 | 0.691 |
| C1 | 0.751 ± 0.012 | 0.193 ± 0.004 | 0.711 |
| C5 | 0.767 ± 0.024 | 0.178 ± 0.013 | — |
| **C3** | **0.782 ± 0.012** | **0.158 ± 0.008** | **0.725** |
| C6 | 0.739 ± 0.013 | **0.212 ± 0.021** | 0.699 |

C3 has the lowest wrong-parameter rate by ~3σ, and the per-task correlation
between score gain and wrong_param reduction is −0.399 (tasks C3 wins:
wrong_param −0.083; tasks it loses: +0.077). So *something* about C3 reduces
decision errors, consistently. The traces say that something is a better
step-1 load path, not the agent re-reading its own code — but the dives covered
4 tasks, and this statistic covers 300 runs. **The process signal is real and
currently unexplained.**

---

## 5. Measurement contamination found along the way

Two classes, 54 cells, both scoring infrastructure failures as wrong answers:

- **48 cells have `evaluation.json` but no `answer.json`** — the run produced
  nothing (in `biomedical-hard-1` anchor reps 0/1 the directory holds only
  `config.json`, `prompt.txt`, `evaluation.json`; no trace, no stats). They
  carry a stale `success: 0`. Concentrated in `astronomy-hard-11` (19),
  `biomedical-hard-1` (13), `astronomy-hard-7` (7) — note astronomy-hard-11 is
  the known manifest-typo benchmark defect.
- A fifth instance surfaced in the losses dive (`biomedical-hard-3`, C3 rep4:
  0.0041s, 0 tokens, config+evaluation only), confirming the pattern is
  cross-arm and not specific to the anchor.
- **6 cells whose answer is a litellm error string** — `"Available Model Group
  Fallbacks=None"`, 209 tokens, 21s — scored 0.

Not contamination: the 131 `"No response from agent"` cells. Those ran a median
85s / 39k tokens / 5 steps. The agent worked and failed to emit an answer; 0 is
the right score.

---

## 6. The verdict on the original question

Dropping the 9 tasks touched by contaminated cells leaves 95 matched tasks:

| arm | clean score ± std |
|---|---|
| **C3 Latest1k+code** | **70.4 ± 3.6** |
| C1 Delta5k | 69.4 ± 2.5 |
| C5 DeltaStats2k | 69.3 ± 2.2 |
| C4 / C7 | 68.7 |
| C2 | 66.6 |
| anchor / C6 | 65.5 |

Paired bootstrap over tasks (4,000 resamples):

| comparison | delta | P(C3 better) |
|---|---|---|
| C3 − anchor | +5.0 | **1.00** |
| C3 − C7 | +1.7 | 0.87 |
| C3 − C5 | +1.1 | 0.70 |
| C3 − C1 | +1.0 | 0.71 |

**C3 is reliably better than the anchor and indistinguishable from C1 and C5.**
"C3 is the top config" is the ranking of a three-way tie. The honest statement
is that three different routes — more rows (C1), rows+stats (C5), and
latest+code (C3) — all buy about the same +4 to +5 over the 1k schema-only
anchor, and the mechanism dives suggest a common cause worth testing directly:
**all three change what the agent sees at step 1, and step 1 is where these
tasks are decided** (29/31 coin divergences occur at the first authored
operator, per HEADROOM_STUDY.md §5).

### What is actually worth taking from C3
1. **Latest+code is free compression.** Same code coverage as delta, 20% less
   context, no stale versions, flat per-step growth. Adopt it for the context
   budget, not for an accuracy claim.
2. **Never run latest without code (C6).** It is the only config that hides
   operator source, it has the worst wrong_param (0.212) and the worst flip
   rate (44.2%), and it costs the most on long traces.
3. **The lever's real niche is silent failures** — an operator returning 0 rows
   with no exception. Delta shows code on error blocks; only code-in-snapshot
   shows it when nothing threw.

### Open, and cheap to close
- Re-run the 54 contaminated cells; they are concentrated in 9 tasks.
- The wrong_param gap (0.158 vs 0.187, ~3σ over 300 runs) has no trace-level
  explanation yet. Four dives is a small sample against it — a targeted sweep
  of the 17 "code-specific" win tasks would settle whether there is a real
  mechanism or whether the aggregate is load-path luck reshuffled.
- `latest + code + stats` was never run; it is the missing cell of the design.

---

### Provenance
Statistics: `judgment_runs/mini_star/final_table.py` and this session's ad-hoc
scans over `system_scratch/*/*/{react_steps,stats,evaluation,answer}.json`.
Context measurements read the agent's actual `inputMessages` at the final step.
Dives: 4 completed trace-pair investigations covering 6 tasks — legal-easy-4,
biomedical-hard-1, environment-hard-10, legal-hard-29 (wins) and
legal-hard-16, biomedical-hard-3 (losses). Every one attributed the outcome to
the step-1/2 load-or-parse choice rather than to the context lever.
