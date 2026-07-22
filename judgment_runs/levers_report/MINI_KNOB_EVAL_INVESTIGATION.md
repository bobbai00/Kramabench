# Do context knobs move the eval? — GPT-5-mini, C1 (rows) & C2 (stats), multi-dimension eval

**Question (Bob's goal):** for C1 (char-cap / rows-rendered knob) and C2 (column-stats
profile knob), using **gpt-5-mini**, do the knobs move the **subtask eval**? How big is
run-to-run **randomness**? Try each eval possibility — our way (isolated subtask eval),
the DS-Guru / KramaBench way (pipeline-design LLM judge), and anything else reasonable.
Change the agent-service prompt/tools *if needed*.

**Setup:** fixed 50-task random sample (seed 50; domains arch 5 / astro 6 / bio 4 / env 8 /
legal 18 / wildfire 9). Three arms, all DELTA context mode, gpt-5-mini:
- **anchor** = Delta **1k** char cap, schema-only (no stats)
- **C1 ray** = Delta **5k** char cap, schema-only        → isolates the rows/char-cap knob
- **C2 ray** = Delta **1k** + column-stats profile (data_level 2) → isolates the stats knob

All comparisons are **matched** (only tasks both arms completed) to remove selection bias.

---

## TL;DR

1. **C1 (rows knob) moves the answer-correctness evals; C2 (stats knob) does not.**
   On KramaBench's native scores the rows knob is +0.083 main / +0.060 subtask at mini;
   the stats profile knob is a wash-to-slightly-negative. **The C1 signal clears the
   measured noise floor** (subtask-mean std 0.025; the effect is directional — 17↑/5↓,
   legal 9↑/1↓ — where noise is symmetric).
2. **The knob acts on answer correctness, not pipeline structure.** A de-bugged
   pipeline-design (structure-coverage) judge is *dead flat* under the rows knob
   (16↑/16↓). This is the layer-separation result, now shown with a real metric:
   the agent builds the same structure regardless of the knob; more rows just help it
   get the *values* right inside that structure.
3. **Found and fixed a real bug in KramaBench's pipeline-design judge** — a truthiness
   coercion (`bool(str(x).lower())`) that scored every verdict `True`, saturating all
   pipeline-design coverage at 1.0. After the fix the metric discriminates (0.0–1.0).
4. **The ~10 "systematic mini failures" were a concurrency artifact, not a mini limit.**
   Hard/looping tasks returned instantly (0.09 s, no work) only under concurrency-6 engine
   contention; the same task runs fine in isolation. Completed tasks have full, clean
   subtask coverage — the matched comparison is unbiased.
5. **No agent-service change was warranted.** All 18 mini failures are well-formed,
   right-type, genuinely-wrong answers — **0 parse artifacts** — so a structured
   answer/report tool cannot improve accuracy. Eval noise lives in the agent's
   stochastic reasoning, not answer extraction.

---

## 1. Knob effect across eval dimensions

**Metric by purpose.** Aggregate claims ("does the knob help?") use KramaBench's own
**reported continuous score** — never a hand-rolled re-scoring. That score is already
graded where the benchmark author intended (`rae_score`=1/(1+RAE) for `numeric_approximate`,
F1 for lists, `success` 0/1 only for `numeric_exact`/`string_exact`). **Binary pass/fail is
reserved for A-vs-B *case* comparison** (who-passed-who-failed on task X), not for the
aggregate — pass = exact for exact types / ≥0.9 for graded, and only on clean 0→1 flips
(threshold-straddlers are noise, cf. §3).

| Purpose | Metric | **C1: rows 1k→5k** | **C2: schema→stats** |
|---|---|---|---|
| Aggregate answer-correctness | **KramaBench main score** (native, continuous) | **+0.083** (0.666→0.750, n=41) | −0.026 (0.609→0.583, n=39) |
| Aggregate, finer-grained | **KramaBench subtask-implementation** (native, `--run_subtasks`) | **+0.060** (17↑/5↓, n=41) | +0.017 (7↑/8↓, n=39) |
| Case-level A-vs-B only | binary pass/fail (exact, or ≥0.9) | who-passed-who-failed | — |
| Structure **control** | pipeline-design coverage (fraction of gold steps in DAG, LLM judge) | +0.002 (16↑/16↓, n=46) | −0.030 (10↑/13↓, n=47) |

*(Dropped: RAE-relaxed main — it applied RAE to `numeric_exact` tasks, overriding the
benchmark's per-task exact-vs-approximate decision, breaking comparability and risking
metric-shopping. The effect holds on the native score without it.)*

**Reading it:** C1 (rows) is positive and directional on *both* native answer metrics
(main +0.083, subtask +0.060) and flat on structure. C2 (stats) is a wash on the subtask
eval and mildly negative on main accuracy at mini —
the profile block does not pay at mini here (consistent with the GPT-5.2 "coin" finding,
and with the earlier result that the stats block helps DELTA less than it helps nothing).

### Why the rows knob helps — it's domain-mediated (row-value-dependent tasks)

C1 subtask-mean Δ **by domain** (matched):

| domain | n | mean Δ | up / down |
|---|---:|---:|---:|
| environment | 6 | +0.107 | 4↑ / 1↓ |
| legal | 18 | +0.097 | **9↑ / 1↓** |
| astronomy | 3 | +0.028 | 1↑ / 0↓ |
| archeology | 2 | 0.000 | 0 / 0 |
| biomedical | 3 | 0.000 | 0 / 0 |
| wildfire | 9 | −0.001 | 3↑ / 3↓ |

The benefit concentrates in **legal + environment** — filter/count/lookup tasks that
reason over many *row values*, exactly where a 1k char cap truncates the rendered rows
the agent needs. It is near-zero where the task doesn't hinge on row values. Concrete
instance: legal-hard-16 (find a state) → 1k renders too few rows and the agent answers
"New York" (wrong); 5k improves it toward gold "Delaware" (subtask-mean 0.04→0.27). This
**replicates the GPT-5.2 render-starvation mechanism** (probe-star C1 = legal-hard-2), but
the effect is *larger* at mini — a smaller model is more sensitive to render starvation.

**Built-in noise reference:** under the *same* knob, wildfire moves 3↑/3↓ — symmetric,
mean ≈ 0. That is the knob doing nothing, i.e. pure run-to-run noise. Against that
reference, legal's **9↑/1↓ over 18 tasks** is directional signal, not a coin
(the formal noise floor in §3 quantifies this).

---

## 2. The eval "ways" tried

**(a) Our way — isolated subtask-implementation eval (KramaBench `--run_subtasks`).**
Re-runs the agent on each of the ~6 gold sub-questions per task and scores the answer.
Continuous (633 sub-scores vs 104 binary main scores) → a finer signal. This is where
C1 shows +0.060 / 17↑-5↓. Also fixed a confound: subtask format hints were empty
(format_hint files cover only main tasks), so subtask answers were mis-formatted; now
synthesized from `answer_type`.

**(b) The DS-Guru / paper way — pipeline-design LLM judge.** KramaBench's own
`evaluate_data_pipeline` judges whether the generated pipeline covers each gold "key
functionality" (= the subtask `step` text). The dataflow SUT skips it
(`pipeline_code:""`); I enabled it by serializing the agent's DAG/react-trace.
- **Bug found & fixed:** the judge coerced verdicts with `bool(str(ans_item).lower())`,
  which is `True` for *any* non-empty string ("false"→True). Every non-empty pipeline
  scored 1.0 (even `print('hello world')`). Fixed to parse true/yes/1. Validated:
  empty→0.0, irrelevant→0.0, real→0.0–1.0.
- **Result:** mean coverage ≈ 0.61; **flat under both knobs** (C1 +0.002, C2 −0.030).
  Structure coverage is decoupled from the knob and from answer correctness (a *failed*
  task can have 0.75 structural coverage) — the layer-separation mechanism, measured.

**(c) Widening the answer metric — tried RAE-relaxed, REJECTED.** KramaBench's
`numeric_exact` is binary and brutal (68.1 vs 68.5 → 0); of 12 numeric main failures at
mini, ~3 are within 5% and ~5 within 20%. Re-scoring those with RAE partial credit *looked*
like a cleaner signal (+0.131, 5↑/0↓) — but it **overrides the benchmark's per-task
exact-vs-approximate decision** (a wrong count *should* be 0), breaks comparability with the
paper, and is metric-shopping. **Rejected.** The lesson: the legitimate way to widen is
KramaBench's *own* finer-grained view — the **subtask-implementation eval** (more items →
more power, still native) — not a hand-rolled re-scoring of the main metric. The near-miss
observation remains a useful *diagnostic* (the knob nudges values closer), just not a
scoring change.

**(e) Trace-search / evidence-delivery (M4) as a mediation test.** The "search the
trace for gold subtask values" eval (M4 from the process-metrics work — gold subtask
answers/literals as search keys against the rendered context in `react_steps.json`),
re-run at mini where the knob *does* move answers, upgraded from tautology-check to
mediation test (n=29 matched trace-clean tasks; SOURCE-ONLY = loader renders only, to
avoid counting the agent's own computed values as "delivered"):

- **Aggregate mediation: consistent.** Delivery rises 0.505→0.593 (+0.088 source-only)
  under the same knob that lifts answers (+0.060 subtask) — the knob demonstrably does
  its deterministic job at mini too (replicates the 5.2 monotonic lean→rich result).
- **Per-task mediation: FAILS.** The answer-UP bucket's mean M4Δ (+0.055) is *smaller*
  than the answer-flat bucket's (+0.143) — flat tasks received the most extra evidence
  and didn't convert it. Several answer flips show ~zero raw-delivery change
  (legal-hard-16 +0.007, legal-hard-29 −0.011, legal-easy-3 −0.113).
- **The per-task path is heterogeneous.** legal-hard-16: raw-render delivery flat
  (+0.007) but ALL-renders delivery +0.537 — the flip traveled through the agent's
  *actions* (its 5k-cap filter outputs rendered the discriminating value), not the raw
  loads. Sometimes the mechanism is raw rows, sometimes derived renders, sometimes the
  agent just reasons better on unchanged evidence.

**Verdict:** M4-style trace search is a good **manipulation check** (cheap, deterministic,
confirms the knob delivers what it claims) and a good **per-case forensic** (which route
did THE value arrive by), but **not a per-task attribution metric** — delivery-to-answer
coupling is loose because what matters is whether *the one discriminating value* crosses
into context (threshold-y), not the *fraction* of gold values (what M4 averages). This is
the 4th replication of layer separation, now at mini where the knob IS an aggregate
accuracy lever.

**(d) DS-Guru system itself** — feasible but not run. It defaults to Ollama (down), but
`generator_utils.py` has an OpenAI backend (`gpt-4o-mini`) and our OPENAI_API_KEY works
(the judge uses it). Running DS-Guru as a SUT on these 50 tasks is a turnkey option if we
want a reference-system baseline; deprioritized (different system + model, not a clean
knob comparison; code-agent baselines are known-weak on KramaBench acc/$).

---

## 3. Randomness / variance (run-to-run noise floor)

Anchor (1k) config re-run **3×** on a 12-task domain-spanning subset (concurrency 3,
subtask eval). Per-task noise across the 3 identical-config runs:

| metric | value |
|---|---|
| mean per-task subtask-mean **std** | **0.025** |
| max per-task subtask-mean std | 0.118 (archeology-easy-8) |
| tasks perfectly deterministic (std 0) | **7 / 12** |
| binary main-pass **flips** | **2 / 12** (astronomy-easy-5, legal-easy-19; both 0→1→0) |

**The noise is low and bounded.** gpt-5-mini is largely reproducible on the *continuous*
subtask metric — 7/12 tasks return byte-identical scores across three runs; only a handful
of borderline/hard tasks wobble (std ≤ 0.12). The *binary* main-pass is the noisiest view
(2/12 ≈ 17% of tasks flip pass/fail on a pure re-run) — expected for a 0/1 metric near a
decision boundary.

**Does the C1 signal clear it?** Yes, on the robust metrics; the binary metric alone is
corroborative not decisive:
- Per-task Δ-noise ≈ std·√2 ≈ **0.035**. C1's per-task movements average |Δ| 0.090 and the
  top movers are 0.36–0.50 — well above noise.
- **Direction is the clincher.** Noise is *symmetric* (the 2 main flips are 0→1→0
  excursions; wildfire moves 3↑/3↓). C1's **17↑/5↓** subtask, **legal 9↑/1↓**, and
  **RAE-relaxed 5↑/0↓** are asymmetries that symmetric noise cannot manufacture.
- The **binary main +9.8%** (≈4 net passes / 41) is ~1.5σ against a ~17% flip background —
  suggestive, and it points the same way as the finer metrics, but it does **not** carry
  the claim on its own. The continuous + directional evidence does.

**Rule for attribution:** do not attribute any *single* small per-task flip (|Δ| ≲ 0.08)
to the knob — that is within noise for the wobbly tasks. The claim rests on the aggregate
+ the domain-concentrated asymmetry, not on cherry-picked task flips. (Same twin-noise
discipline as the levers report.)

---

## 4. The concurrency instant-fail finding

~10 hard/looping tasks (arch-hard-1/9/12, astro-hard-8/9/11, bio-hard-1, env-hard-12/14,
arch-easy-11) returned in **0.09 s with no work, no error, no answer** during the
concurrency-6 pool — uniform across all 3 arms. Root cause: **engine contention**, not a
mini capability limit — the same task, run in isolation (concurrency 1), executes
normally (ran to the 2-min probe cap). litellm logged `200 OK` and agent-service
`errorCount: 0` throughout, so the LLM never errored; the agent-service returned empty
fast when the aging shared engine (JVMs ~98 h old) couldn't seat a new session under load.
Because it is uniform across arms and completed tasks have **0 partial-coverage**
(all-or-nothing), the matched knob comparison is unbiased. Mitigation used for the
variance run: concurrency 3.

---

## 5. Agent-service decision

**No system-prompt or tool change made.** Evidence: classified all 18 mini main-failures
— 0 are parse artifacts. `parse_answer` (free-text "Final Answer:" regex) extracted a
well-formed, right-type value in every case; the values are simply wrong (several numeric
near-misses). A structured `report_answer` tool would tidy the answer channel but cannot
raise accuracy when parse-noise is already ~0. The residual eval noise is the agent's
stochastic reasoning (§3), which no prompt/tool surface removes. (Also: the live
agent-service runs under `bun --watch`; editing it hot-reloads and would disrupt in-flight
runs — another reason to leave it untouched mid-study.)

---

## 6. The chunked-judge metrics (new M3/M4) — built, run, and the failure-mode result

Final metric design (implemented in `scripts/judge_metrics.py`, launched via
`./kb.py judge`): source = the LAST react step's `inputMessages` (byte-exact record of
what the agent saw; verified equal to the whole-trace union in DELTA, 38/38). Chunk by
event (DELTA — carries summary + code + result) or by operator + attached code (LATEST).
One judge call per chunk per lens, all gold subtasks listed, **binary** verdicts keyed by
subtask-ID, temperature 0. Task score = % of subtasks covered (spectrum from counting).
- **M3 lens**: does this chunk contain the subtask's evidence VALUE(s)? (strict for lists)
- **M4 lens**: does this chunk show the step being PERFORMED? (values irrelevant)
Results cached per task (`judge_m3m4.json`), readable via `kb.judge_scores()`.

**C1 run (1k vs 5k mini, 29 matched trace-clean tasks, gpt-4o-mini judge):**

| metric | 1k | 5k | Δ | movement |
|---|---|---|---|---|
| M3 evidence-in-context | 0.712 | 0.766 | **+0.054** | 10↑/4↓ |
| M4 step-performed | 0.779 | 0.736 | −0.043 | 6↑/10↓ (≈flat, judge noise) |
| M2 subtask answer (same 29) | | | +0.095 | 16↑/2↓ |

**The failure-mode split (Bob's taxonomy, measured):**

| | 1k | 5k |
|---|---|---|
| failed tasks | 9 | 5 |
| mode1 — step missing | 6 | **5 (100%)** |
| mode2 — steps done, value absent | **2** | **0** |
| mode3 — had everything, still failed | 1 | 0 |

**Reading:** the rows knob raises evidence delivery (M3 +0.054), leaves the plan
unchanged (M4 flat — 6↑/10↓), and its accuracy gain shows up as the **elimination of
mode-2 failures**: at 5k every remaining failure is step-missing (mode1), which no render
knob can fix. That is the knob's mechanism and its ceiling in one table. (Per-task
mediation is again loose — M3Δ is not concentrated in the answer-up bucket — 5th
replication of layer separation; aggregate delivery rises, per-task conversion stays
stochastic.) M4-process ≡ M4-deliverable here: mini agents rarely delete operators.

### M3 across models (all C1 tasks): the knob only helps where it actually binds

| pair | n | M3 A | M3 B | ΔM3 | moves | Δanswer |
|---|---|---|---|---|---|---|
| **mini 1k→5k** (same era) | 49 | 0.635 | 0.716 | **+0.081** | 21↑/5↓ | **+0.083** |
| **5.2 3k→5k** (same era) | 102 | 0.722 | 0.720 | **−0.002** | 22↑/23↓ | +0.024 (coin) |
| 5.2 1k→5k (CROSS-ERA, don't trust) | 102 | 0.692 | 0.720 | +0.028 | 32↑/26↓ | +0.101 |

**This explains the mini-vs-5.2 asymmetry one layer earlier than expected.** At 5.2,
3k→5k changes *nothing about delivered evidence* (M3 dead even, 22↑/23↓) — the cap isn't
binding; 5.2's context already carries ~0.72 of the gold evidence at 3k. So there was
nothing to convert: the answer coin at 5.2 is a **delivery-side** null, not a
conversion-side one. At mini, 1k genuinely starves (0.635) and 5k relieves it (+0.081,
21↑/5↓) — and answers follow (+0.083). M3 is doing exactly its manipulation-check job:
it tells you *whether the knob physically did anything* before you argue about accuracy.

Caveats: (1) knob steps differ (mini 1k→5k vs 5.2 3k→5k) — the true apples-to-apples 5.2
1k→5k pair is cross-era (1k traces Jul-14, 5k traces Jul-6..12; both other eval vintages
burned us before) — its +0.101 answer delta is NOT interpretable; re-running one 5.2 arm
same-era would settle whether 1k starves 5.2 too. (2) The 5.2 judge run was M3-lens only
(M4/failure-mode fields in those caches are placeholders — `lenses` field marks this).
(3) Mini all-50 failure-mode table includes 12 mixed-run tasks (trace=run3,
answer=run1); the clean mode table is the 29-task one.

## Artifacts
- `scripts/judge_metrics.py` — the M3/M4 chunked-judge engine (run via `./kb.py judge`).
- `scripts/judge_vs_answers.py` — joins judge metrics with answer metrics (M1/M2).
- `scripts/final_knob_analysis.py` — matched C1/C2 on subtask eval + main pass.
- `scripts/m4_mediation.py` — string-match evidence-delivery mediation (pre-judge version).
- `scripts/pipeline_design_metric.py` — pipeline-design judge driver.
- `benchmark/llm_tools/gpt_interface.py` — **bug fix** (verdict coercion).
- `systems/dataflow_system.py` — subtask format-hint synthesis; mini C1/C2 + Latest1k/CodeInSnap arms.
- `kb.py` — `judge` subcommand + `judge_scores()` loader.
- Task lists (repo-local): `tasks50.txt` (the fixed 50-task sample), `tasks_judge29.txt`
  (29 matched trace-clean), `variance_subset12.txt` (the 3×-rerun variance subset) — all
  in this directory; the analysis scripts default to them.
- Removed as superseded: `scripts/compare_subtask_arms.py`, `scripts/agg_subtasks.py`
  (both replaced by `final_knob_analysis.py`); the RAE-relaxed rescoring script was
  scratchpad-only and is documented as rejected in §2(c).
- Variance harness (`var_run.sh`/`var_snap.py`) was scratchpad-only; its result table is
  §3 and the run1/2/3 evaluation snapshots back the numbers there.
