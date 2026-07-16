# Dynamic knobs — the signals in the data that say which knob to turn

A re-read of every deep-dived case (10 counter-intuitive flips + 16
common-core failures + the attributed flip + the 1k experiment) through one
question: **was there a signal, observable in the data or the run itself, that
would have told a controller which rendering knob to turn for that operator?**
This is the design ground for tuning the knobs *dynamically* — per operator,
conditioned on what the agent is looking at — instead of the global static
settings we benchmarked.

## 1. The signal inventory (per-case evidence)

Each row: an observable signal → the knob action it justifies → the case(s)
that prove it → when the signal is computable.

| # | signal (observable) | knob action | evidence | computable |
|---|---|---|---|---|
| S1 | **source op loads multiple files** (glob/concat in code, or engine sees k>1 files) | force **stats ON** for that op (the profile's `duplicate rows: N of M` line is the dedup evidence) | legal-hard-15 — the ONE attributed flip: winner wrote `drop_duplicates` in the same step it saw `duplicate rows: 359 of 764 (47%)`; loser's fused 1×1 render hid it | at op-write time (code inspection) — cheapest, most causal signal we have |
| S2 | **schema has `Unnamed:` columns or str-share ≈ 100%** | stats ON for that op, and/or raise its row budget — rows and profile are *substitutes* for structure facts | C2 stats-wins family (env-hard-8/-11, beach datasheets 7–11 unnamed cols); 1k experiment: stats +5 at starved budget, +3 at 3k | after first execution, from the schema line itself |
| S3 | **out_rows ≤ ~100** | render ALL rows for that op (cross the full-visibility threshold; probes vanish) | F1/F3: 5k-cheaper cohort med 95-row files, rendered 86% vs 43%, ops/task 4.5 vs 6.1 (biomedical-easy-9: 11 steps→4) | at render time (engine knows row count) |
| S4 | **out_rows ≥ ~1k** | rows are decoration — schema/stats only for that op | F1: ≥101 rows neither 3k nor 5k ever reaches 90% coverage; both-elide = waste | at render time |
| S5 | **multiple ID-like columns** (`*ID*` names) on a table about to be joined | render a **key micro-profile**: per-candidate null-share + cross-table overlap count | wildfire-hard-17 — all 3 arms picked `WX ID` because the true key `NWS ID` showed NaN in its only rendered row (822/2965 null, row-0 trap); a null/overlap line identifies it instantly | at join-write time; overlap needs one cheap engine pass |
| S6 | **non-delimited / spec-referenced / mid-file-shifting formats** (.dat, fixed-width, task names a spec file) | attach **raw first+middle lines** of the file to the source op's observation; sanity-check parsed column ranges | astronomy-hard-9 — blind `read_fwf` silently truncated DOY; parse *looked* valid at the head; plain code agent wins by raw-probing. Also env-hard-9 (silent col mis-pick) | at load time, trivially |
| S7 | **parse-anomaly**: parsed values implausible for the column's role (DOY max=9 over 8784 rows; a "density" column at 1e10) | render an anomaly line; treat as loud failure → force re-parse | astronomy-hard-9 (DOY), astronomy-hard-7 (22-orders scale blow-up), env-hard-9 (winner was SAVED by loud 0-row failures — loudness is the mechanism) | post-execution, cheap range checks |
| S8 | **an operator reaches its 2nd+ landed version** (multi-edit begins) | history/versions knob: render that op's revision history at full fidelity; leave single-shot ops lean | F6 — multi-edit share sorts all 12 comparison cells (31–36% where richer arm wins, 3–8% where lean wins); "iteration begets information value" | at write time (edit counter) — append-only-safe by construction |
| S9 | **churn flag fires** (sink-share ≥50% ∧ ops ≥8, or a resubmission with difflib >0.92) | behavioral intervention: inject a delete-nudge / consolidation prompt; do NOT just add render (churn is thrash-despite-evidence) | wildfire-hard-17 + astronomy-hard-9 Latest walks: every re-probe had its refutation rendered; 7/103 flagged tasks run 29% vs 80% pass, 4× steps | at run time from the live DAG |
| S10 | **render budget starved** (small cap / high context pressure) | shift evidence channel: profile over rows (fixed-size structure facts) | 1k experiment: stats gain grows as cap shrinks (+3 @3k → +5 @1k); S2×S10 interaction is where accuracy actually moved | configuration + live context size |
| S11 | **operator depth ≥ 2** | minimal render regardless of other signals | F4: cardinality collapses 153/39/1/1 by depth; all render pressure is at the data edge | at render time (DAG position) |

Two anti-signals worth recording (places a naive controller would overfit):
- **More of the same window is NOT a signal.** wildfire-hard-17's 5k arm held a
  superset of the winner's rows and still picked the wrong key — the fix is a
  *different* fact (null share), not more rows of the same head.
- **Stats can mislead.** biomedical-hard-7: `distinct=15` reinforced the
  header-eaten wrong count; wildfire-hard-12: the stats arm invented a
  spurious `|corr|>0.2` rule. The controller should deliver *targeted* facts
  (S1's dup line, S5's key profile), not maximal facts.

## 2. What this means for a tuning method / learned controller

**The object to learn is a per-operator, per-version rendering policy**
π(op-signals) → {row-budget ∈ {all, sample, schema-only}, stats ∈ {off,
profile, profile+key-micro}, history ∈ {lean, full-revisions}} — decided **at
write/execute time and never revised** (append-only ⇒ prompt-cache-safe; the
mutation-loses-to-cache law is replicated 4×, so any policy that retroactively
re-renders old events is dead on arrival). This is exactly the render-prefs
mechanics, but policy-driven rather than agent-declared — render-prefs v1
failed because the *agent* over-economized; the signals above are computed by
the engine, not chosen by the model.

- **Rule seed (no ML needed to start):** S1, S3, S4, S6, S11 are direct
  engine-side rules with measured effect sizes. S2+S10 add the
  budget-conditional stats rule. This rule stack IS the "static router"
  version and is worth benchmarking on its own.
- **Where ML genuinely helps:** the interactions (S2×S10 substitution rate;
  S3's threshold as a function of width; when S8 history fidelity pays vs the
  DELTA replay tax) and the churn predictor (S9 fired early rather than at
  sink-share 50%). Training substrate already exists: `case-metrics` emits
  per-op features (depth, cardinality, dirtiness, edits, render pressure),
  and the venn/recovery machinery emits outcome labels per (task, config) —
  plus the earlier learned-context-selector work (utility model + knapsack,
  offline AUC 0.893) is precisely this shape and can be retargeted from
  "which events to keep" to "which knobs per op".
- **Honest headroom statement (the tradeoff):** the walks showed only ONE
  render-attributed accuracy flip in C1/C2/C3, and 13/16 common-core failures
  live outside the render space (conventions, execution limits). So the
  learned policy's expected wins are: (a) **cost** — recover the 37-task
  stats tax and the small-table render waste while keeping the 9-task dirty
  wins (the C2 asymmetry, now budget-dependent per the 1k result);
  (b) **accuracy at starved budgets** (+5 at 1k is the proof the knobs move
  accuracy when the operating point is lean — relevant for long-horizon /
  many-source tasks like LakeQA where budgets are inherently starved);
  (c) **accuracy via NEW micro-signals** (S5 key-profile, S6 raw-preview,
  S7 anomaly lines) — these aren't in the current knob space at all; they're
  cheap render *additions* whose triggers the same policy owns. The thing a
  tuner canNOT do is fix convention misreads or step-budget exhaustion.
- **Cost model constraint:** every signal-triggered render addition pays
  bytes on that op forever (append-only). The policy objective is the same
  cache-aware `cost_usd` + pass-rate pair we benchmark with; the twin-noise
  floor (±3 passes, ±10% cost) defines the minimum detectable improvement.

## 3. The experiment this file justifies (run first, cheap)

The S6/S7/S5 bundle is testable TODAY without engine changes, as a prompt
protocol (see `TODO_RAW_PROBE_PROMPT.md`, now being piloted): probe raw
text before writing loaders (all formats — the dirty cases in our core are
mostly .csv/.xlsx), read named spec files, sanity-check parses, key-profile
before joins, delete probes after verification. Targets: the format-blinded
failures (astronomy-hard-9, environment-hard-9, wildfire-hard-17 on their
failing arms; archeology dirty-header trio on the best arm). Success is
judged at the *mechanism* level in traces (did it probe → did the probe
change the loader/key), because the target tasks are chronic flippers where
single-run pass/fail is noise.
