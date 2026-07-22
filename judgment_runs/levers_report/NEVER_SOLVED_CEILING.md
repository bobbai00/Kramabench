# The theoretical ceiling and the never-solved core (2026-07-18)

Oracle-router union across all scored arms (best arm picked per task), plus a
per-task diagnosis of every task no arm solves, and the levers that could move
them. Grounded in gold solutions + current DeltaStats3kD2 traces + re-executed
fingerprints (4 diagnosis agents; files cited per task).

## 1. Ceilings

| Union (oracle picks best arm per task) | Solvable / 104 |
|---|---|
| Best single arm (pre-probe DeltaStats3kD2) | 83 |
| Probe-star 4 arms (C1/C2/C3) | 82 |
| All GPT-5.2 dataflow arms (49) | **93** |
| Everything incl. code agent (71 arms) | **95** |
| — never solved by ANY arm | **9** |

## 2. Where knob-tuning actually recovers tasks (82 → 93)

The 11-task gap between the probe-star union and the all-dataflow union is the
knob-recoverable set — the concrete payoff of a per-task/per-step knob policy:

| task | recovered by | knob |
|---|---|---|
| astronomy-hard-9 | 14 arms | column stats |
| environment-hard-9 | 13 arms | column stats |
| legal-hard-15 | 13 arms (pre-probe only) | 5k rows-window / stats dup-line |
| astronomy-easy-4 | 12 arms | budget / stats |
| legal-hard-1 | 6 arms | stats |
| environment-hard-17 | 3 arms | **explore mode** |
| environment-easy-3 | 3 arms | latest / converge |
| wildfire-hard-14 | 2 arms | latest + stats |
| astronomy-hard-7 | 1 arm | (coin) |
| astronomy-hard-12 | 1 arm | (coin) |
| wildfire-hard-21 | 1 arm | (coin) |

Reading: **stats + larger budget is the dominant recovery channel** (~7 of the
11 tasks), plus one explore-mode-only task. A per-step controller that turns
stats on when its signal fires (DYNAMIC_KNOBS S1/S2/S10) captures most of this
without paying stats cost everywhere. The last three are single-arm lottery
(chronic coins), not reliable knob wins.

Within the probe-star four arms, only 2 tasks are single-arm-solved
(archeology-hard-7 → 1k, environment-hard-11 → latest) — both chronic coins, so
the star's *internal* knob spread buys almost nothing; the recovery comes from
adding the stats/budget arms.

## 3. The 9 never-solved — taxonomy

| task | gold | arms converge to | category |
|---|---|---|---|
| archeology-hard-2 | 38.42 | 47.29 | **parse: wrong column** (lever-fixable) |
| archeology-hard-1 | 8577.53 | 380.82 / 8477.86 | **parse: wrong column** + domain (K=Potassium) |
| environment-hard-16 | 60 | ~75 | interpretation (entity key) — half lever |
| archeology-easy-8* | 52 | 82 | **code-agent edge** (inspect loop) |
| astronomy-hard-11 | 4.638e-13 | no-response | **benchmark bug** (filename) + exec-limit |
| wildfire-hard-19 | 32.76 | no-response / 0.00 | **benchmark bug** (missing file) |
| archeology-hard-12 | 409 | 447 | interpretation + **benchmark instability** (seed) |
| archeology-hard-9 | 0.015648 | ±0.1 scatter | interpretation (gold contradicts prompt) |
| archeology-hard-5 | 66158 | 36829 / 0.026 | parse (profile) + interpretation (kyr-bucket) |
| astronomy-easy-3 | 7.95e-13 | 8.02e-13 (all 88 arms) | interpretation → 0.9% near-miss |
| archeology-easy-11* | 17.4274 | 17.1667 | interpretation (primary-capital) |

(* = solved only by the code agent, not in the 9; included as the 93→95 gap.)

Split: **3 benchmark defects**, **~2 clean lever-fixable**, **~2 code-agent-edge**,
**~4 task-intrinsic gold conventions**.

## 4. Benchmark defects — not agent-fixable, file upstream

- **wildfire-hard-19**: the required input `WeatherEvents_Jan2016-Dec2022.csv`
  (Kaggle) is *absent from the repo* — `data/wildfire/input/load_data.py` admits
  it "needs to be downloaded ... not included due to size." 32.76 is unreachable
  by construction. The best achievable proxy (`noaa_wildfires.rain_sum>0.05` →
  27.35) still fails the 0.9 gate. → stage the file or drop the task.
- **astronomy-hard-11**: the prompt manifest AND `ground_truth.json` list
  `omni2.txt`; the file on disk is `omni2.text`. The declared path 404s. →
  fix the manifest.
- **archeology-hard-12**: gold's own script is PYTHONHASHSEED-nondeterministic
  (unseeded → 410; seed 0/42 → 409) — the published answer isn't stable. → pin
  a seed or widen tolerance. (Joins the wildfire-hard-18 gold-script-bug family.)
- **astronomy-easy-3** (borderline): all 88 arms answer 8.02e-13, gold 7.95e-13,
  a 0.9% gap under `numeric_exact`; a ≥1% tolerance passes every arm. Arguably a
  scoring-tolerance defect on an under-specified question.

## 5. Lever-addressable — the new context/render levers, ranked

### Lever 1 — Load-failure recovery probe (fail-fast + list + nearest name)
On `FileNotFoundError` (or N identical failed loader edits), auto-render the
`dirname` listing + nearest-filename suggestion, and **block re-submitting the
same path**.
- **Worked example — astronomy-hard-11**: the arm hit `FileNotFoundError:
  ...omni2.txt` at step 1 and resubmitted ~10 near-identical loaders (changing
  reader/encoding, even a no-op `.replace('omni2.txt','omni2.txt')`) for **24 of
  26 steps**, never once listing the directory, then step-capped → no-response
  (405,921 input tokens). A load-failure render ("not found; dir has:
  **omni2.text**, omni2_2024.dat, ...") breaks the loop at step 2; the real
  pipeline fits in ~22 steps.
- **Also salvages wildfire-hard-19's 0-score arm**: it listed the dir only at
  step 24/26; fail-fast reaches the 27.35 proxy instead of no-response.
- Gain: converts 2 no-responses into answers (≥1 near-miss); frees budget on any
  discovery-heavy task. Cheap, append-only, cache-safe.

### Lever 2 — Header-aware named-schema render at load
When a load yields `Unnamed:` columns or a data value appears as a header,
re-render the raw first ~8 rows AND the schema *after* the candidate `skiprows`,
so real column names are visible (the probe prompt already tells the agent to
re-load; the render must then SHOW the named re-load).
- **Worked example — archeology-hard-2**: `climateMeasurements.xlsx` has 5
  metadata rows (real header on row 5) and stacked sub-tables. The arm reloaded
  `header=None` and grabbed positional col 26 = `ODP 967 Dust proxy` (a decoy)
  instead of col 29 = `ODP 967 wet-dry index`, computing the increasing-fraction
  of the wrong series → 47.29 vs gold 38.42 (re-executed: col26-diff = 47.29
  exactly). A `skiprows=5` named schema makes `Dust proxy` vs `wet-dry index`
  visible — the disambiguation is then trivial. This is the single most
  cleanly-fixable never-solved task.
- Partially helps archeology-hard-1 (same sheet; arm used `Unnamed:25/26` = age
  + dust as "K" and time axis) and archeology-hard-5's profile arm (surfaces a
  real `Al` column, killing the 0.026 degenerate).

### Lever 3 — Distinct-value inspection + re-inspect-after-transform
Render the sorted unique values (or a random distinct sample) of a *derived/
cleaned* column — not just row samples or top-5 — and re-inspect after each
cleaning pass. This transplants the code agent's inspect-and-refine loop, which
is its real edge (not planning).
- **Worked example — archeology-easy-8** (code-only): dataflow stops at the
  scalar `unique_sources: 82` — token dirt is invisible, the pipeline "succeeds,"
  nothing triggers iteration. The passing code agent prints `sorted(unique)[:40]`
  each round, *sees* `'BNP.'` vs `'BNP'` and a stray `'1995'`, and iterates
  82 → 53 → 52 = gold. Rendering the distinct token values gives dataflow the
  same trigger.
- **Also helps environment-hard-16**: the group-by key `Beach Name` is
  high-cardinality (distinct=672), so stats suppress its values and no sample row
  ever contained an `@`; the arm keyed by `Community|Beach Name` → 75 vs gold 60.
  A distinct-value sample of the key surfaces the `@ Left/@ Right/@ Middle`
  monitoring-segment pattern → invites canonicalization (`split('@')[0]`).
  (Residual: must also ignore `Community` — 62 vs 60 — a gold convention.)

### Lever 4 — Multi-interpretation fan-out (weak; raises union, not single-shot)
For interpretation-ambiguous tasks, run K readings and reconcile. Without an
oracle it can't *pick* the gold reading, so it lifts only the union ceiling.
- **Worked example — astronomy-easy-3**: the whole 0.9% gap is one reading —
  "midnights of the initial-state days" (arm, n=119) vs "all midnights per
  density file, keyed by filename" (gold, n=469). A fan-out that includes the
  "entity key is in the filename → prefer per-file join over calendar join"
  reading would generate the gold candidate — but selection still needs the key.

## 6. Task-intrinsic — no context lever reaches them

Gold semantics that live nowhere in the data or renders (sometimes contradicting
the question):
- **archeology-hard-9**: gold keeps roman hit by `idxmax(rank)` while the
  question says "take the last sample"; target is a near-zero correlation scored
  at 6 dp. Re-executed ablation: the tie-break alone flips the sign
  (L2+idxmax=+0.0156 vs L2+last=−0.210). Un-transmittable.
- **archeology-hard-5 (kyr bucket)**: even a clean parse gives 36,829; gold's
  `round(Age_ky.1)` integer-kyr bucketing creates the 13-row "multiple closest
  years" tie whose max Al = 66,158. The rounding granularity is unstated.
- **archeology-hard-12 (semantics)**: gold reads "attribute" as a filter
  requiring BOTH actors to substring-match a worldcities country, plus an
  overlap-interval merge, and applies NO duration filter — arms' natural reading
  (duration ≥ 1 yr, attribution = no-op) gives 447.
- **archeology-easy-11 (primary)**: "capital city" = `capital=='primary'` only;
  arms use `capital.notna()` → 17.1667. The enum was fully rendered; it's a
  semantics call. (Secondary NaN-pop quirk is lever-addressable but only matters
  after the read is won.)
- **environment-hard-16 (community)** and **astronomy-easy-3 (which midnights)**:
  the residual convention after the lever does its half.

## 7. Bottom line

- **The realistic dataflow ceiling is ~93/104, not 100.** Of the 9 never-solved,
  ~3 are benchmark defects (file upstream), ~4 are gold-private/unstable
  conventions no render can carry, and only ~2–3 are genuinely moved by new
  context levers — and for those, the lever mostly converts a no-response or a
  degenerate answer into a *closer* one, not always a pass.
- **The high-value, reliable win is per-task/per-step knob selection over the
  82→93 gap** (stats + budget + explore), not chasing the never-solved tail.
- **The code agent's 2-task edge is a feedback mechanism** (value-level prints,
  crash messages as evidence), reproducible in dataflow via Lever 3 — the one
  architectural idea worth importing.
- Levers 1–3 are append-only, cache-safe, and fire only on their trigger
  (load failure / Unnamed columns / high-cardinality key), so they add
  near-zero cost on the 93 tasks that don't need them.
