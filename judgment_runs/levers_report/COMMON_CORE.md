# The common-failure core — what the arms fail at TOGETHER

Directive-1 (do the arms fail the same tasks?) + directive-4 (deep-dive every
failure of the best arm, DeltaStats3kD2). Data:
`common_failures.json`; per-task dives in `deep_dives/`.

## Do the arms fail the same tasks? Largely yes.

Per-arm fail counts (of 104, pass th 0.9): Delta3k 24, Delta5k 23,
**DeltaStats3kD2 21** (best), Latest3k 25.

- **16-task ALL-ARM common core** = 52% of the any-arm-fails union (31).
  Only **2/16 are chronic flippers** → this core is *stable hard*, not noise.
- Pairwise fail-set Jaccard **0.64–0.78** — the arms overlap heavily; they
  are the same system failing the same tasks, not four different profiles.
- **Unique failures (this arm fails, all others pass): 0–2 per arm, ALL
  chronic** — Delta3k {env-hard-7*, env-hard-8*}, Stats3kD2 {biomed-hard-7*,
  wildfire-hard-12*}, Latest3k {biomed-hard-5*}, Delta5k {none}. Every
  "exclusive" failure sits inside the twin-noise band (the semantic walks
  already showed these are coin-flips). **No arm has a systematic private
  weakness.**

Conclusion: accuracy is a property of the *task set*, not the render config
— which is exactly why C1/C2/C3 sit at the same aggregate score. The
common core is the real target for future accuracy work.

## The 16 common-core tasks, classified (best-arm dives)

| task | failure class | one-line mechanism |
|---|---|---|
| archeology-easy-11 | convention | `capital` rank enum: all read notna() not `=="primary"` (all 4 → 17.17 vs 17.4274) |
| archeology-easy-8 | underspecified parse | bibliography `;`-split + normalization spec not given (872/82/1/55 vs 52) |
| archeology-hard-1 | header trap + method | interpolate-at-2-points read as window-mean (profile got furthest) |
| archeology-hard-2 | header trap + ordering | missing chronological re-sort before differencing (cluster 47–50 vs 38.42) |
| archeology-hard-5 | header trap + selection | closest-year/max-Al mis-selected (3 arms → 36828.72) |
| archeology-hard-9 | distance-metric | near-zero correlation amplifies L2-vs-box match (scatter across 0) |
| archeology-hard-12 | convention | "attribute to countries" read as annotation not filter (all 4 → 447 vs 409) |
| astronomy-easy-3 | near-miss / metric | ~1% off (8.02e-13 vs 7.95e-13), killed by numeric_exact |
| astronomy-hard-7 | task complexity | VAR(1)+regression pipeline, 4 different failure stages |
| astronomy-hard-11 | execution-limit | NRLMSISE-00 physics model, all 4 step-cap, no answer |
| astronomy-hard-12 | execution-limit | SP3 geodesy pipeline, all 4 step-cap, no answer |
| environment-hard-16 | convention | "no violation throughout" scope; cluster 73–75 vs 60 |
| environment-hard-17 | convention | "summer with most rainfall" is a year-selector; all 4 → 21.43 vs 47.37 |
| wildfire-hard-14 | near-miss / metric | AQI "generally unsafe" threshold; all 4 → 0.42 (score 0.74) vs 0.65 |
| wildfire-hard-19 | task complexity | 1km geospatial join; best arm step-capped, lean arms 0.86 |
| wildfire-hard-21 | convention + robustness | "keep missing rows" ignored (lose WA); one arm gave up on a readable file |

## The four families (and which are context-addressable)

1. **Convention / interpretation misreads (7): archeology-easy-11,
   -hard-12, environment-hard-16, -hard-17, wildfire-hard-21, (+ the
   flip-side of easy-8, hard-2).** The task's English underspecifies a
   filter/selector/definition; the arms converge on a reasonable-but-wrong
   reading (often the *identical* wrong number across all four). **Not
   context-addressable** — fixable only by disambiguating the task or the
   agent asking. This is the single largest family.
2. **Dirty-header / parse traps feeding a downstream method error (4):
   archeology-hard-1, -hard-2, -hard-5, easy-8.** The header/parse
   sub-problem IS render-assisted (the D2 profile reliably gets the stats
   arm past the skiprows-5 header — visible as its shorter step counts), but
   the *downstream* reasoning (interpolation, chronological sort, closest-year
   selection, normalization) has no render surface. Render moves the failure
   later; it doesn't close it.
3. **Task-intrinsic complexity / execution-limit (4): astronomy-hard-7,
   -hard-11, -hard-12, wildfire-hard-19.** Multi-stage physics/geo/ML
   pipelines that don't fit the 25-step budget or exceed the agent's
   modeling reliability. Two see all-arm step-cap deaths. **Not
   context-addressable**; needs a larger step budget and/or domain tools.
   wildfire-hard-19 is the notable anti-example: the richest-context arm
   (stats) was the *only* one to time out.
4. **Near-miss killed by the metric (2): astronomy-easy-3 (~1%),
   wildfire-hard-14 (score 0.74).** Essentially-correct answers that fail a
   strict exact/approximate threshold. A scoring artifact, not a capability
   gap.

## Implications

- **Accuracy on KramaBench is dominated by task semantics + execution
  budget, not context rendering.** 13/16 core failures are convention,
  complexity, or metric artifacts — outside the sampling/stats/history
  parameter space entirely. This is the strongest form yet of the levers
  report's thesis (levers are cost/behavior knobs, one narrow accuracy
  channel).
- **The render lever's real accuracy contribution is the dirty-header
  anti-iteration effect** (family 2 + the CASE_METRICS F2/F5 finding): the
  profile shortens the path through messy loads. It just can't fix what
  happens after the load.
- **Two concrete, non-render levers the core points to:** (a) a
  clarify/convention mechanism for underspecified filters/selectors (family
  1, ~7 tasks); (b) a larger step budget + domain-tool access for the
  physics/geo pipelines (family 3, ~4 tasks). These would move the common
  core; more rows/stats/history would not.
- **Two "failures" are metric artifacts** (family 4) — worth excluding or
  re-scoring when reporting a ceiling.
