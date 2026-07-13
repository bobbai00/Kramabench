# The A/B Comparison Pipeline

The standard for comparing two context configurations on KramaBench. Distilled
from the July-2026 experiment program (frontier decay, rank-3/4, E1, explore
mode, render prefs, and the levers report); every rule below exists because
skipping it produced a wrong conclusion at least once.

## 0. Design the pair

- **One knob per pair.** Define both arms as SUT classes differing in exactly
  one field; assert it with a config-parity unittest
  (`systems/test_static_rule_configs.py` pattern).
- **Same code vintage, both sides.** Never compare against a run from another
  day/checkout: an identical config re-run days apart moved ±3 passes and
  ±10% cost, and the renderer accumulates permanent (unflagged) changes.
  If the baseline run is old, re-run a fresh control (the render-prefs
  "interim win" reversed to a loss against its true control).
- For multiple levers, use a **star**: one anchor arm, one ray per lever
  (levers_report/REPORT.md §2). Add mirrored pairs only as a secondary
  interaction section — levers are mode-dependent.

## 1. Run + equalize (recovery-first)

```bash
./run_experiment.sh <ARM_A> <ARM_B>          # full run + 2× --all-failed recovery + scores
# or, to equalize already-run arms:
./kb.py rerun-failed --sut <ARM> --all-failed --parallel --isolate --watchdog-min 8   # ×2, each arm
```

Recovery converts transient failures (watchdog kills, one-off spirals) into
stable outcomes. Both arms must receive the SAME number of rounds
(symmetric), or the flip lists are polluted by recovery asymmetry.

## 2. Calibrate the noise floor (twin pairs)

Chronic flippers = tasks that flip between IDENTICAL configurations. Current
set (23 tasks) in `levers_report/chronic_flippers.json`, computed from three
same-config rerun pairs. Regenerate when new twin pairs exist. Magnitudes to
respect: **9–12 flips per identical pair, ±3 net passes, ±10% cost.** Any
aggregate delta inside that band is unmeasurable; any single flip on a
chronic task defaults to variance.

## 3. Venn + category extraction

```bash
./kb.py venn --sut <ARM_A> <ARM_B>     # --th 0.9, --top 8, --chronic <json>
```

Outputs: the outcome Venn (A-only / both-pass / B-only / both-fail),
the both-pass **cost split** (who is cheaper on how many tasks, gross
savings), chronic tagging on every flip, and **per-category operator/file
profiles** (roles, multi-edit share, LOC, file ext/size) for A-only wins,
B-only wins, and each side's top cost-gap both-pass tasks.

Reading the profiles: multi-edit share is the strongest single sorter —
categories where the richer arm wins run 24–36% multi-edit ops (parse
iteration on big/weird files: .xlsx/.gpkg/.cdf/.tle/.text); categories where
the leaner arm wins run 3–9% (clean single-shot pipelines).

## 4. Accuracy: flip attribution (both directions)

For every non-chronic flip — and any chronic one you want to claim — the
trace must show the lever's INFORMATION doing the work:

- locate the decision the task hinges on (usually one: a dedup, a key, a
  sheet, a parse spec);
- show what evidence each arm's context carried about it at decision time
  (rendered observations, not assumptions — sweep-era traces have EMPTY
  thoughts, so attribute via code + observations);
- accept if the winning arm's evidence explains its action AND the losing
  arm's absence explains its error. Strongest forms: the losing arm is
  unstable across its own reruns on that decision (evidence-starved
  coin-flip), or two different levers independently fix the same failure
  (legal-hard-15). Reject method-choice divergence that predates the
  lever's first rendered difference (environment-easy-3, astronomy-easy-4).
- Report BOTH directions. The mined "richer-wins" list is half the picture.

Useful mechanism scans (all in this repo's history):
identical-probe repetition (the LATEST thrash blind spot, astronomy-hard-9),
stale-history answers (environment-hard-13), resubmission-similarity
(difflib > 0.92), failure-mode classification (timeout vs wrong-answer).

## 5. Cost: same-answer cohort + per-step invariants

Totals are day-noise-prone. The robust cuts, in order of strictness:

1. both-pass + **same normalized answer** cohort (drop different-workflow
   noise);
2. per-step invariants: input/step, uncached-input/step (stable when totals
   swing);
3. **same-step cases** (|Δsteps| ≤ 1): the pure render-byte gap — the
   cleanest lever cost evidence (e.g. +31% at identical 6-step trajectories);
4. cache-aware `cost_usd` ONLY. Raw token deltas mislead: every mutation
   experiment cut tokens and raised cost (cache churn).

## 6. Case-scoped footprints

```bash
python scripts/analyze_lever_footprints.py   # whole-arm lever footprints
./kb.py venn ...                             # per-category profiles
```

Characterize WHERE the lever binds: role/depth/LOC/file-size/dirtiness
(dirtiness parsed from a stats-bearing arm's Output Table profiles =
full-data facts). Established regularities: levers bind at the data edge
(sources, depth 0; sinks never); sampling/profiling are volume levers
(big files); history is a difficulty lever (small weird formats).

## 7. Report

Per comparison: aggregates (with the noise-floor caveat) → flip attribution
with representative cases (one per category: a flip case + a cost case per
direction, each with task, decision, evidence, mechanism) → cost cohorts →
category profiles. Template: `levers_report/REPORT.md`.

## Known traps (each cost us a wrong conclusion once)

- **Vintage trap**: cross-day baselines flip verdicts (render-prefs).
- **Aggregate blindness**: every real lever effect here was invisible in
  aggregates (±1–3 passes) and only visible in attributed cases.
- **One-directional mining**: richer-wins-only lists miss half the flips.
- **Raw-token accounting**: ignores the prompt cache; four experiments'
  "savings" were cost increases.
- **Un-recovered arms**: transient failures masquerade as lever effects
  (recovery moved every arm 3–6 passes).
- **Empty-thought traces**: sweep arms recorded tool calls only — plan
  attribution around code + rendered observations.
