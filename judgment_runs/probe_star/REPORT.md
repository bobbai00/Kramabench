# Probe-star report — C1′/C2′/C3′ at the new operating points, under the probe prompt

Bob's redesigned star, run 2026-07-17 on a fresh stack (the first attempt was
scrapped after diagnosed engine senescence — 6.4-day-old JVMs, actor deaths;
this run is post-restart, e2e-gated). All four arms share the raw-probe
prompt vintage (dataflow-agent `acf87127f`→`57bd2fd0a`); every pair is
one-knob (parity-verified) and recovery-equalized (full + 2× `--all-failed`).
Chronic gate uses the OLD-vintage `chronic_flippers.json` — advisory, flagged
per case. Walks: `semantic_walks/`; dives: `deep_dives/`; data:
`venn_C*.txt`, `case_metrics/`, `common_failures.json`.

## 1. Arms and aggregates

| arm | pass | cost | steps |
|---|---|---|---|
| Delta**1k**SchemaOnly-Probe (anchor C1′/C2′) | 73/104 (70.2%) | $4.97 | 744 |
| Delta**5k**SchemaOnly-Probe (C1′ ray, C3′ anchor) | **77**/104 (74.0%) | $5.61 | 747 |
| Delta**Stats1kD2**-Probe (C2′ ray) | 76/104 (73.1%) | $6.12 | 790 |
| **Latest5k**SchemaOnly-Probe (C3′ ray) | 75/104 (72.1%) | $5.74 | 827 |

| pair | knob | A-only | both | B-only | neither | material cost split (≥10%) |
|---|---|---|---|---|---|---|
| **C1′** 1k vs 5k | char cap | 2 | 71 | 6 | 25 | A cheaper 49 ($0.56) / B 22 ($0.39) |
| **C2′** schema vs stats @1k | profile | 3 | 70 | 6 | 25 | A cheaper 37 ($0.73) / B 33 ($0.44) |
| **C3′** delta vs latest @5k | history | 6 | 71 | 4 | 23 | A cheaper 27 ($0.48) / B 44 ($0.49) |

## 2. The headline: knob gaps COMPRESS under the probe prompt

Same knobs, pre-probe vs probe vintage (aggregate pass gap):

- **Sampling (1k→5k): +11 pre-probe → +4 now.**
- **Profiling @1k: +5 pre-probe → +3 now.**
- History @5k: −2 (Delta ahead by 2; pre-probe 3k pair was −1 the other way
  — both inside noise, as always).

Reading: the probe habit is a **third substitutable evidence channel**. Raw
previews deliver structure facts that extra rows or the profile used to be
the only carriers of, so both knobs' marginal value shrinks. (The probe
prompt itself is not harmful at any operating point on clean infra: the 1k
anchor is 73 vs its pre-probe 70, cheaper and with zero no-responses — the
earlier "probe hurts at 1k" scare was the dying engine, withdrawn.)

## 3. Flip attribution — one attributed flip per comparison, each via its knob's canonical channel

27 exclusive-win pairs walked (gold-plan procedure, both directions).
**3 ATTRIBUTED, 24 CHRONIC/REJECTED**:

- **C1′ ATTRIBUTED — legal-hard-2 (5k > 1k, non-chronic): render starvation
  at the window edge.** The dedup evidence (first cross-state row; the
  diagnostic leaderboard of multiplied multi-state names) sat literally one
  row below the 1k render cut, on-screen at 5k. Swapping the loser's
  `groupby.sum` for dedup reproduces the winner's exact 869.494. (Caveat:
  determinism is single-roll; 1k never produced the right answer in ≥5
  observed attempts.)
- **C2′ ATTRIBUTED — biomedical-hard-5 (stats > schema @1k, chronic-tagged
  but rule-clearing): the answer-relevant profile line.** Winner applied the
  exclusion filter immediately after rendering
  `Case_excluded top_5={No=144, Yes=9}`; the schema-only loser at the same
  budget saw only a column name and kept excluded case S043 (2.4241 vs gold
  2.6563). Second attributed stats flip ever — same family as legal-hard-15.
- **C3′ ATTRIBUTED — environment-hard-12 (delta > latest @5k, non-chronic):
  history retention.** The probe op's canonical-name render (`Wollaston
  Beach`) was EVICTED by Latest's compaction before the label-authoring step
  (fell back to filename stub `Wollaston`, exact-match fail); Delta's
  history held it 6×. Caveat: a naming-retention effect, not dataflow
  reasoning — but rendered-evidence-presence cleanly separates the arms.

Everything else replicated the known taxonomy, vintage-invariantly: the
×100 unit coin (environment-hard-7, all four arms bit-identical underneath),
keep-first station (env-hard-11), L2-vs-L∞ (archeology-hard-7),
denominator-interpretation (legal-hard-22: one coin surfacing in three
cells), which-total-is-the-denominator (legal-easy-19: 10 same-config
attempts split 3/3/2 across the three readings), filter-before-shift code
order (legal-easy-9, base year rendered twice to the loser), plan-hop
omission at evidence parity (legal-hard-18), wrong grain (wildfire-easy-9),
Latest self-hallucinated filter (environment-hard-20). biomedical-easy-2 vs
biomedical-hard-5-C3′ is the cleanest coin proof on record: the same
delta/latest knob flips the same `Case_excluded` filter in OPPOSITE
directions on two tasks.

## 4. Why the old attributed channel closed — legal-hard-15 post-mortem (now an all-arm failure)

All four arms answer the exact un-deduped 593524. Per-arm causes:

1. **Delta5k: ATTRIBUTED probe-prompt REGRESSION.** The probe beat bred
   confidence → a single FUSED per-file load→filter→accumulate op emitting a
   1×1 — the concatenated intermediate that used to display duplicate rows
   never existed, so the 5k window had nothing to show. Its pre-probe twin
   materialized the table, saw the 94×2 repeats, deduped → gold.
   **Mechanism: probe-confidence induces pipeline fusion; fusion destroys
   the evidence surface the render lever needs.**
2. **DeltaStats1k: provenance-column muting (engine bug class).** Stamping
   `state_file` onto rows makes them non-byte-identical, so the table-level
   `duplicate rows: 47%` line goes mute (the old passing stats arm only
   fired it because its source column was accidentally all-NaN); the
   remaining per-column dup stats rendered only alongside the
   already-computed scalar.
3. Delta1k: plain render starvation (nearest repeat pair fully elided).
4. Latest5k: six repeated (msa,count) pairs visibly rendered; summed anyway.

## 5. New mechanisms this vintage (for DYNAMIC_KNOBS / engine work)

- **Probe protocol = recovery ramp, not prophylaxis** (wildfire-hard-17):
  0/3 joining arms picked the right key FIRST; both winners joined on the
  trap key, got matched=0 from their verify ops, THEN ran the key audit →
  exact gold. The verify clause converts silent key errors into recoverable
  ones.
- **Latest evicts probe evidence** (2 sightings; decisive in
  environment-hard-12, logged in legal-hard-18): the delete-probes hygiene
  plus Latest's no-history rendering = permanent evidence loss. In Delta,
  deleted probes persist in history. Design implication: in latest mode,
  retain a one-line fact from deleted probe ops, or defer deletion.
- **Fusion vs evidence surface** (legal-hard-15): argues for a
  materialize-before-aggregate nudge and/or engine-side duplicate detection
  computed BEFORE aggregation.
- **Provenance columns mute dup signals** (legal-hard-15): duplicate-row
  detection should ignore `*_file`/provenance columns.
- **Stats' harms are budget-dependent too** (wildfire-hard-12): the old
  |corr|-trap can't recur at 1k because correlation stats never render at
  that budget — symmetric to the 1k experiment's finding that stats' help
  grows when starved.
- **Probe-target stochasticity** (environment-hard-8): WHICH file gets
  raw-probed first is a new coin (probing a single-station beach vs a
  multi-station one changed the plan).

## 6. Common failures: core grew 16 → 22

56% of the any-arm-fail union (39) is everybody-fails. The old 16-core is
fully contained; the six joiners: astronomy-easy-4*, astronomy-hard-9*,
environment-easy-3, environment-hard-9*, legal-hard-1*, **legal-hard-15**
(§4). Mostly the old chronic exclusive-win tasks whose coins landed wrong in
all four arms at these operating points — plus one genuine channel closure
(legal-hard-15). Fail-set Jaccard remains high (0.72–0.79): accuracy is
still a task-set property.

## 7. What Bob's redesign taught (vs the old star)

1. The star at 1k/5k **widened the C1 sampling signal** enough to catch a
   real render-starvation attribution (legal-hard-2) that the 3k/5k pair
   never showed — the sampling knob's accuracy channel is visible only at
   starved-vs-fed contrasts.
2. Every comparison now owns exactly one attributed flip, each through its
   canonical channel: **rows-window (C1′), profile line (C2′), history
   retention (C3′)** — the cleanest possible statement of "knobs are
   evidence-delivery channels with narrow, real accuracy surfaces."
3. The probe prompt compresses those surfaces (§2) and can even close one
   (§4.1) — evidence channels SUBSTITUTE and INTERACT; a controller must
   treat {rows, stats, history, probes} jointly, not as independent dials.
4. One benchmark bug to file upstream: **wildfire-hard-18** — gold's own
   script yields the loser's signs; the published answer requires indicator
   columns that are anti-coded against their own dictionary (5k arms pass by
   adopting the mis-described indicator; 1k arms fail by being
   gold-code-faithful).

Caveats: chronic gate is old-vintage (a probe-vintage twin pair is the
outstanding calibration debt); single-roll determinism caveats noted per
attributed case; cross-vintage comparisons (pre-probe rows) are directional
only.
