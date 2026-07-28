# Randomness floor — gpt-5-mini single-shot replicates

Goal: quantify run-to-run M1 randomness. Each knob (anchor + C1–C6) run as
2 extra config-identical SUTs (Replicate1/Replicate2), gpt-5-mini only,
single round0, NO retries. Compared against the original round0 (also raw
single-shot) → up to 3 independent raw samples per knob.

## Raw single-shot M1 (3 samples per knob where available)

| knob | samples (orig round0, Rep1, Rep2) | mean | range | std |
|---|---|---|---|---|
| anchor Delta1k | 62.3, 56.0, 53.6 | 57.3 | 8.7 | 3.7 |
| C1 Delta5k | 69.4, 61.4, 58.0 | 62.9 | 11.4 | 4.8 |
| C2 DeltaStats1k | 62.6, 55.6, 60.2 | 59.5 | 7.0 | 2.9 |
| C3 Latest1k+code | 53.4, 61.5, 58.6 | 57.8 | 8.1 | 3.4 |
| C4 DeltaStats5k | 64.4, 54.1, 57.3 | 58.6 | 10.3 | 4.3 |
| C5 DeltaStats2k | 70.7, 61.4 (Rep2 quota-corrupt) | 66.0 | 9.3 | 4.7 |
| C6 LatestStats1k | 65.0 (both reps quota-corrupt) | — | — | — |

## Finding

**The single-shot randomness floor is large: mean range 9.1 pts (7.0–11.4),
std ~3–5 pts per knob.** Identical config, same 104 tasks, re-run → M1 swings
±~4–5 pts from pure run-to-run nondeterminism (agent sampling + judge temp
where applicable + engine timing).

**Implication:** the recovered knob M1 deltas from the main study (sampling
C1 +1.7, stats −1.3, latest ≈0) are **well inside this ±4–5 pt noise band**.
Single-shot M1 cannot distinguish the knobs — only recovery-equalization
(retries collapse transient fails) + the mechanism metrics (M3/M5 evidence
delivery) separate them. This is the quantitative justification for the
recovery-first + twin-noise protocol.

## Incomplete (OpenAI quota exhausted mid-run, ~11:00 2026-07-24)
- HTTP 429 `RateLimitError: You exceeded your current quota` on gpt-5-mini.
- 3.03 arms captured the error string as their answer → scored 0 (excluded above):
  DeltaStats2kD2Replicate2 (104), LatestStats1kD2Replicate1 (104),
  LatestStats1kD2Replicate2 (104), + 3 tasks in DeltaStats2kD2Replicate1.
- Re-run these once OpenAI quota/billing is restored to complete C5 (3rd
  sample) and C6 (2nd/3rd samples). Stack itself is healthy.

## FINAL — all 7 knobs, 3 raw single-shot samples (quota-refill recovery complete)

| knob | samples | mean | range | std |
|---|---|---|---|---|
| anchor Delta1k | 62.3, 56.0, 53.6 | 57.3 | 8.7 | 3.7 |
| C1 Delta5k | 69.4, 61.4, 58.0 | 62.9 | 11.4 | 4.8 |
| C2 DeltaStats1k | 62.6, 55.6, 60.2 | 59.5 | 7.0 | 2.9 |
| C3 Latest1k+code | 53.4, 61.5, 58.6 | 57.8 | 8.1 | 3.4 |
| C4 DeltaStats5k | 64.4, 54.1, 57.3 | 58.6 | 10.3 | 4.3 |
| C5 DeltaStats2k | 70.7, 70.2, 68.6 | 69.8 | 2.1 | 0.9 |
| C6 LatestStats1k | 65.0, 64.7, 62.0 | 63.9 | 3.0 | 1.3 |

## Run-condition confound (important)

Variance splits cleanly by HOW the replicate ran, not by knob:
- anchor/C1–C4 reps ran inside the 14-arm, ~10.5h continuous orch5 pool
  → range **9.1 pts**, std **3.8**.
- C5/C6 reps ran in short, fresh ≤6-arm pools (orch4b + post-refill orch5b)
  → range **2.6 pts**, std **1.1**.

Instantaneous concurrency was P4 in BOTH — the difference is run duration/
engine drift: over a 10 h continuous pool, late tasks accumulate transient
watchdog-kills/timeouts that a NO-RETRY single-shot cannot recover, inflating
apparent variance. Fresh short runs sit near the true model-sampling floor.

**Revised reading of the randomness floor:**
- Intrinsic model-sampling floor (fresh runs): **~±1 pt** (C5/C6).
- Long unrecovered-run floor (10 h heavy pool): **~±4–5 pt** (anchor/C1–C4),
  dominated by transient-failure accumulation, NOT model nondeterminism.

**Implication:** the recovered knob M1 deltas (sampling +1.7, stats −1.3,
latest ≈0) are within the LONG-RUN floor but comparable to the intrinsic ~±1 pt
floor — so on clean, recovered, same-era runs a real ~1–2 pt sampling edge is
plausible, while the big single-shot swings are a run-condition artifact.
This is why the protocol matters: recovery + same-era matched runs collapse the
±4–5 pt artifact and expose the ±1 pt regime where knob effects live.

## CLEAN TRIPLES — Rep0/Rep1/Rep2 (3 same-protocol single-shot runs, no recovered/round0-slice numbers)

Replicate0 = fresh re-run of the 7 base arms as new SUT names (base round0 traces
were overwritten by their recovery rounds). All three now clean single-shot,
full traces.

| knob | Rep0 | Rep1 | Rep2 | mean | range | std |
|---|---|---|---|---|---|---|
| anchor Delta1k | 67.1 | 56.0 | 53.6 | 58.9 | 13.5 | 5.9 |
| C1 Delta5k | 62.4 | 61.4 | 58.0 | 60.6 | 4.4 | 1.9 |
| C2 DeltaStats1k | 63.1 | 55.6 | 60.2 | 59.6 | 7.5 | 3.1 |
| C3 Latest1k+code | 68.2 | 61.5 | 58.6 | 62.8 | 9.6 | 4.0 |
| C4 DeltaStats5k | 67.2 | 54.1 | 57.3 | 59.5 | 13.1 | 5.6 |
| C5 DeltaStats2k | 61.5 | 70.2 | 68.6 | 66.8 | 8.7 | 3.8 |
| C6 LatestStats1k | 62.0 | 64.7 | 62.0 | 62.9 | 2.7 | 1.3 |

Overall: mean range **8.5 pts**, mean std **3.6**.
Per-run means: **Rep0 64.5 · Rep1 60.5 · Rep2 59.8**.

## Refined conclusion — variance is RUN-LEVEL, not per-task

The clean triples reveal a **run-level offset**: Rep0 (fresh restarted stack,
7-arm pool) averages **+4–5 pts** above Rep1/Rep2 (which ran in the 14-arm,
10.5 h orch5 pool for anchor–C4). The biggest per-knob ranges (anchor 13.5,
C4 13.1) are exactly where Rep0's fresh-run score sits far above the two
heavy-pool reps.

So run-to-run M1 variance is dominated by **which run** (pool load, engine
freshness, run duration) — a whole-run offset of ±4–5 pts — NOT by per-task
model sampling. Where all three reps happen to share conditions (C1, C6) the
range collapses to 2.7–4.4 pts, near the true per-task floor (±1–2 pts).

**Bottom line for the knob study:** the recovered knob deltas (sampling +1.7,
stats −1.3, latest ≈0) are smaller than the ±4–5 pt run-level swing but on the
order of the per-task floor. Only *same-run, matched, recovery-equalized*
comparison controls the run-level offset — which is exactly the protocol used
for the headline M1/M3/M5 results. Single-shot cross-run numbers cannot rank
the knobs; the protocol can.

## CA-guided code agent (gpt-5-mini via litellm) — 5 single-shot samples per cap

Guided code-agent (custom data-pitfalls prompt), stdout-preview caps 1k vs 5k.
Base class = sample 1, Replicate1-4 = samples 2-5. Same protocol as the
dataflow replicates (single round0, NO retries), own P4 pool run 2026-07-27
(concurrent with the dataflow rep pool — code agent never touches the engine).

| arm | samples | mean | range | std |
|---|---|---|---|---|
| CA-guided 1k | 56.5, 60.4, 55.4, 58.1, 54.2 | 56.9 | 6.2 | 2.2 |
| CA-guided 5k | 54.8, 62.0, 59.7, 64.3, 56.7 | 59.5 | 9.5 | 3.4 |

Read: 5k mean +2.6 over 1k, but ranges overlap heavily (1k spans 54.2-60.4,
5k spans 54.8-64.3) — the stdout-cap knob's effect on the code agent is within
single-shot noise at n=5, same conclusion as the dataflow rows knob. Code-agent
single-shot noise (std 2.2-3.4) is comparable to the dataflow per-task floor.

## FINAL — 8 knobs x 5 clean single-shot reps (gpt-5-mini, full 104, no retries)

C7 = Delta2kSchemaOnly (new mid-point of the rows axis). Rep3/Rep4 run 2026-07-26/27
in one 19-arm global pool; C7-Rep4's last 15 tasks re-run + re-scored after an
operator kill error (corrected OVERALL 65.2).

| knob | 5 reps | mean | range | std |
|---|---|---|---|---|
| anchor Delta1k | 67.1, 56.0, 53.6, 64.6, 59.5 | 60.2 | 13.5 | 5.1 |
| C1 Delta5k | 62.4, 61.4, 58.0, 70.8, 68.6 | 64.2 | 12.8 | 4.7 |
| C2 DeltaStats1k | 63.1, 55.6, 60.2, 63.9, 65.5 | 61.7 | 9.9 | 3.5 |
| C3 Latest1k+code | 68.2, 61.5, 58.6, 74.7, 64.5 | 65.5 | 16.1 | 5.6 |
| C4 DeltaStats5k | 67.2, 54.1, 57.3, 66.2, 71.7 | 63.3 | 17.6 | 6.6 |
| C5 DeltaStats2k | 61.5, 70.2, 68.6, 66.3, 66.5 | 66.6 | 8.7 | 2.9 |
| C6 LatestStats1k | 62.0, 64.7, 62.0, 64.7, 62.1 | 63.1 | 2.7 | 1.3 |
| C7 Delta2k | 65.4, 64.3, 71.8, 62.6, 65.2 | 65.9 | 9.2 | 3.1 |

Overall: mean range 11.3 pts (2.7–17.6), mean std 4.1.

### Axis reads at n=5 (means)

- **Rows axis (schema-only): 1k 60.2 → 2k 65.9 → 5k 64.2.** The starved 1k
  anchor is worst; 2k already captures the benefit (5k adds nothing over 2k).
  1k→2k gap (+5.7) is comparable to per-knob std (3–5) — suggestive, not
  conclusive at n=5, but the direction matches every prior signal (M3/M5,
  recovered M1, gpt-5.2 star).
- **Stats axis (D2): 1k 61.7 → 2k 66.6 → 5k 63.3.** Same shape: 2k best.
  Stats-on vs schema-only at matched cap stays within noise (1k: 61.7 vs 60.2;
  2k: 66.6 vs 65.9; 5k: 63.3 vs 64.2) — stats adds nothing at any cap.
- **Latest vs delta at 1k**: C3 65.5 / C6 63.1 vs anchor 60.2 — latest-mode
  arms sit above the delta 1k anchor on means, but C3's spread (16.1) is the
  2nd-widest; treat as noise-compatible.
- Single-shot std remains 3–6 pts for most knobs (C6's 1.3 the outlier);
  5 reps tighten means but per-run swings (e.g. C3 58.6→74.7, C4 54.1→71.7)
  keep single-run comparisons unreliable — the recovery-equalized matched
  protocol remains the only way to rank knobs.

### Interesting emergent read

The two 2k arms (C5 66.6, C7 65.9) top both axes with the SMALLEST spreads
among delta arms (8.7/9.2). At mini, ~2k result chars looks like the sweet
spot: enough rows to deliver evidence, not enough to bloat context — and more
stable run-to-run than either 1k (starved, coin-flippy) or 5k (long contexts,
more timeout exposure).

## C8 — Latest 5k + code-in-snapshot, 5 single-shot reps (run 2026-07-27)

| arm | 5 reps | mean | range | std |
|---|---|---|---|---|
| **C8 Latest5k+code** | 68.4, 66.0, 72.7, 65.6, 72.2 | **69.0** | 7.1 | 3.0 |
| C3 Latest1k+code | 68.2, 61.5, 58.6, 74.7, 64.5 | 65.5 | 16.1 | 5.6 |
| C1 Delta5k schema | 62.4, 61.4, 58.0, 70.8, 68.6 | 64.2 | 12.8 | 4.7 |

Reads:
- **C8 is the best-mean arm of the entire mini replicate study (69.0)** — above
  the previous top C5 DeltaStats2k (66.6) — and with a TIGHT spread (7.1 range,
  std 3.0, all 5 reps ≥65.6).
- **Latest-code axis (1k→5k): 65.5 → 69.0 (+3.5)** with spread collapsing
  16.1 → 7.1. Wider sampling doesn't just raise latest+code's mean, it
  stabilizes it — consistent with the folded snapshot benefiting most from
  richer per-operator renders (no history to fall back on).
- **Latest+code vs delta at 5k: 69.0 vs 64.2 (+4.8)** — the largest paired-arm
  gap in the study, and C8's worst rep (65.6) beats C1's mean. Still n=5
  single-shot, but unlike every other knob comparison, this one is directional
  AND consistent (5/5 C8 reps ≥ C1 mean).
- Combined with code-in-snapshot's near-tautological M4 lift, the picture:
  at mini, LATEST + code + wide sampling is the strongest context recipe —
  snapshot mode stops paying its evidence-eviction tax once the render is
  wide enough and the agent's own code stays visible.
