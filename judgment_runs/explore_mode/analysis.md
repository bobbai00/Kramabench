# Exploration Mode (no-oracle): Delta vs Latest under File Discovery

Date: 2026-07-11. Arms: `DataflowSystemGPT52{Latest,Delta}Stats3kD2Explore` —
identical configs to the oracle twins (`SmallTableControl` / `FoldControl`),
run with `kb.py --no-oracle`: the prompt carries the domain lake's recursive
glob instead of the task's gold files. Full 104-task GPT-5.2 pass + two
`--all-failed` recovery rounds per arm. Domain lakes: archeology 5 files,
biomedical 8, wildfire 23, environment 37, legal 132, astronomy 1,539.

## Four-way result (104 shared tasks)

| Arm | Pass@0.9 | Cost | Steps | Cache hit |
| --- | ---: | ---: | ---: | ---: |
| Latest · oracle | 80 (76.9%) | $5.04 | 774 | 86.9% |
| Latest · explore | 69 (66.3%) | $5.90 | 943 | 87.6% |
| Delta · oracle | 79 (76.0%) | $5.23 | 694 | 83.9% |
| Delta · explore | **61 (58.7%)** | $6.58 | 920 | 87.9% |

## Finding 1 — the context modes finally separate, and only under exploration

Oracle mode: dead-even (76.9 vs 76.0) — replicating every prior sweep and the
delta_vs_latest_3k audit. Explore mode: **Latest beats Delta by 7.6 points**
(66.3 vs 58.7) at 10% lower cost. Head-to-head flips inside explore mode:
Latest-only 11 vs Delta-only 3 — a directional gap, not flip noise.

Mechanism (consistent with the carrying-cost audit): exploration fills the
trajectory with dead ends — wrong-file loads, discarded probes, mis-parses.
DELTA replays that history verbatim every step; the agent navigates its own
noise. LATEST folds failed exploration into the current healthy DAG and shows
only what survived. With gold files handed over (oracle), there is little
dead-end history and the modes converge. **The aggregated canonical state is
the robust memory under exploration — the dataflow-leverage claim, measured.**

## Finding 2 — the file-discovery tax, per mode

| Mode | Oracle → Explore | Tax |
| --- | --- | ---: |
| Latest | 76.9% → 66.3%, +17% cost, +22% steps | −10.6 pts |
| Delta | 76.0% → 58.7%, +26% cost, +33% steps | −17.3 pts |

Same-task losses concentrate in the large lakes: legal (132 files) 6–8 tasks,
environment 4–6, astronomy 3–4; the 5–8-file domains lose ≤1. Both modes also
GAIN 4 tasks each under exploration — freedom to choose different files
rescues a few chronic oracle failures.

This also de-saturates the benchmark: 20–40% headroom (vs oracle's flat 76%)
makes accuracy-side context experiments meaningful again, per SANA's
observation that prior-guided KramaBench never tested search.

## Finding 3 — the step cap is a minor factor after recovery

Only 3 Latest-explore failures remain cap-bound at 25 steps post-recovery
(cheap Explore40 rerun can quantify the residual). Most early cap-hits and
watchdog kills converted in the recovery rounds.

## Implications

1. Delta's carrying cost is not just a token tax — under exploration it is an
   accuracy liability. The context-mode recommendation becomes
   setting-dependent: LATEST for exploratory regimes.
2. Exploration mode is the natural habitat for the previously-rejected levers:
   orphan probes proliferate (rank-4 material), and the pull tool (E1) faces
   real information gaps instead of an already-sufficient push. Both deserve
   re-testing in this mode.
3. Bridge to LakeQA: same shape (discover → inspect → compute), 100× the
   search space. KramaBench-explore results predict the LakeQA Phase-B design.

Artifacts: `system_scratch/DataflowSystemGPT52{Latest,Delta}Stats3kD2Explore`,
driver logs `logs/explore-20260711_142749/`.

## Finding 4 — cross-paradigm inversion: the code agent's discovery tax is NEGATIVE

| Arm | Oracle | Explore | Tax |
| --- | ---: | ---: | ---: |
| Dataflow · Latest | 76.9% ($5.04) | 66.3% ($5.90) | −10.6 |
| Dataflow · Delta | 76.0% ($5.23) | 58.7% ($6.58) | −17.3 |
| Code agent (3k, guided) | 69.2% ($6.45) | **72.1%** ($6.65) | **+2.9** |

Mode integrity verified: the oracle prompt names gold files; the explore prompt
carries only the lake glob (0 gold-file mentions). The code agent paid +34%
steps (744 → 996) for exploration and converted them into accuracy instead of
losing it. Under exploration the paradigm ordering inverts: Code 72.1 >
DF-Latest 66.3 > DF-Delta 58.7. (Caveat: the code-oracle baseline is the
July-6 rerun-v2 vintage — cross-date pairing — but a sign flip vs −10.6/−17.3
is far beyond vintage noise.)

Mechanism — the flat agent's exploration is structurally cheap at this scale:

1. **Free enumeration.** One `glob`/`os.listdir` inside a single execute_code
   call lists the whole lake. The dataflow agent has no listing primitive — it
   guesses paths or spends operators to probe.
2. **Self-trimming dead ends.** A failed probe leaves ≤3k chars of printed
   output in the transcript; the fresh-namespace-per-call design that costs
   the code agent recomputation in oracle mode becomes an advantage under
   exploration — dead ends are garbage-collected for free. Dataflow probes
   MATERIALIZE as DAG operators: step-expensive and context-polluting
   (the orphan problem), or verbatim-replayed history (Delta).
3. Within-paradigm, aggregated state still wins decisively (Latest ≫ Delta);
   the inversion is about the COST OF PROBING, not the value of state.

## Revised fix agenda (the orphan problem, sharpened)

The deficit is a missing primitive, not a broken paradigm: **exploration
should be tools, commitment should be operators.**

1. `listFiles` / `previewFile` pull-style tools (append-only, like
   inspectResult) — free enumeration + cheap peeking without DAG pollution;
   parity with the code agent's glob, plus provenance once committed.
2. Delete-nudge prompt: after committing to sources, delete abandoned probes
   (agent-native cleanup; append-only event; no cache break).
3. Rank-4 probe retirement retested in explore mode (orphans now abundant).
4. Boundary-aligned exploration fold at the commit point (edit-convergence
   machinery).

This is also exactly LakeQA Phase-B's interface shape (search/download tools
feeding a persistent pipeline) — the KramaBench-explore deficit predicts what
the dataflow agent needs there.

## Finding 5 — inventory-push (list mode) scales badly with lake size

Latest-3k with the FULL domain inventory enumerated in the prompt (+ "only a
small subset is relevant" note), no-oracle. Run 2026-07-11; the Delta-list arm
was intentionally stopped mid-run (Latest three-point sufficed).

| Latest arm | Pass | Cost |
| --- | ---: | ---: |
| oracle (gold files) | 76.9% | $5.04 |
| LIST (full inventory) | **60.6%** | **$8.02** |
| glob (self-enumerate) | 66.3% | $5.90 |

Per-domain: at legal (132 files) LIST beats glob on accuracy AND cost
(24/30 $1.71 vs 23/30 $1.89) — the crossover point. At astronomy (1,539
files) LIST collapses: 3/12 at $3.39 (3.5× glob's cost, worse accuracy) — the
~20k-token inventory rides in every step's prompt, diluting attention and
encouraging name-based picking over probing.

Design law (same push/pull economics as E1, on the discovery axis): a pushed
inventory pays its carrying cost every step whether consulted or not. Below
~100 files, push the list; above, the agent must pull (enumerate/preview on
demand). By extrapolation a million-file lake (LakeQA) makes inventory-push
impossible — search/preview TOOLS are the only scalable design. The
evidence chain E1 → explore → list-mode grounds the pull-primitive agenda.
