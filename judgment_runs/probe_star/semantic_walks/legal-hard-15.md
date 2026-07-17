# legal-hard-15 — semantic walk (PROBE-STAR, all-arm common failure)

## Task + gold

Q: "How many total Identity Theft reports were there in 2024 from cross-state
Metropolitan Statistical Areas?" — numeric_exact, gold **243377**.
Data: 52 per-state CSVs in `data/legal/input/State MSA Identity Theft data/`
(CSN Data Book 2024). Each file: 2 junk header lines, then
`Metropolitan Area,# of Reports` rows, then footer junk. **Every cross-state
MSA is listed once in EACH member state's file with the SAME count**
("Columbus, GA-AL … 1,302" appears in Alabama.csv AND Georgia.csv;
"Washington-Arlington-Alexandria, DC-VA-MD-WV … 19,689" appears in FOUR files).

Gold plan (`solutions/legal/legal-hard-15.py`): read every file
`skiprows=2` + dropna → **concat** → state token = text after comma in
`Metropolitan Area` → cross-state = `-` in token → de-comma `# of Reports` →
**`drop_duplicates()`** → sum over cross-state rows = **243377**.
Un-deduped fingerprint: 94 per-state cross-state rows (43 unique MSAs,
51 repeats) sum to **593524 exactly**. Every arm below answers 593524.

Prior vintage: this task was the program's ONLY render-attributed flip
(levers report, dual-lever convergence) — two independent evidence prongs:
(i) **sampling @5k**: the rendered cross-state/clean table shows literal
duplicate rows; (ii) **profiling @3k-stats**: raw-concat table profile line
`duplicate rows: 359 of 764 (47%)`. Either fixed it.

**Not chronic** (`legal-hard-15` ∉ `judgment_runs/levers_report/chronic_flippers.json`).

## Scoreboard (probe arm vs config-identical no-probe twin)

| probe-star arm | answer | twin (no probe) | twin answer | probe-caused? |
|---|---|---|---|---|
| Delta1kSchemaOnlyProbePrompt (C1'/C2' anchor) | 593524 ✗ | Delta1kSchemaOnly | 593524 ✗ | no — twin fails too |
| Delta5kSchemaOnlyProbePrompt (C1' sampling ray) | 593524 ✗ | Delta5kSchemaOnly | **243377 ✓** | **YES — pass→fail** |
| DeltaStats1kD2ProbePrompt (C2' profiling ray) | 593524 ✗ | DeltaStats1kD2 | 593524 ✗ | no — twin fails too |
| Latest5kSchemaOnlyProbePrompt (C3') | 593524 ✗ | Latest5kSchemaOnly | 593524 ✗ | no — twin fails too |

Reference open channels this vintage: Delta5k/7kSchemaOnly ✓,
DeltaStats3k/5k/7kD2 ✓ (DeltaStats3kD2 still renders
`duplicate rows: 359 of 764 (47%)` and writes
`drop_duplicates(subset=['metropolitan_area','reports'])` into its clean op
→ 401 rows → 43 cross-state → 243377).

## Arm walks

### 1. Delta1kSchemaOnlyProbePrompt — 593524 ✗ (6 steps)

- s0 probe `raw_msa_csv`: head 6 + mid 4 raw lines of first 3 files → header quirk learned (`skiprows=2`).
- s1 `msa_2024`: glob all 52, `read_csv(skiprows=2)`, per-file clean, **adds `state_file`**, concat → 452x3.
- s2 `cross_state_msa_2024` v1: `extractall(r',\s*([A-Z]{2})\b')` grouping bug → 0x0.
- s3 v2 regex `,\s*([A-Z]{2}(?:-[A-Z]{2})+)` → 94x5.
- --> s4 `total_cross_state_reports_2024`: plain `sum()` — **no dedup** → 593524. s5 final.

**Dedup evidence rendered: NO. Dedup attempted: NO.** At the s4 decision the
94-row table rendered exactly 5 rows (1k head/tail elision):

```
0  Columbus, GA-AL Metropolitan Statistical Area   1302  Alabama  GA-AL  2
1  LaGrange, GA-AL Micropolitan Statistical Area    453  Alabama  GA-AL  2
2  Fort Smith, AR-OK Metropolitan Statistical Area  302  Arkansas AR-OK  2
...
92 La Crosse-Onalaska, WI-MN …                      152  Wisconsin …
93 Minneapolis-St. Paul-Bloomington, MN-WI …       5552  Wisconsin …
```

The nearest repeat pair is rows 0↔9 (Columbus under Alabama then Georgia) —
entirely inside the `...` elision. Schema-only ⇒ no stats block at all
(the `msa_2024` block renders 3 head + 3 tail rows and a bare `Schema (3 cols)` line).

### 2. Delta5kSchemaOnlyProbePrompt — 593524 ✗ (4 steps) — the probe regression

- s0 probe `raw_state_msa_files`: file list → 52x1.
- s1 probe `raw_state_msa_sample`: head 6 + mid 6 raw lines of **Alabama.csv only**.
- --> s2 `state_msa_2024_total_cross_state`: ONE fused load op — per-file loop
  `{read_csv(skiprows=2); rename; de-comma; dropna; per-file cross-state regex
  filter; total += int(cross['# of Reports'].sum())}` → output **1x1 scalar**:

```
[state_msa_2024_total_cross_state] Output Table: 1 rows, 1 cols
    total_cross_state_reports_2024
0   593524
```

- s3 final 593524.

**Dedup evidence rendered: STRUCTURALLY IMPOSSIBLE. Dedup attempted: NO.**
No concat table and no cross-state table ever existed as artifacts — the only
renders in the whole trace are 52 file paths, 12 raw preview lines, and the
scalar. The 5k row budget had nothing to spend itself on.

Twin contrast (Delta5kSchemaOnly, PASS): load-concat first (`764x2`, **no
source column**) → clean 452x2 → cross-filter `94x2` **rendered at 5k as
byte-identical duplicate rows** → then a separate total op that opens with
`df.drop_duplicates(subset=['msa'])` → 243377. The twin's build shape
(load → look → clean → look → filter → look → aggregate) is exactly what the
probe beat replaced with probe → confidence → one answer-shaped op.

### 3. DeltaStats1kD2ProbePrompt — 593524 ✗ (4 steps)

- s0 THREE probes batched (file list; head sample — **errors**,
  `RuntimeError: generator raised StopIteration`; mid sample OK — renders
  `"Birmingham, AL …","3,968"` then `"Columbus, GA-AL …","1,302"` inside
  Alabama.csv: a cross-state MSA inside one state's file, hint not proof).
- s1 head-sample fix → header lines.
- --> s2 FOUR calls in ONE batch: `msa_2024_rows` (concat, `skiprows=2`,
  **adds `state_file`** → 452x3) + `cross_state_msa_2024_total` (fused
  filter+sum → scalar) + retire both probes. All renders arrive AFTER the batch.
- s3 final 593524.

**Dedup evidence rendered: YES — but simultaneously with the computed answer.
Dedup attempted: NO.** The s3 context contains, in the same delta:

```
- "state_file" (str): null=0, distinct=52, duplicate_values=400
- "Metropolitan Area" (str): null=0, distinct=401, duplicate_values=51
- "reports_2024" (numeric): null=0, mean=2835, min=76, max=71624
```
(451 distinct of 452 would be clean; **distinct=401 / duplicate_values=51 IS
the repetition signal**, 94−43=51) — and two blocks later:
```
[cross_state_msa_2024_total] … 0	593524
```

Because load and aggregate were submitted in one batch, no decision point
existed between evidence and answer; the agent read the scalar off.
Also note the old table-level trigger could not fire here: the load
pre-cleans (`skiprows=2`) and carries `state_file`, so **full-row duplicates
= 0** — the raw junk table that fires `duplicate rows: 359 of 764 (47%)` in
the passing 3kD2 arm never existed.

Twin contrast (DeltaStats1kD2, FAIL): no probes, but same outcome — it
rendered `duplicate rows: 104 of 764 (14%)` on its raw concat (diluted from
47% because its REAL `__source_file` values break cross-file row identity)
and `- "msa": distinct=401, duplicate_values=51` on its clean table, then
**fused filter+sum anyway** → 593524. The 1k-stats operating point fails
with or without the probe prompt (it also failed in the pre-probe vintage).

### 4. Latest5kSchemaOnlyProbePrompt — 593524 ✗ (6 steps)

- s0 probe `raw_files`: file list.
- s1 `all_state_msa_it`: concat, **no skiprows, adds `__source_file`** → 764x3.
- s2 `clean_msa_it_2024`: drop header/footer/blank rows, de-comma → 452x3.
- s3 `cross_state_msa_2024`: regex filter → 94x3.
- --> s4 `cross_state_total_2024`: plain `sum()` — **no dedup** → 593524. s5 final.

**Dedup evidence rendered: YES, loudest of any arm, at the exact decision
step. Dedup attempted: NO.** The s4 context renders 44 of 94 rows (0–18 +
76–93) with at least SIX visible repeated (msa, count) pairs:

```
0   Columbus, GA-AL Metropolitan Statistical Area                 1302   Alabama.csv
1   LaGrange, GA-AL Micropolitan Statistical Area                  453   Alabama.csv
…
6   Washington-Arlington-Alexandria, DC-VA-MD-WV …               19689   DistrictofColumbia.csv
9   Columbus, GA-AL Metropolitan Statistical Area                 1302   Georgia.csv
10  LaGrange, GA-AL Micropolitan Statistical Area                  453   Georgia.csv
12  Chicago-Naperville-Elgin, IL-IN …                            37486   Illinois.csv
16  Chicago-Naperville-Elgin, IL-IN …                            37486   Indiana.csv
…
82  Washington-Arlington-Alexandria, DC-VA-MD-WV …               19689   Virginia.csv
87  Washington-Arlington-Alexandria, DC-VA-MD-WV …               19689   WestVirginia.csv
90  Winchester, VA-WV Metropolitan Statistical Area                191   WestVirginia.csv
```

The agent wrote a separate, unfused sum op on top of this and did not dedup.
One mitigating mechanism: with `__source_file` attached the repeats are not
byte-identical rows (they differ in the last column), so the mechanical
"duplicate rows → drop_duplicates" reflex that the twin-vintage 94x2 render
triggered has to pass through a semantic step ("same MSA counted once per
member state ⇒ national total double-counts"). It didn't.

Twin contrast (Latest5kSchemaOnly, FAIL): fused clean+filter+sum into one op
at s1 (no cross-state artifact at all) → 593524. Latest has never passed this
task at 5k schema-only in the program — the old sampling lever was Delta-only.

## Why the attributed channel closed (per arm)

- **Delta1kSchemaOnlyProbePrompt → (a) render starvation, pre-existing.**
  Both prongs absent by config: schema-only kills the stats prong; the 1k
  head/tail elision hides all 51 repeat pairs (5 of 94 rows rendered; nearest
  pair 0↔9 elided). Twin fails identically — nothing probe-specific.
- **Delta5kSchemaOnlyProbePrompt → (c) probe beat eliminated the artifact —
  the genuine probe-prompt REGRESSION, and the only real flip here.**
  Non-chronic task, config-identical pair, twin passes via the exact
  old channel (5k render of a 94x2 table with byte-identical duplicate rows
  → `drop_duplicates`). Under the probe prompt the agent probed raw structure
  (file list + ONE file's head/mid), acquired the full load spec, and leapt
  to a single fused per-file load→filter→accumulate op whose only output is
  the 1x1 scalar. Nuance vs the (c) wording: it is not "per-file loads, no
  concat" across several ops — it is per-file **accumulation inside one op**;
  same family, same consequence: the concat/cross-state table that carried
  the duplicate evidence was never materialized, so a 5k budget rendered
  nothing. Probe-confidence → step fusion → evidence-bearing intermediate
  artifacts skipped. Verdict: **ATTRIBUTED (probe-prompt regression)**,
  divergence step = s2 (fused op replaces load-concat).
- **DeltaStats1kD2ProbePrompt → not (a): stats rendered; failure =
  operating point + decision-point compression.** The per-column prong DID
  render at 1k (`Metropolitan Area: distinct=401, duplicate_values=51`), but
  (i) it arrived in the same delta as the precomputed 593524 because load and
  aggregate were batched in one step, and (ii) the table-level
  `duplicate rows:` prong was structurally muted — the probe-era load
  pre-cleans and stamps `state_file`, zeroing full-row duplicates (47% → 0;
  the no-probe twin's real `__source_file` already diluted it to 14%). The
  twin fails the same way without probes, as did pre-probe DeltaStats1kD2 —
  1k-stats was never an open channel. Verdict: common-failure operating
  point; probe beat only removed the last chance to react.
- **Latest5kSchemaOnlyProbePrompt → (b) evidence rendered, unheeded.**
  The sampling prong was OPEN in-context: 44/94 rows with six repeated
  (msa, count) pairs on screen at the sum-writing step; the agent summed
  anyway. Twin also fails (by fusing instead); Latest never passed this
  operating point. The `__source_file` column making repeats
  non-byte-identical is the one render-side change that plausibly weakens
  the dedup reflex, but with repeats this visible the verdict stays
  method-choice/coin on a mode that was already 0-for-this-task.

**Cross-cutting probe-era pattern worth tracking:** 3 of 4 probe arms (and
both failing twins) stamp a provenance column (`state_file`/`__source_file`)
onto the concat. That single habit mutes BOTH old evidence prongs at the
source: full-row duplicates stop existing (stats line 47% → 14% → 0) and
sample-row repeats stop being byte-identical. The still-passing
DeltaStats3kD2 is the accidental counterfactual: its `__source_file` was
all-NaN (`pd.Series([None]*len(df))`), so `duplicate rows: 359 of 764 (47%)`
fired and it wrote dedup into its clean op.

## Verdict

All-arm common failure ≠ all-arm common cause. Three arms
(Delta1k, DeltaStats1k, Latest5k) fail exactly like their no-probe twins —
their operating points never had the channel (1k starvation; 1k-stats
ignored/muted profile; Latest-mode blindness). Exactly ONE arm is a genuine
probe-prompt regression on the attributed channel:
**Delta5kSchemaOnly (pass, dedup after seeing duplicate rows) →
Delta5kSchemaOnlyProbePrompt (fail, probe→fused per-file scalar op, no table
ever rendered)** — ATTRIBUTED, mechanism = probe-confidence step fusion
destroying the evidence-bearing intermediate render.

Repro: `cd ~/Desktop/bobflow/Kramabench && python3 scripts/extract_walk.py
--sut <ARM> --task legal-hard-15`; full-context block dumps via
`scripts/extract_walk.py`'s `op_blocks(step_ctx(step))` on
`system_scratch/<ARM>/legal-hard-15/react_steps.json`.
