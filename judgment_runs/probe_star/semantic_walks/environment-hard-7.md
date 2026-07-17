# Semantic walk: environment-hard-7 (PROBE-STAR vintage, raw-probe prompt)

## Task + gold answer

**Question:** What was the difference in bacterial exceedance rates (to 2 decimal places) for marine beach samples collected in 2023 between communities with more than 50% EJ populations and those with less than 25% EJ populations?

**Gold answer:** `10.87` (numeric_exact). Gold value `10.874744486461001`.

**Two pairs (both `environment-hard-7` ∈ `chronic_flippers.json`):**
- **C2p** stats>schema @1k: winner `DeltaStats1kD2ProbePrompt` (10.87, PASS) vs loser `Delta1kSchemaOnlyProbePrompt` (0.11, FAIL).
- **C3p** latest>delta @5k: winner `Latest5kSchemaOnlyProbePrompt` (10.87, PASS) vs loser `Delta5kSchemaOnlyProbePrompt` (0.11, FAIL).

**Config diffs (validity gate):** C2p — one lever, the stats bundle `column_stats:true + data_level:2` vs `false/1` (both delta, both 1k/3k char limits). C3p — one knob, `context_mode: latest` vs `delta` (both `column_stats:false, data_level:1`, both 5k/3k). Both pass one-knob gate.

## Gold semantic plan

Source: `solutions/environment/environment-hard-7.py`

| # | Plan item |
|---|---|
| G1 | Load `water-body-testing-2023.csv` (default header); lowercase Beach Name/Community/Violation |
| G2 | Filter `Beach Type Description == 'Marine'` |
| G3 | Load `environmental-justice-populations.csv`; lowercase Municipality |
| G4 | Split EJ on `Percent of population in EJ BGs`: `>50` and `<25` |
| G5 | Inner-merge marine with each EJ subset on `Community == Municipality` |
| G6 | rate = `len(Violation=='yes')/len(subset) * 100` per side; answer = `rate_>50 − rate_<25` → **10.87** |

**The load-bearing detail: G6 multiplies each rate by `* 100`** — the answer is in percentage *points*, not a fraction.

## The four arms compute a bit-identical dataflow

All four traces execute the same semantic pipeline (marine filter → EJ merge on community → `>50`/`<25` split → violation-mean per side → difference). The rendered intermediates prove the underlying value is **bit-identical across every arm**:

- `rate_>50 = 0.1512360639844886`, `rate_<25 = 0.042488619119878605`, **fraction diff = `0.10874744486461`** (rendered verbatim in Delta1k, Latest5k, Delta5k; DeltaStats1k renders the already-×100'd `10.87`).

The ONLY divergence is the **final op's unit convention**:

| arm | final op code (the coin) | rendered | answer |
|---|---|---|---|
| DeltaStats1kD2 (W, C2p) | `high['violation_flag'].mean() * 100` … `round(diff,2)` | `10.87` | **10.87 PASS** |
| Latest5kSchemaOnly (W, C3p) | `hi_rate=…mean()`; `out['diff'] = out['diff'] * 100` | `hi 0.1512… lo 0.0424… diff 10.874744486461` | **10.87 PASS** |
| Delta1kSchemaOnly (L, C2p) | `exceed_rate=('exceed','mean')` — **no ×100** | `rate_gt50 0.1512… rate_lt25 0.0424… diff 0.10874744486461` | **0.11 FAIL** |
| Delta5kSchemaOnly (L, C3p) | `diff = rate['gt50'] − rate['lt25']` — **no ×100** | `difference 0.10874744486461` | **0.11 FAIL** |

Both losers report the fraction `0.10874744486461` rounded → `0.11`; both winners multiply by 100 → `10.874744486461` rounded → `10.87`. `0.10874744486461 × 100 = 10.874744486461` — one value, two conventions.

## First semantic divergence

For every arm the pipeline matches G1–G5 exactly; the divergence is at **G6, the final compute op, on the ×100 convention alone**. No upstream step differs in grain, key, filter, or scope. Minor cosmetic differences (winners merge before splitting vs losers group-by; `mean` vs `len(...)/len(...)`) are algebraically identical and all produce the same fraction.

## Evidence at the decision — the coin is unforced

Nothing rendered in ANY arm signals "express as percentage." The question says "bacterial exceedance rates (to 2 decimal places)" — a "rate" that reads equally as a fraction or a percentage; the two-decimal instruction is satisfied by both `0.11` and `10.87`. The stats profile (C2p winner) and the latest-vs-delta framing (C3p winner) render nothing about output units. The winners' contexts contain no ×100 hint the losers' lacked; the fraction is identical on screen in both. The choice is a free unit-convention pick made in the last line of code.

## Pair verdicts

**C2p (DeltaStats1kD2 > Delta1kSchemaOnly): REJECTED — unit-convention coin flip.** Identical dataflow and bit-identical fraction `0.10874744486461`; the flip is solely `* 100` in the final op. The stats lever (column_stats/data_level) renders no output-unit signal and cannot explain the ×100 choice. Task is a chronic flipper → defaults to variance.

**C3p (Latest5kSchemaOnly > Delta5kSchemaOnly): REJECTED — unit-convention coin flip.** Same mechanism; both arms render the identical fraction, winner alone applies `out['diff'] * 100`. Orthogonal to `context_mode`. Chronic flipper → variance.

**Unit-coin replication:** YES — this vintage reproduces the prior vintage exactly. All arms compute the bit-identical `0.10874744486461`; the sole flip driver is the ×100 percentage-vs-fraction convention in the final operator, unattributable to either context lever.

Artifacts: `system_scratch/{DataflowSystemGPT52DeltaStats1kD2ProbePrompt,DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt,DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt,DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt}/environment-hard-7/`; gold `solutions/environment/environment-hard-7.py` (value 10.874744486461001, repro with `.venv/bin/python`).
