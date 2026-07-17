# Deep-dive: environment-hard-7 (PROBE-STAR vintage) — CHRONIC, ×100 unit-convention coin

## Task
Q: What was the difference in bacterial exceedance rates (to 2 decimal places) for marine beach samples collected in 2023 between communities with more than 50% EJ populations and those with less than 25% EJ populations?

D:
- `data/environment/input/water-body-testing-2023.csv` — one row per beach water sample. Real rows:
  ```
  Community Code,Community,County Code,County Description,Year,Sample Date,Beach Name,Beach Type Description,Organism,Indicator Level,Violation
  242,Provincetown,001,Barnstable,2023,2023-07-10 00:00:00,333 Commercial Street,Marine,Enterococci,5,No
  242,Provincetown,001,Barnstable,2023,2023-08-28 00:00:00,333 Commercial Street,Marine,Enterococci,87,No
  ```
  Relevant columns: `Beach Type Description` (`Marine`/… — the marine filter), `Community` (merge key vs EJ `Municipality`), `Violation` (the exceedance flag). Quirk: `Violation` is title-cased `No`/`Yes`; gold lowercases before matching `== 'yes'`.
- `data/environment/input/environmental-justice-populations.csv` — one row per municipality. Real rows:
  ```
  OBJECTID,Municipality,EJ criteria,Number of EJ block groups,Total number of block groups,Percent of EJ block groups,Population in EJ block groups,Total population,Percent of population in EJ BGs
  1,Acton,M,3,16,18.8,5621,24021,23.4
  3,Adams,I,6,8,75.0,6761,8166,82.8
  ```
  Relevant columns: `Municipality` (merge key vs sample `Community`), `Percent of population in EJ BGs` (the >50 / <25 split). Quirk: this column is already a percentage (0–100), not a fraction; the EJ split is on the number, but it does not set the ANSWER's units.

## Solution
solutions/environment/environment-hard-7.py — reproduces `10.874744486461001`:

```
load(water-body-testing-2023.csv) → lower(Beach Name, Community, Violation)
   → filter(Beach Type Description == 'Marine')  ───────────────┐
                                                                ├─ inner-merge(Community == Municipality), per EJ side
load(environmental-justice-populations.csv) → lower(Municipality)
   → split: EJ>50 / EJ<25  (Percent of population in EJ BGs) ────┘
   → rate_side = len(Violation=='yes') / len(side) * 100    ← ×100 to percentage POINTS
   → answer = rate_>50 − rate_<25 = 10.87
```
- load specs: both default header/sep.
- predicate: `Beach Type Description == 'Marine'`; EJ split `>50` and `<25` on `Percent of population in EJ BGs`.
- key: `Community == Municipality` (both lowercased), inner merge per EJ side.
- grain: exceedance rate per EJ side = share of `Violation=='yes'`.
- final compute: **`* 100`** on each rate → difference in percentage points = **10.87** (the load-bearing detail).

## What Delta5kSchemaOnly (mode X, loser) does
- load marine 2023 → merge to EJ → split `>50`/`<25` → violation-rate per side. Matches the solution through the merge and grain (G1–G5).
- final op: `diff = rate['gt50'] − rate['lt25']` — **no ×100**.
- --> renders `difference 0.10874744486461`, rounds → **`0.11` FAIL**. <— the only divergence from the solution: gold multiplies each rate by 100 (percentage points); this arm reports the raw fraction.

## What Latest5kSchemaOnly (mode Y, winner) does
- identical pipeline through G1–G5 (marine filter → EJ merge on community → >50/<25 split → violation-mean per side).
- final op: `out['diff'] = out['diff'] * 100` → renders `hi 0.1512… lo 0.0424… diff 10.874744486461`, rounds → **`10.87` PASS**.
- C2p mirror (same task, other pair): `DeltaStats1kD2ProbePrompt` writes `high['violation_flag'].mean() * 100 … round(diff,2)` → `10.87` PASS — the same coin landing right under the stats lever instead of the latest lever.

## Why Y succeeded but X failed
Evidence is identical. Every arm computes and renders the bit-identical fraction `0.10874744486461` (verbatim in Delta5k, Latest5k, and Delta1k; DeltaStats1k renders the already-×100'd `10.87`). `0.10874744486461 × 100 = 10.874744486461` — one value, two conventions; the sole flip driver is `* 100` in the final operator.

Nothing rendered in ANY arm signals "express as percentage." The question asks for "bacterial exceedance rates (to 2 decimal places)" — a "rate" reads equally as a fraction (`0.11`) or a percentage (`10.87`), and the two-decimal instruction is satisfied by both. The stats bundle (C2p's lever) renders no output-unit signal; the latest-vs-delta framing (C3p's lever) renders no output-unit signal. The winners' contexts contain no ×100 hint the losers' contexts lacked — the fraction is identical on screen. This is a free unit-convention pick made in the last line of code, not a context effect. Method-choice / chronic-variance, not a lever.

Both pairs → **CHRONIC** (task ∈ chronic_flippers.json):
- C3p (Latest5kSchemaOnly > Delta5kSchemaOnly): REJECTED — ×100 coin, orthogonal to `context_mode`.
- C2p (DeltaStats1kD2 > Delta1kSchemaOnly): REJECTED — same ×100 coin, orthogonal to the stats bundle (`column_stats`/`data_level`).

Vintage-invariance of the convention coin: this is the **third vintage in a row** the ×100 percentage-vs-fraction coin recurs on environment-hard-7, and it recurs across two DIFFERENT levers (stats @1k in C2p, latest/delta @5k in C3p). A coin that flips the same way independent of context vintage and of which knob is under test is not attributable to either lever — convention coins are vintage-invariant.

Artifacts: `system_scratch/DataflowSystemGPT52{Latest5kSchemaOnly,Delta5kSchemaOnly,DeltaStats1kD2,Delta1kSchemaOnly}ProbePrompt/environment-hard-7/`; gold `solutions/environment/environment-hard-7.py` (10.874744486461001, repro with `.venv/bin/python`).
