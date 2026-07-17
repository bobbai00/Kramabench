# environment-hard-8 — semantic flip walk (PROBE-STAR vintage)

Q: What % of Boston-Harbor-beach samples that FAILED the standard (Enterococcus > 104)
had rainfall in the prior 24h (1-Day Rain > 0)?  GOLD = **54.03** (= 1119/2071).

Arms:
- **C1p** DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt  — WINNER (54.03)
- **C2p** DataflowSystemGPT52DeltaStats1kD2ProbePrompt     — WINNER (54.03)
- **L**   DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt   — LOSER  (51.49)

Pairs under test: C1p 5k>1k (chronic*) · C2p stats>schema @1k (chronic*).

---

## Gold semantic plan
Each of the 8 beach CSVs is a wide multi-STATION sheet: row0 title, row1 station names,
row2 the real header, and every station repeats a `Tag,Enterococcus` column pair.
Station counts (from raw row1):

| beach | stations | row1 names |
|---|---|---|
| constitution | **3** | North, Middle, South |
| carson | **2** | I Street, McCormack Bathhouse |
| pleasure_bay_and_castle_island | **3** | Pleasure Bay @ Broadway, @ Flagpole, Castle Island Playground |
| wollaston | **4** | Milton Road, Channing Street, Sachem Street, Rice Road |
| malibu / m_street / city_point / tenean | 1 each | single station |

Plan: load each sheet (header=row2) → **MELT EVERY station's Enterococcus column** to long
(→ 21986 samples) → filter Enterococcus > 104 (→ 2071 "failed") → among failed count
1-Day Rain > 0 (→ 1119) → pct = 1119/2071 = 54.03.
Prior-vintage failure mode = keep only `candidates[0]` (the bare first `Enterococcus`
column) per sheet, dropping the extra stations.

---

## Arm walks

### C1p (WINNER) — staged, probed the 4-station beach
| step | action | semantics | vs gold |
|---|---|---|---|
| S0 | raw probe txt, malibu, city_point, **wollaston** | previews rows | — |
| S1 | `load_beaches_csvs` concat all 8, header=2 → 10956×13 | pandas auto-suffixes dup cols → `Enterococcus`,`.1`,`.2`,`.3` | load OK |
| S2 | `beach_samples_long` melts the **4 hardcoded pairs** (`''`,`.1`,`.2`,`.3`) → 21986 | **melt-all** | ✔ plan step |
| S3 | `failed_samples` > 104 → 2071 | filter | ✔ |
| S4 | `failed_rain_pct` 1119/2071 | = **54.03** | ✔ |
No divergence.

### C2p (WINNER) — staged, probed the 3-station beach
| step | action | semantics | vs gold |
|---|---|---|---|
| S0 | raw probe txt, **constitution** | previews rows | — |
| S1 | 8 separate loads, header=2 (rendered col-counts: constitution 10, carson 8, pleasure_bay 10, wollaston 12, rest 6) | per-beach schema | load OK |
| S2 | `all_samples_long` `to_long()` melts **every col `startswith('enterococcus')`** per beach → 21986 | **melt-all (prefix match)** | ✔ plan step |
| S3 | `failed_samples` > 104 → 2071 | filter | ✔ |
| S4 | `failed_with_rain_24h_pct` 1119/2071 | = **54.03** | ✔ |
No divergence.

### L (LOSER) — single-shot batch, probed only 1-station + 2-station beaches
| step | action | semantics | vs gold |
|---|---|---|---|
| turn1 | raw probe txt, **malibu (1-stn)**, **carson (2-stn)** | previews rows | — |
| turn2 | **12 ops in ONE turn (blind)**: 8 loads + carson melt + all_samples + failed + pct | see below | **DIVERGE** |
| → | `carson` op melts carson's 2 stations (`ent_positions` loop) — the ONE multi it probed | 2-stn melt | partial ✔ |
| → | `all_samples` keeps only bare `'Enterococcus'` for [malibu,constitution,tenean,m_street,wollaston,city_point]; **omits pleasure_bay entirely**; appends carson's 2 → **10096 rows** | **keep candidates[0]** + drop a whole beach | ✗ plan step |
| turn3 | filter > 104 → 1037 → 534/1037 = **51.49** | wrong denom/numer | ✗ |
**First divergence = the `all_samples` op** (turn2): drops constitution `.1/.2`, wollaston
`.1/.2/.3`, and all of pleasure_bay — falls short of "melt EVERY station's Enterococcus."

---

## Rendered multi-station evidence (quoted, per arm)

Raw-probe (probe beat), at code-authoring time:
- **C1p** wollaston preview: `,,,,Milton Road,,Channing Street,,Sachem Street,,Rice Road,`
  then `Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus,Tag,Enteroc...` → **4-station signal PRESENT**.
- **C2p** constitution preview: `,,,,North,,Middle,,South,` then
  `Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus,Tag,Enterococcus` → **3-station signal PRESENT**.
- **L** carson preview: `,,,,I Street,,McCormack Bathhouse,` then
  `Date,1-Day Rain,2-Day Rain,3-Day Rain,Tag,Enterococcus,Tag,Enterococcus` → **2-station signal PRESENT** (only carson; malibu single). The 3 widest multi-beaches (constitution/pleasure_bay/wollaston) were **never probed**.

Post-batch load observations (returned to L AFTER it had already written all 12 ops):
- `[constitution] Output 1880x11: ... Tag Enterococcus Tag.1 Enterococcus.1 Tag.2 Enterococcus.2 beach`
- `[wollaston] Output 1904x13: ... Tag Enterococcus Tag.1 Enterococcus.1 Tag.2 Enterococcus.2 Tag.3 Enterococcus.3 beach`
- `[pleasure_bay_castle_island] Output 860x10: ... Tag Enterococcus Tag.1 Enterococcus.1 Tag.2 Enterococcus.2`

→ L's context **did render** the full multi-station schema — but only after it committed the
`candidates[0]` assembly in the same blind turn, and it went straight to the final answer
(turn3) without revising. So the render was not starved; the code was authored ahead of it.

---

## Verdict: REJECTED-method-choice → CHRONIC/VARIANCE

- **Not dual-lever convergence.** Both winners fix the SAME fact (melt all `Enterococcus.N`)
  through the SAME channel — the rendered duplicate column-names — differing only in melt
  code (C1p hardcodes 4 pairs; C2p prefix-matches `enterococcus*`). No rows-vs-profile split;
  the stats block was NOT C2p's deciding channel (it melts off column names, not stats).
- **Not lever-attributable.** C2p runs at the loser's IDENTICAL 1k budget (differs only
  stats-vs-schema) and still wins → the 5k-rows lever (C1p) and the stats lever (C2p) are
  falsified as the cause. A 1k arm wins here; a 1k arm loses here.
- **Loser not evidence-starved.** Multi-station was in its raw probe (carson 2-stn) and fully
  rendered in its post-batch load obs (wollaston 13 cols incl `Enterococcus.1/.2/.3`). It even
  correctly melted the one beach it probed (carson). The miss is a self-inflicted planning
  slip: it single-shot-batched the whole pipeline after under-sampling probes (only 1-of-4
  multi-beaches), so it generalized "carson special, everything else single-station" and wrote
  `keep candidates[0]` (plus dropped pleasure_bay) blind.
- Failure driven by stochastic probe-target selection + stage-vs-batch planning, both
  orthogonal to the knob under test → assign to variance, consistent with both pairs' chronic* tags.
