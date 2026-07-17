# biomedical-easy-2 — semantic walk (PROBE-STAR, raw-probe prompt)

**C3p, A-only pair.** Winner **`…Delta5kSchemaOnlyProbePrompt` → 68.5 (✓)**
vs loser **`…Latest5kSchemaOnlyProbePrompt` → 68.1 (✗)**. One knob:
`context_mode` delta vs latest (both schema-only, 5k, raw-probe prompt).
**Chronic flipper** (`*`; confirmed in `chronic_flippers.json`).
**Question:** "What is the average age of patients with serous tumor samples
analyzed in the study?" **Gold: 68.5** (`numeric_exact` — 68.1 fails hard).
Delta 5 st / $0.0162; Latest 5 st / $0.0167 (cost-neutral; not a work gap).

## Gold plan
`solutions/biomedical/biomedical-easy-2.py`: mmc1 meta sheet
`UCEC_CPTAC3_meta_table_V2.1` → filter **`Case_excluded=='No'`** →
`Histologic_type∈{Endometrioid,Serous}` → keep `Serous` → `mean(Age)` = **68.5**.
The `Case_excluded` filter is the sole discriminator: dropping the excluded
serous case(s) lifts the mean from **68.077 → 68.5** (the historic loser
mechanism = *filter omitted*).

## Per-arm divergence table (first divergence = the single process step)
| arm | filter written | Case_excluded applied? | answer | vs gold |
|---|---|---|---|---|
| **Delta (win)** | `Case_excluded != 'yes'` then `Histologic_type=='serous'` | **yes** | 68.5 | ✓ excludes the excluded case |
| Latest (lose) | `Histologic_type=='serous'` only | **no** | 68.077→68.1 | ✗ keeps the excluded case |

- Delta `serous_age`: `df = df[df['Case_excluded'].astype(str).str.strip()
  .str.lower().ne('yes')]` → serous → `mean(Age)` = **68.5**.
- Latest `serous_age_avg`: `ser = df[Histologic_type=='serous']` (no
  Case_excluded guard) → `mean(Age)` = **68.0769** → "68.1".

## Evidence at decision time (both arms, entering the process step)
Both arms rendered the **identical** meta observation — schema-only, the
`Case_excluded` column present at position 3 of the first-25 header list
(`idx  Proteomics_Participant_ID  Case_excluded  Proteomics_TMT_batch …`),
**name+type only, no value distribution**. Neither arm saw that
`Case_excluded` even takes `Yes`/`No` values. The context_mode knob produces
**no rendered difference** here: at a one-process-step task, DELTA's
"new/changed only" and LATEST's "full latest state" both reduce to the same
single loaded table. Delta *self-supplied* `Case_excluded!='yes'` from
column-name reasoning; Latest reasoned only to `Histologic_type`.

## Verdict — REJECTED-method-choice / CHRONIC-VARIANCE
Both arms had identical schema-only evidence; nothing rendered to Delta and
withheld from Latest explains the filter. The win is Delta self-generating a
defensive filter from the column *name*, not a rendered-evidence advantage of
the delta knob. Decisive corroboration: the **mirror pair on the same C3p
knob** — `biomedical-hard-5` Pair 2 (`…5kSchemaOnlyProbePrompt`, context_mode
latest vs delta) — flips the *other way*, with **LATEST** self-supplying
`Case_excluded!='yes'` and DELTA omitting it. Same knob, opposite arm wins →
the Case_excluded filter is a coin-flip across context_mode on this chronic
task. Not attributable to the lever; **variance**.
