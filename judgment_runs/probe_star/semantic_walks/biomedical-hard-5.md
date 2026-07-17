# biomedical-hard-5 — semantic walk (PROBE-STAR, raw-probe prompt)

**Two pairs** on one task. Winners → **2.6563 (✓)**, losers → **2.4241 (✗)**.
**Chronic flipper** (`*`). **Question:** "Median number of variants per Mbp for
the serous tumor samples. Round to 4 dp." **Gold: 2.6563.**

- **Pair 1 (C2p, stats>schema@1k):** winner `…DeltaStats1kD2ProbePrompt`
  ($0.0292, 6 st) vs loser `…Delta1kSchemaOnlyProbePrompt` ($0.0216, 5 st).
  One knob: `column_stats` false→true (+`data_level` 1→2).
- **Pair 2 (C3p, latest>delta@5k):** winner `…Latest5kSchemaOnlyProbePrompt`
  ($0.0309, 7 st) vs loser `…Delta5kSchemaOnlyProbePrompt` ($0.0259, 6 st).
  One knob: `context_mode` latest vs delta (both schema-only).

## Gold plan
`solutions/biomedical/biomedical-hard-5.py`: mmc1 meta table → filter
`Case_excluded=='No'` AND `Histologic_type∈{Endometrioid,Serous}`; take Serous
idx; from mmc7 **`B-APM subtypes`** sheet take those idx; `2**Log2_variant_per_Mbp`;
`median` → **2.6563**.

**Mechanism (verified against the data):** 14 serous rows; `Case_excluded`
No=12 / Yes=2 (**S043, S104**). S104's variant is NaN (dropped); **S043 =
1.4954 variants/Mbp, below the median**. Keep S043 → 13 valid rows →
**median 2.4241**; drop it (`Case_excluded=='No'`) → 12 rows → **2.6563**. Every
arm here computes `Log2_variant_per_Mbp` straight from **mmc1** (numerically
equal to gold's mmc7 route — median commutes with `2**x`), so the sole
discriminator is **the Case_excluded filter**. This is the task's stated
"keeping excluded S043" failure, reached via a missing filter rather than
sheet-concat — same excluded case, same 2.4241.

## Per-arm divergence table (first divergence = the serous-filter step)
| pair | arm | filter written | rows | answer | vs gold |
|---|---|---|---|---|---|
| 1 | **stats (win)** | `Case_excluded=='no' & Histologic_type=='serous'` | 12 | 2.6563 | ✓ excludes S043 |
| 1 | schema (lose) | `Histologic_type=='serous'` only | 13 | 2.4241 | ✗ keeps S043 |
| 2 | **latest (win)** | `Tumor_Normal=='tumor' & Case_excluded!='yes' & serous` | 12 | 2.6563 | ✓ excludes S043 |
| 2 | delta (lose) | `Histologic_type=='serous'` only | 13 | 2.4241 | ✗ keeps S043 |

## Evidence at decision time
- **Pair 1 — winner (stats), entering the filter step**, rendered column stats:
  `"Case_excluded" (str): null=0, distinct=2, top_5={"No"=144,"Yes"=9}` and
  `"Histologic_type" (str): distinct=4, top_5={"Endometrioid"=86,"Serous"=14,…}`.
  The **`Yes`=9** count directly surfaces that excluded cases exist → the arm
  wrote `Case_excluded=='no'`.
- **Pair 1 — loser (schema-only)** saw only `Case_excluded (str)` (name+type,
  **no value distribution**). No signal that any case is excluded → filter
  omitted → S043 retained.
- **Pair 2 — both arms schema-only**: neither renders `Case_excluded` values
  (confirmed: the LATEST arm's extra 5-row preview shows the column header but
  no `Yes`). The LATEST winner *self-generated* `Case_excluded!='yes'` +
  `Tumor_Normal=='tumor'` from **column-name reasoning** (and a more elaborate
  7-step exploration); the DELTA loser reasoned only to `Histologic_type`.

## Verdicts
- **Pair 1 (C2p): ATTRIBUTED to the stats knob.** Winner's rendered
  `Case_excluded top_5={No=144,Yes=9}` explains applying the filter; loser's
  schema-only render lacks any excluded-case signal, explaining the omission —
  the one knob (`column_stats`) *is* exactly the rendered difference. Both accept
  clauses met. This is the mirror of wildfire-hard-12 (same `DeltaStats1kD2`
  arm): **stats helps precisely when the rendered stat is answer-relevant**
  (here it names the 9 excluded cases), and is neutral/harmful when it is not.
- **Pair 2 (C3p): REJECTED-method-choice / CHRONIC-VARIANCE.** Both arms had
  **identical schema-only evidence** — neither saw that any case is excluded.
  The LATEST win is self-supplied defensive filtering from column semantics plus
  an extra preview step, not a rendered-evidence advantage of the latest knob.
  No rendered difference explains the flip; on a chronic task this is variance,
  not an attributable lever effect.
