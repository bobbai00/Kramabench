# Semantic walk: environment-hard-12 (PROBE-STAR vintage, raw-probe prompt)

## Task + gold answer

**Question:** Which Boston Harbor beach had the highest number of samples that failed to meet swimming standards when there was no rainfall in the preceding three days? A sample meets the standard if it contains fewer than 104 counts of Enterococcus per 100 mL.

**Gold answer:** `Wollaston Beach` (string). Source `solutions/environment/environment-hard-12.py`.

**Pair (C3p, A-only):** winner `Delta5kSchemaOnlyProbePrompt` (`Wollaston Beach`, PASS — `llm_paraphrase:1`) vs loser `Latest5kSchemaOnlyProbePrompt` (`Wollaston`, FAIL — `llm_paraphrase:0`). **NOT** in `chronic_flippers.json`. Prior vintages: both 3k arms PASSED.

**Config diff (validity gate):** one knob — `context_mode: delta` vs `latest` (both `column_stats:false, data_level:1`, both 5k char limit). Passes one-knob gate.

## Gold semantic plan

Loop over all 8 harbor beach datasheets → `prepare_beach_datasheet` (skiprows=1, header=[0,1], ffill station names, melt every Tag/Enterococcus site column into tidy one-row-per-(Date,site)) → filter `3-Day Rain == 0` → filter `Enterococcus > 104` → count rows per beach → argmax = **Wollaston Beach** (92 exceedances). Answer is a beach NAME; robust to counting-grain quirks, so the load-bearing requirement here is **emitting the canonical name "Wollaston Beach".**

## Walk: Delta5kSchemaOnly (WINNER — PASS, 8 steps, $0.088)

| step | op | semantics | plan-item |
|---|---|---|---|
| 0 | 4 raw probes incl. `raw_boston_harbor_beaches` | previews `boston-harbor-beaches.txt` → renders canonical list `0 Constitution Beach … 8 Wollaston Beach` | naming source |
| 1–2 | 8 `read_csv(skiprows=2)` loads | all 8 datasheets loaded, multi-site Enterococcus cols visible | load ✓ |
| 3–4 | `pleasure_bay…_long`, `all_beaches_long` | melts multi-site sheets to sample-rows; labels beaches with **full canonical names** `mk(city_point,'City Point Beach')`, `mk_multi(wollaston,'Wollaston Beach',…)` | grain ✓ |
| 5 | `no_rain_fail_counts` | filter all-3-rain==0, `Enterococcus>=104`, groupby beach `.size()` → `Wollaston Beach 92` | argmax ✓ |
| 6→7 | Final Answer | `Wollaston Beach` | ✓ |

**No semantic divergence.** Answer = gold.

## Walk: Latest5kSchemaOnly (LOSER — FAIL, 5 steps, $0.060)

| step | op | semantics | plan-item |
|---|---|---|---|
| 0 | 3 raw probes incl. `raw_harbor` | previews `boston-harbor-beaches.txt` → renders canonical `0 Constitution Beach … 8 Wollaston Beach` | naming source |
| 1 | 8 `read_csv(header=2)` loads | all 8 datasheets loaded | load ✓ |
| 2/3 | `no_rain_fail_counts` | ANY-column `>=104` under all-3-rain==0, per sheet; labels beaches with **abbreviated names** `{'beach':'Wollaston'}`, `{'beach':'M Street'}` … (filename-derived) | grain ✓ (Wollaston is still argmax) |
| 3→4 | Final Answer | `Wollaston` | **DIVERGES: dropped the "Beach" suffix** |

**First semantic divergence:** the final label. The Latest arm **correctly identified Wollaston** — its computation is right — but emitted the abbreviated `Wollaston`, which fails the string/paraphrase grade against gold `Wollaston Beach`.

## Evidence at the decision — the canonical name was rendered, then EVICTED

Both arms rendered the identical canonical-name observation at step 2 (from `raw_harbor` / `raw_boston_harbor_beaches`):

```
  Output Table: 29 rows, 1 cols
  	line
  0	Constitution Beach
  ...
  8	Wollaston Beach
```

The knob then splits them. Per-step count of `"Wollaston Beach"` in each arm's rendered context (`inputMessages`):

| json step | LATEST (loser) | DELTA (winner) |
|---|---|---|
| 2 (probe obs first rendered) | **4** (raw_harbor visible) | 3 |
| 3 (**Latest authors short labels**) | **0** (raw_harbor EVICTED) | 3 |
| final answer step | **0** | **6** |

- **Latest:** `raw_harbor` (with `Wollaston Beach`) was on screen at step 2, but Latest compaction **dropped it by step 3** — the step where it authored `no_rain_fail_counts` with abbreviated labels — and it stayed gone at the answer step (`"Wollaston Beach"`×0, `"Constitution Beach"`×0). With zero canonical text in view, it fell back to a filename-derived `Wollaston`. Its own op summary ("count samples failing…return counts by beach") never re-surfaced the full name.
- **Delta:** delta history **retained** `raw_boston_harbor_beaches` and the canonical `Wollaston Beach` through every step (×3→×6), so the gold-form name was in front of the agent both when it authored its `'Wollaston Beach'` labels (step 4) and when it wrote the final answer (step 7).

## Verdict

**C3p (Delta5kSchemaOnly > Latest5kSchemaOnly): ATTRIBUTED — to `context_mode` (history retention of the canonical label).** Both arms computed the correct beach (Wollaston); the flip is purely whether the gold-form name `Wollaston Beach` survived to the answer step. The single knob controls exactly that: Latest's compaction **evicted** the `raw_harbor` observation (canonical names ×0 by the label-authoring step and answer step), starving it of the "Beach" suffix so it echoed the filename-stub `Wollaston`; Delta **retained** the observation (`Wollaston Beach` ×6 at answer) and emitted the full name. Winner's evidence explains its action (canonical name rendered); loser's absence explains its error (canonical name evicted); the divergence coincides with — does not predate — the first rendered difference. Non-chronic. **Caveat:** this is a naming/format retention effect, not a dataflow-reasoning effect — the semantic pipeline was equivalent in both arms.

Artifacts: `system_scratch/{DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt,DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt}/environment-hard-12/`; gold `solutions/environment/environment-hard-12.py` (Wollaston Beach).
