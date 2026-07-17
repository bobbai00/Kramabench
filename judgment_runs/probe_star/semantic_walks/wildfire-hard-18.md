# Semantic walk: wildfire-hard-18 (PROBE-STAR, C1p) — chronic*

## Task + gold answer

**Question:** "Based on the NOAA dataset, controlling for the weather, do more aggressive suppression actually contribute to fire ending faster and affecting less buildings?"

**Gold answer:** "More aggressive suppression does not help fires end faster but helps fires affect less buildings." (string_approximate)

**Judge marks:** Delta5kSchemaOnlyProbePrompt `llm_paraphrase=1` (answer: "…fewer buildings threatened but *longer* durations — so it doesn't make fires end faster…", PASS); Delta1kSchemaOnlyProbePrompt `llm_paraphrase=0` (answer: "…fires ending faster … associated with *more* buildings threatened…", FAIL — both clauses inverted vs the key). B-only win in `venn_C1p.txt`, task **IS** in `chronic_flippers.json` (chronic*) → a single flip defaults to variance unless strongly attributed.

**Config diff (validity gate):** exactly one behavioral knob — `max_operator_result_char_limit: 1000` vs `5000`. Both arms fresh-run 2026-07-17 in star exp `093430` (oracle), symmetric recovery; walked 1k trace = its rec2 attempt (10:58, 3rd consecutive fail of the star), walked 5k trace = main-run pass (11:09; absent from its recovery rounds). Chronic texture confirmed by the same-day superseded star (exp `014004`, identical config): there the **5k arm failed this task too** (present in both its rerun-failed rounds), and the 1k arm failed at every observed checkpoint. Sibling arms, final states: Latest5k (5k render) PASS with the same inverted-signs phrasing ("Controlling for weather (and fire size), more aggressive suppression is associated with lo[nger durations]…"); DeltaStats1kD2 (1k render) FAIL with the loser's phrasing ("…faster-ending fires, but it does not…").

## Gold semantic plan — and a gold answer↔gold code contradiction

Source: `solutions/wildfire/wildfire-hard-18.py`

| # | Plan item |
|---|---|
| G1 | Load `noaa_wildfires.csv` (6658×37) |
| G2 | Treatment: `dominant_strategy_{25,50,75}_indicator = np.where(dominant_strategy_{25,50,75}_s == "Full Suppression", 1, 0)` — derived from the strategy STRINGS, one threshold per model |
| G3 | `dropna()` full-frame (n=3521) |
| G4 | OLS `duration ~ indicator + avrh_mean + wind_med + erc_med + rain_sum + hec` (levels, HC3), ×3 thresholds |
| G5 | OLS `prim_threatened_aggregate ~ same X` (prim only), ×3 thresholds |
| G6 | Read the signs → stated answer: "not faster + fewer buildings" |

**Data facts (verified, `dataflow-agent/.venv` python, statsmodels 0.14.6).** Executing the gold script's own regressions on the shipped data:

```
GOLD ind_25: duration −20.887 (p=5.7e-42) | prim_threat +542.653 (p=0.0061)
GOLD ind_50: duration −20.173 (p=9.9e-43) | prim_threat +543.543 (p=0.0054)
GOLD ind_75: duration −19.300 (p=1e-42)   | prim_threat +475.463 (p=0.014)
```

i.e. gold's own code says Full-Suppression fires end **faster** and threaten **more** primary structures — **the opposite of the published gold answer on both clauses.** The published answer's signs are only reachable via the dataset's PRE-BUILT indicator columns, which are **anti-coded against both the strategy strings and their own data-dictionary description** ("Indicator for wildfire with at least X% suppression strategies (1 = …)"):

```
crosstab dom_strat_ind_75 × dominant_strategy_75_s:
                 Full Suppression   Other
ind_75 = 0                  5633     121
ind_75 = 1                     0     904      (ind_25, ind_50 identical pattern: 838/872 ones, all 'Other')
```

`ind_*=1` never coincides with 'Full Suppression' (84.6% of fires are Full-Suppression at all three thresholds). Counterfactual matrix — the treatment column is the ONLY load-bearing choice; every other spec difference is sign-irrelevant:

```
LOSER exact spec (strings, no size ctrl):  duration −18.302 (p=3e-164) | buildings +964.3 (p=2e-6)
LOSER + hec:                               duration −18.954            | buildings +555.1   (signs unchanged)
LOSER spec, treatment=ind_75:              duration +20.693            | buildings −1032.7  (flips to key)
WINNER exact spec (ind_75, log, FE, hec):  log_dur +1.1199 (p=1e-177)  | log_bld −1.2850 (p=1e-42)
WINNER minus log_hec / minus FE:           +1.08…+1.03                 | −1.43…−1.73        (signs unchanged)
WINNER spec, treatment=(75_s==Full Supp):  log_dur −1.0347 (p=6e-159)  | log_bld +1.1284 (p=9e-37) (flips to data-true)
```

So the hec-omission hypothesis (starved render → no size control) is a red herring for the flip: adding `hec` to the loser changes nothing; removing it from the winner changes nothing. Pass/fail on this task is decided by ONE bit: operationalize "aggressive" from the strings (gold-code-faithful, data-true signs, FAILS the key) or from the anti-coded `dom_strat_ind_*` columns (matches the miskeyed answer, PASSES).

## Walk: DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt (WINNER — PASS)

**Final answer:** "longer durations (+206.5%), fewer buildings (−72.3%) → doesn't end faster, reduces building impacts" — matches the key. 8 agent steps, 6 ops, 87,414 in / 65,792 cached, $0.0916, 68.3s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_desc`, `raw_wf_probe` | read var-dictionary (UTF-8 error + cp1252 loader hint); `head(50)` of the fire table | G1 recon |
| 1 | `raw_desc` (edit) | reload cp1252 → 37×3 dictionary | aux ✓ |
| 2 | `wildfires` | full load 6658×37 | G1 ✓ |
| 3 | `wf_model_data`+`wf_regression`+`wf_sentence` | buildings = prim+comm+outb sum; log1p outcomes; OLS `log_duration/log_buildings ~ dom_strat_ind_75 + weather + log_hec + C(region) + C(start_year)` HC3; sentence formatter | **DIVERGES from gold plan at G2: treatment = pre-built `dom_strat_ind_75`** (anti-coded; gold derives from strings); also log outcomes, sum-of-3 buildings, region/year FE (all sign-irrelevant). NameError np |
| 4–6 | re-edits ×3 | add missing `import numpy` in each op successively | error-driven fixes, not semantic |
| 7 | (text) | Final Answer: not faster + fewer buildings | PASS (vs key) |

**First semantic divergence (vs the gold PLAN): step 3 — treatment column.** It regresses on `dom_strat_ind_75` and NARRATES it as "more aggressive suppression (>=75% suppression strategies)". On the shipped data that column flags 'Other'-strategy fires, so its stated finding inverts the data-true relation — and thereby lands on the (equally inverted) answer key.

**Rendered evidence at the model-spec decision (step-3 input, 10,426 chars):** the `raw_desc` dictionary block rendered **27 of 37 rows (2,899 chars)** — on screen were the ind_* definitions and the fire-size variable, plus the Young-et-al Table-1 column that essentially names gold's X vector:

```
29  dom_strat_ind_75  NaN  Indicator for wildfire with at least 75% suppression strategies (1 = wildfire with at least 75% supp...
33  hec  Fire Size  Wildfire incident size in hectares
3   avrh_mean  Average Relative Humidity …   4 wind_med  Wind …   5 erc_med  Energy release component …   6 rain_sum  Precipitation …
```

Its treatment choice (ind_75), size control (log_hec) and weather set quote this dictionary. **But the refutation of ind_75's description was ALSO on its screen:** its `raw_wf_probe`/`wildfires` blocks rendered row 0 (`… Full Suppression Full Suppression Full Suppression … 1 0 0 0 …` = gt_100=1, ind_75/25/50=**0,0,0**) and tail rows 6656–6657 (`… Other Other Other Monitor … 1 1 1 1 …` = ind columns **1,1,1** on 'Other' fires) with full 37-column values — four sample rows all contradicting "1 = suppression". It trusted the description over the visible values (evidence-disregard, benign w.r.t. the key).

## Walk: DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt (LOSER — FAIL)

**Final answer:** "ends faster, more buildings threatened" — data-true signs, inverted vs the key. 6 agent steps, 6 ops, 53,063 in / 46,080 cached, $0.0458, 45.6s.

| step | op | semantics (from code) | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `raw_var_desc`, `raw_wildfires_head` | dictionary read (same UTF-8 error + hint); `nrows=200` head | G1 recon |
| 1 | `raw_var_desc` (edit) | cp1252 reload → 37×3 | aux ✓ |
| 2 | `wildfires`, `var_desc` (+2 re-executes) | full load 6658×37; dictionary as named op | G1 ✓ |
| 3 | `analysis_prep` | `aggressive = (25_s=='Full Suppression') & (50_s==…) & (75_s==…)`; buildings = prim+comm+outb (fillna 0); subset dropna → 6657×10 | **G2 ✓ in kind — string-derived treatment, gold's own construction** (AND-of-3 variant; sign-identical to gold's single thresholds). Buildings=sum-of-3 (gold: prim; sign-irrelevant). **Omits size control** — G4's `hec` (sign-irrelevant, see matrix) |
| 4 | `regression_effects` | OLS levels: `duration/buildings ~ aggressive + weather` → −18.30 (p=3.5e-164) / +964.26 (p=1.7e-6) | G4/G5 minus hec; signs = gold's own code output |
| 5 | (text) | Final Answer: faster + more buildings | FAIL (vs key) — while agreeing with gold's script |

**First semantic divergence (vs the gold PLAN): step 3 — the missing `hec` control** (its only real departure; verified sign-irrelevant). On the load-bearing item — treatment definition — the loser is MORE gold-plan-faithful than the winner.

**Rendered evidence at the model-spec decision (step-3 input, 8,822 chars):** the dictionary was read twice and both renders collapsed under the 1k cap to **rows 0–1 + 36 only (540/538 chars)**:

```
0  start_year  Discovery Year  Year wildfire incident is discovered…
1  region_ind  NaN  Numeric value representing the region
...
36 total_fire_west  Number of Fires - West  Total number of wildifire incidents in the western U.S.…
```

Rows 2–35 — every substantive definition: the ind_* descriptions, `hec = Fire Size`, the Young-et-al Table-1 mapping — never rendered in this arm's entire trajectory. `dom_strat_ind_75/25/50` existed for it only as bare numeric names in the schema line, while `dominant_strategy_*_s` values ('Full Suppression'/'Other') were self-describing in the visible sample rows (row 0 + row 6657/199) — so it operationalized "aggressive" from the strings it could see. The same two sample rows also showed it the ind↔string inversion (FS,FS,FS with 0,0,0; Other,Other,Other with 1,1,1), equally un-acted-on.

## Pair verdict

**C1p Delta5k > Delta1k: CHRONIC / TASK-INTRINSIC — accuracy attribution REJECTED; render mechanism real but key-aligned, not truth-aligned.** The flip's single load-bearing bit is the treatment column, and the accept rule fails on the loser's side: its "error" is not a semantic shortfall against the gold plan — it implements gold's own treatment construction (strings == 'Full Suppression') and reports the signs gold's own script produces (−19..−21 duration / +475..+543 prim, replicated); it is graded wrong against an answer key that contradicts the gold code, reachable only through the anti-coded `dom_strat_ind_*` columns. The winner PASSES via an off-plan choice (adopting ind_75 per its dictionary description) whose stated interpretation is false on the data — with the falsifying sample rows rendered on its own screen. That said, the render lever's fingerprint on WHICH side each arm lands is genuine and doubly corroborated: at 5k the dictionary body (27/37 rows incl. the ind_* descriptions) was on screen at the model-spec step and both 5k-render arms (Delta5k, Latest5k) adopted the described indicators and passed; at 1k the dictionary rendered as 3 rows twice, and both 1k-render arms (Delta1k, DeltaStats1kD2) derived treatment from the visible strings and failed. So the 1k divergence IS render-starvation-shaped in the literal sense the caller asked (the 5k arm had the deciding artifact — the dictionary — on screen, the 1k arm never did), but what the richer render delivered was a MIS-description that happens to agree with a mis-keyed gold answer. Chronic flag stands (5k itself failed this task twice in the same-day superseded star). **Action item beyond this pair: file as a KramaBench task bug — `wildfire-hard-18`'s answer contradicts its own solution script on the shipped data (`dom_strat_ind_*` anti-coded vs both the strings and the dictionary); passes on this task measure key-alignment, not analysis correctness.**

Artifacts: `system_scratch/{DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt,DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt}/wildfire-hard-18/`; regressions replicated with `~/Desktop/bobflow/dataflow-agent/.venv/bin/python` (statsmodels 0.14.6) against `data/wildfire/input/noaa_wildfires.csv`.
