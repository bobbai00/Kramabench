# Semantic walk: environment-hard-20 (PROBE-STAR vintage, raw-probe prompt)

## Task + gold answer

**Question:** In 2015, what are the three most polluted beaches of the city that had the least rainfall in the summer (June, July, August)?

**Gold answer:** `['Bucks Creek', 'Pleasant Street', 'Forest Street']` (Chatham beaches). Source `solutions/environment/environment-hard-20.py`.

**Pair (C3p, A-only):** winner `Delta5kSchemaOnlyProbePrompt` (gold exactly, PASS) vs loser `Latest5kSchemaOnlyProbePrompt` (`['Tenean (DCR)','Malibu (DCR)','Constitution (DCR) @ Middle']` = Boston beaches, FAIL). **NOT** in `chronic_flippers.json`.

**Config diff (validity gate):** one knob — `context_mode: delta` vs `latest`. Passes one-knob gate.

## Gold semantic plan

| # | Plan item |
|---|---|
| G1 | For each of the 4 fresh cities `[boston, chatham, amherst, ashburnham]`: load `monthly_precipitations_<city>.csv`, filter `Year==2015`, sum `Jun+Jul+Aug` |
| G2 | **argmin over ALL 4 cities → least-rain city = Chatham** (2015 summer rain: Boston 9.29, **Chatham 8.71**, Amherst 16.49, Ashburnham 11.9) |
| G3 | Load `water-body-testing-2015.csv`, filter `Community == 'Chatham'` |
| G4 | Per beach: exceedance rate = (`Violation=='yes`)/count; sort desc; top-3 → `['Bucks Creek','Pleasant Street','Forest Street']` |

Load-bearing step is **G2: take the min over all four fresh cities.** (Verified: all 4 cities ARE present in `water-body-testing-2015.csv`'s 220 communities.)

## Walk: Delta5kSchemaOnly (WINNER — PASS, 5 steps, $0.076)

| step | op | semantics | plan-item |
|---|---|---|---|
| 0–1 | 5 raw probes + 4 precip loads + water load | per-city summer_rainfall computed | G1 ✓ |
| 2 | `summer_rainfall_2015` | iterates the 4 named cities, `Year==2015`, sort ascending → `Chatham 8.71 rank 1` | **G2 ✓** |
| 2 | `most_polluted_beaches_least_rain_city` | filter water to `Community==Chatham`, rank beaches, top-3 → `Bucks Creek …` | G3+G4 ✓ |
| 3 | Final Answer | `['Bucks Creek','Pleasant Street','Forest Street']` | ✓ |

**No divergence.** Answer = gold.

## Walk: Latest5kSchemaOnly (LOSER — FAIL, 8 steps, $0.050)

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0–1 | globbed precip probe + concat load (116×15, `file` col) + water load | all 4 precip files loaded; knows the 4 cities | G1 ✓ |
| 2 | `summer_rain_2015_by_city` | `Year==2015`, then **`p = p[~p['file'].isin([ashburnham, amherst, chatham])]`** → only **Boston** survives | **DIVERGES at G2:** excludes 3 of 4 fresh cities |
| 3 | `least_rain_city` | `nsmallest(1)` over the 1 remaining row → Boston (trivially) | G2 broken |
| 4–5 | filter water to `Community==Boston`, rank → top-3 Boston beaches | G3/G4 on the WRONG city |
| 6 | Final Answer | `['Tenean (DCR)','Malibu (DCR)','Constitution (DCR) @ Middle']` | wrong city → wrong beaches |

**First semantic divergence:** step 2, op `summer_rain_2015_by_city`, plan item G2 — the hardcoded `~file.isin([ashburnham, amherst, chatham])` collapses the candidate set to Boston. Its op summary states the rationale: *"keep Year=2015 and only cities that appear in `beach2015` (Boston)"* — a false premise (Chatham/Amherst/Ashburnham all appear in the beach data).

## Evidence at the decision — identical (null) beach-community evidence

Neither arm ever probed the distinct `Community` values of `water-body-testing-2015.csv`. Per-step count of community tokens in each arm's rendered context at the divergence:

| json step | LATEST (loser) | DELTA (winner) |
|---|---|---|
| 2 (**Latest excludes 3 cities here**) | `Boston`×0, `Chatham`×0, `Provincetown`×7 | `Boston`×0, `Chatham`×0, `Provincetown`×7 |
| 3 | `Boston`×0, `Chatham`×0 | `Chatham`×1 (its own G2 output) |

At the exact step the Latest arm authored the fatal exclusion, its rendered evidence about beach communities was **only "Provincetown"** — `Boston`×0 and `Chatham`×0. Its stated justification ("only cities that appear in beach2015 (Boston)") is contradicted by its own rendered preview (which showed Provincetown, never Boston): a fabricated restriction, not an evidence-driven one. The Delta arm had the **identical** beach-community evidence (Provincetown-only, no distinct-community probe); it simply did not impose the spurious filter — it took the min over all four cities and looked Chatham up (which succeeds because Chatham is in the data). `Chatham` first enters Delta's context at step 3 as the **output** of its own correct G2 op, not as an input that Latest lacked.

## Verdict

**C3p (Delta5kSchemaOnly > Latest5kSchemaOnly): REJECTED — method-choice / reasoning hallucination, not attributable to `context_mode`.** The Latest arm invented an ungrounded "restrict fresh cities to those in the beach data = Boston only" pre-filter, excluding Chatham/Amherst/Ashburnham; the Delta arm used the correct min-over-all-cities method. The divergence lives in the code each arm authored, not in what was rendered: both had identical beach-community evidence (Provincetown-only preview, neither probed the distinct community list), and the Latest arm's exclusion even contradicts its own rendered observation. Delta's win is a correct-method pick, not delta-history evidence that Latest lacked. The divergence predates any latest-vs-delta rendered difference → reject per the method-choice rule. Non-chronic but unattributed.

Artifacts: `system_scratch/{DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt,DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt}/environment-hard-20/`; gold `solutions/environment/environment-hard-20.py` (Chatham → Bucks Creek/Pleasant Street/Forest Street).
