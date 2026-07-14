# Semantic walk: astronomy-easy-4

## Task + gold answer

**Question:** Determine the approximate period of solar activity cycles and identify the top five years of minimum and maximum activity between 1960 and 2020 using historical yearly mean sunspot numbers. Peaks need prominence 20 and distance 5; same for minima.

**Gold answer:** "The average period is 11 years, with maxima in 1968, 1979, 1989, 2000, and 2014, and minima in 1964, 1976, 1986, 1996, and 2008."

**Judge marks:** Delta5kSchemaOnly `llm_paraphrase=1.0` (PASS), Delta3kSchemaOnly `llm_paraphrase=0.0` (FAIL). Task is in `chronic_flippers.json` (flips between identical configs).

## Gold semantic plan

Source: `solutions/astronomy/astronomy-easy-4.py`

| # | Plan item |
|---|---|
| G1 | Load `data/astronomy/input/SILSO/SN_y_tot_V2.0.csv` with `sep=';'`, `header=None`, 5 named cols (Year, MeanSunspotNumber, StdDev, Obs, Definitive) |
| G2 | Year grain: `floor(Year)` → integer year (1960.5 → 1960) |
| G3 | Filter `1960 <= YearInt <= 2020` → 61 yearly rows |
| G4 | Maxima: `find_peaks(ssn, prominence=20, distance=5)` → 1968, 1979, 1989, 2000, 2014 |
| G5 | Minima: `find_peaks(-ssn, prominence=20, distance=5)` → 1964, 1976, 1986, 1996, 2008 |
| G6 | **Average period = `mean(diff(minima_years))` — min-to-min only** → (12+10+10+12)/4 = **11.0** |
| G7 | Report period + maxima years + minima years (all 5 found each, so "top five" = the full lists) |

Period-formula arithmetic on the (identical) extrema lists: min-to-min = 11.0 (gold), max-to-max = 11.5, combined max+min diffs = 11.25.

## Walk: DataflowSystemGPT52Delta5kSchemaOnly (WINNER — PASS)

**Final answer:** "Average period ≈ 11.25 years; maxima years: 1968, 1979, 1989, 2000, 2014; minima years: 1964, 1976, 1986, 1996, 2008" — judged PASS (11.25 within judge tolerance of "11").

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `sunspots_yearly_raw` | `pd.read_csv(...)` default sep → 324x1 single semicolon-packed column | G1 alternate-path (wrong sep, recovered next step) |
| 1a | `sunspots_yearly_parsed` | split col 0 on `;`, strip, name 5 cols, to_numeric | G1 ✓ (recovery) |
| 1b | `sunspots_1960_2020` | `year_fraction.between(1960, 2020.9999)`, `astype(int)` (=floor) → 61x5 | G2 ✓ G3 ✓ |
| 1c | `sunspots_peaks_troughs` | `find_peaks(x, prominence=20, distance=5)` + `find_peaks(-x, ...)`; top-5 by value re-sorted by year; **period = `mean(concat(diff(max_years), diff(min_years)))`** | G4 ✓ G5 ✓ — **DIVERGES at G6**: combined max+min diffs (11.25) instead of min-to-min (11.0) |
| 2 | (text) | Final answer 11.25 + both gold-identical lists | G7 ✓ (period off by +0.25, tolerated) |

**First semantic divergence:** step 1, third tool call (`sunspots_peaks_troughs`), plan item G6 — period formula averages max-to-max AND min-to-min gaps instead of gold's min-to-min only. Note this choice was committed at step 1, when the ONLY rendered observation was the step-0 raw load; total context at that point is far below even a 3k budget, so it renders identically under either arm's setting.

**Rendered evidence before step 1 (its formula-decision step):**
> `[sunspots_yearly_raw] Output 324x1: result: | Output Table: 324 rows, 1 cols | 1700.5;   8.3; -1.0;    -1;1 | 0	1701.5;  18.3; ...`

Nothing in this render (or the question text) speaks to which extrema-pair grain to difference.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly (loser — FAIL)

**Final answer:** "Average period ≈ 11.5 years; maxima years: 1968, 1979, 1989, 2000, 2014; minima years: 1964, 1976, 1986, 1996, 2008" — judged FAIL (11.5 outside tolerance; both extrema lists gold-identical).

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | `sunspots_yearly` | read_csv default sep (try/except sep=';' never fires — default parse "succeeds" as 324x1) | G1 alternate-path |
| 1 | `sunspots_yearly_clean` | split on `;`; **`year_mid.round(0)`** — banker's rounding (1701.5→1702, 1702.5→1702) | **DIVERGES at G2** (round, not floor) — later self-corrected |
| 2 | `sunspots_yearly_agg` | groupby(year).mean → **162 rows** (adjacent years collapsed pairwise) | corrupted grain, symptom rendered |
| 3 | `sunspots_1960_2020` | filter 1960–2020 → **31 rows** (should be 61) | symptom rendered; agent detects |
| 4 | `sunspots_yearly_reload` | reload with `sep=';', header=None, comment='#'` → 325x5 | G1 ✓ (recovery begins) |
| 5 | `sunspots_yearly_mean` | rename int keys `{0:..., 1:...}` → KeyError rendered | error, recovered next step |
| 6 | `sunspots_yearly_mean` (edit) | rename str keys; **`astype(float).astype(int)`** (=floor) | G2 ✓ (recovered) |
| 7 | `sunspots_1960_2020_fix` | filter 1960–2020 → **61x2** | G3 ✓ (recovered; data now equivalent to winner's) |
| 8 | `sunspots_peaks_troughs_1960_2020` | `find_peaks` both directions, prominence=20 distance=5; top-5 each; **`period = mean(diff(peak_years))`** | G4 ✓ G5 ✓ — **DIVERGES at G6**: max-to-max only (11.5) instead of min-to-min (11.0) |
| 9 | (text) | Final answer 11.5 + both gold-identical lists | G7 period wrong |

**First semantic divergence:** step 1 (`round` instead of `floor` on the year), plan item G2 — but this was self-detected from the rendered 31-row count and FULLY RECOVERED by steps 4–7 (61-row table identical in substance to the winner's). It cost ~7 extra steps / +69k input tokens / +$0.02, not the answer. **First unrecovered (load-bearing) divergence:** step 8, plan item G6 — period computed from max-to-max gaps only → 11.5.

**Rendered evidence before step 8 (its formula-decision step):**
> `[sunspots_1960_2020_fix] Output 61x2: Inputs: sunspots_yearly_mean (325 rows, 2 cols) | result: | ... | Output Table: 61 rows, 2 cols | year	sunspot_mean | 0	1960	159`

The correct filtered table was freshly rendered — the loser was NOT evidence-starved at its divergence step. **Did the winner have evidence the loser lacked? No — the opposite.** The winner committed its (passing) formula at step 1 with strictly LESS rendered evidence than the loser had at step 8, and no render in either arm, at any step, contains information bearing on min-to-min vs max-to-max vs combined. The question text (identical in both prompts, never compacted) is silent on it; gold's min-to-min convention is unhinted anywhere in the data.

## Pair verdicts

**C1 Delta5k > Delta3k: REJECTED-method-choice** (confirming the levers report; consistent with the task's chronic tag). Both arms produced gold-identical maxima/minima lists and diverged from gold only at plan item G6, the period formula — an unforced convention choice (winner: combined diffs → 11.25, tolerated; loser: max-to-max → 11.5, rejected; gold: min-to-min → 11.0). The winner's choice was committed at step 1 under a context far below either budget (identical render either way), i.e., the divergence predates the arms' first possible rendered difference, and the loser had richer evidence at its own decision step — so the 5k-vs-3k render lever cannot explain the flip; the formula coin-flip is exactly what makes this task a chronic flipper.
