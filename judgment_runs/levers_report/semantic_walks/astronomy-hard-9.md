# Semantic walk — astronomy-hard-9

Pairs judged: **C1** Delta3k > Delta5k (chronic\*), **C3** Delta3k > Latest3k (chronic\*).
Task is in `chronic_flippers.json` (verified). Sweep-era traces: empty thoughts; all labels below are from code + rendered observations.

## Task + gold answer

Best lag (0–48 h) between semi-major-axis change (km) from TLE data of SATCAT 43180 and the OMNI2 hourly AP index, maximizing r² over May 1–30, 2024; TLE epochs rounded to nearest hour; mu = 398600.4418 km³/s². **Gold answer: 24** (numeric_exact).

Arms:

| arm | answer | steps | cost_usd | input_tokens | outcome |
|---|---|---|---|---|---|
| Delta3kSchemaOnly | 24 | 9 | 0.114 | 107k | PASS |
| Delta5kSchemaOnly | 19 | 7 | 0.118 | 103k | FAIL (wrong answer) |
| Latest3kSchemaOnly | (no response) | 25 | 0.451 | 324k | FAIL (step-exhausted, never answered) |

## Gold semantic plan

From `solutions/astronomy/astronomy-hard-9.py`:

1. **OMNI2 load spec**: `read_fwf(omni2_2024.dat)` with **explicit 55-column widths** per the `omni2.text` spec; time = Jan-1(year) + (doy−1) days + hour (UTC); keep hourly `ap` = **column index 49** (0-based); slice a wide window (Apr 1–Jun 30) so lag shifts don't starve the join.
2. **TLE load spec**: strict 2-line pairs from `TLE/43180.tle` (asserts even line count); per pair a = (mu/n_rad_s²)^(1/3), n = no_kozai rad/min ÷ 60; epoch per TLE field.
3. **Filter**: epochs in [2024-05-01, 2024-06-01), sorted by epoch.
4. **Diff grain**: semi-major-axis change = diff of successive TLEs, assigned to the second epoch; drop leading NaN.
5. **Duplicate-epoch handling**: round epoch to nearest hour (`round("h")`), then **groupby rounded hour → mean** of the change (multiple TLEs in one hour are averaged, not dropped).
6. **Final compute**: for lag 0..47, shift AP forward by lag, inner-join on the hour, Pearson r → r²; argmax r² → **24**.

### The decisive data hazard (established during this walk, reproduced locally)

`pd.read_fwf(omni2_2024.dat, header=None)` **without explicit widths** infers colspecs from the first 100 rows (January, DOY 1–2 digits). Inferred field 1 = chars 7–8 (one digit), so **DOY is silently truncated to its last digit for every row with doy ≥ 10**: raw line 2900 `2024 121 20 …` parses as `[2024, 1, 20]`. Times still parse (NaT share 0.0) but **zero rows can ever fall in the May window** (verified: May rows = 0 after an Arrow round-trip in the UDF venv). Gold is immune (explicit widths); `read_csv(delim_whitespace=True)` is immune (no width inference). Both losers loaded with bare `read_fwf` at step 0; the winner used whitespace-CSV. Every mysterious `Output Table: 0 rows, 0 cols` in both losing traces is this one defect.

Also verified: the omni2 spec render was truncated to ~100 chars of the content cell in BOTH Delta arms (`"O M N I 2  DATA SET\n … Th..."`); the AP column position was **never rendered in any arm**. Delta3k's correct `ap = col 49` was prior knowledge, not rendered evidence.

## Walk: DataflowSystemGPT52Delta3kSchemaOnly — PASS, answer 24

| step | op | semantics | plan-item |
|---|---|---|---|
| 0 | tle_43180 | manual 2-line TLE pair parse (epoch l1[18:32], mm l2[52:63] rev/day) → 134×8 | 2 ✓ |
| 0 | omni2_spec | read `omni2.txt` → FileNotFoundError (file is `.text`) | — (probe) |
| 0 | omni2_2024_raw | **`read_csv(header=None, delim_whitespace=True)`** → 8784×55, clean numerics | 1 ✓ (hazard-immune load) |
| 1 | omni2_spec | fallback `.txt`→`.text` → loads; render truncated ~100 chars | — |
| 2 | omni2_2024_ap | time = year/doy/hour (cols 0/1/2), **ap = col 49**, May 1–31 slice | 1 ✓ (KeyError: string col names) |
| 3 | omni2_2024_ap | same semantics via `cols[i]` → **720×2** May AP | 1 ✓ |
| 4 | tle_sma_change_hourly | a=(mu/n²)^⅓; round epoch to hour; diff; May filter; **groupby hour → mean** | 3,4,5 ✓ (TypeError: tz vs str) |
| 5 | tle_sma_change_hourly | same, Timestamp compare | (TypeError: tz vs naive) |
| 6 | tle_sma_change_hourly | same + tz normalize → **126×2** (134 TLEs → 126 hours: duplicate hours averaged) | 3,4,5 ✓ |
| 4/7 | lag_r2_scan | lag 0..48, AP shifted forward, inner join, corr² → **best 24, r²=0.663, n=121** | 6 ✓ |
| 8 | (text) | Final Answer: 24 | ✓ |

First semantic divergence: **none** (lag range 0..48 vs gold's 0..47 and May-only AP slice vs gold's Apr–Jun are harmless variants; same argmax). Four errors (spec path, KeyError, 2× tz TypeError) were all mechanical and each fixed in ≤2 edits with no semantic wobble. Duplicate-epoch handling matches gold exactly (groupby-mean per rounded hour).

## Walk: DataflowSystemGPT52Delta5kSchemaOnly — FAIL, answer 19 (gold 24)

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | omni2_spec | read `omni2.txt` → FileNotFoundError | — (probe) |
| 0 | omni2_2024_raw | **`read_fwf(header=None, dtype=str)`** → 8784×55 | **DIVERGES plan-1 (load spec): DOY truncated to last digit for doy≥10; May window unreachable. Latent — render looks normal.** |
| 0 | tle_43180_raw | raw lines → 268×1 | 2 ✓ |
| 1 | omni2_spec | fallback `.text` → loads (truncated render) | — |
| 2 | omni2_2024_ap | year/doy/hour ✓ but **ap = col 50 (f10.7)** | **DIVERGES plan-1 (ap col)** — KeyError, never executed |
| 3 | omni2_2024_ap | same via `cols[i]` (ap still 50) → **`Output Table: 0 rows, 0 cols`** — no error, no sample, no diagnostic | executed but empty (fwf DOY hazard); wrong-ap masked |
| 4 | omni2_2024_ap | **HINGE: reinterprets time as year/month/day/hour = cols 0/1/2/3** (doy→month, hour→day, brn→hour); ap corrected to 49 → **814×2 "May" rows** | **DIVERGES plan-1 (time basis)** — rows are Jan/Feb/… lines whose truncated DOY ends in 5; exact 814 reproduced locally |
| 5 | ap_hourly / tle_elements / tle_sma / drag_series / lag_scan / best_lag | AP groupby-hour → **330×2** (a May hourly series should have ≤720 uniques — red flag rendered); TLE side fully correct: 134 TLEs → a=6877.16 km → diff → groupby-mean → 125×2 (plan 2–5 ✓); lag scan vs garbage AP → **best 19, r²=0.296, n=49** (tiny join, rendered) | 2–5 ✓ / **6 poisoned by AP series** |
| 6 | (text) | Final Answer: 19 | ✗ |

First semantic divergence: **step 0 (`omni2_2024_raw`, bare `read_fwf`)** — the point where the data went bad; first code-level divergence step 2 (ap=50, masked by the empty output); the divergence that fixed the wrong answer: **step 4's time-basis flip**.

Evidence at the hinge (rendered before step 4, in full):
> `[omni2_2024_ap] Output 0x0: Inputs: omni2_2024_raw (8784 rows, 55 cols) | result: … Output Table: 0 rows, 0 cols | Schema (2 cols): time (datetime), ap (numeric)`

Nothing in the render distinguishes "your DOY/hour reading is right but the fwf table is corrupt" from "your column mapping is wrong" — the agent guessed the latter. The downstream tells (814 > 720 rows, 330 unique hours, n=49 join) were rendered and ignored.

## Walk: DataflowSystemGPT52Latest3kSchemaOnly — FAIL, no answer (step-exhausted)

| step | op | semantics | plan-item / DIVERGES |
|---|---|---|---|
| 0 | tle_43180_raw | **3-line-record parse (name/l1/l2, stride 3) over a 2-line file → 44/134 TLEs** (every 3rd pair survives the startswith checks) | **DIVERGES plan-2 (TLE load spec)**; render shows `name` = a line-2 string ("2 43180 97.3997…") — visible, never noticed |
| 0 | omni2_2024_raw | **`read_fwf(header=None)`** → 8784×55 | **DIVERGES plan-1 — same DOY-truncation trap as Delta5k** |
| 0–5 | omni2_spec | `.txt` ×3, path-replace, glob `omni2*.txt` — **all fail; `.text` never tried although the task prompt names "omni2.text" verbatim**; final render `exists=false` | spec never loaded |
| 3–14 | omni2_ap_may / _hourly / debug / probe / timecheck | AP column guesses **26, 24, 24, 3, 24, 7, 8** with year/doy/hour time build; every May-filtered execution → `0 rows, 0 cols`; step 9 `timecheck` head(48) with **no** filter → 48×4 correct January times (the decisive contrast: table parses, May doesn't exist — rendered, never acted on) | 1 ✗ repeatedly; diagnosis never made |
| 5,7 | tle_elements / tle_drag | a ✓ formula, floor(+30min) rounding, May filter, groupby-hour mean of a then diff → **43×2** drag points (vs 125 in Delta arms — the 44-TLE deficit, rendered) | 3,5 ~✓ on broken input; 4 variant (diff of hourly means) |
| 15–20 | omni2_ap_colscan | **column-scan probe (per-column May min/median/q95/max to find AP) resubmitted 6×**; consecutive-pair similarity 0.67 / 0.89 / 0.60 / 0.80 / **0.995** (>0.92 identical-probe flag); every run 0x0 / NameError / KeyError because the May mask itself is empty | churn |
| 21 | ap_hourly_fromscan | robustified chooser on empty scan → 0x0 | churn |
| 22 | may_window + ap_candidates + ap_hourly_fixed + best_lag | **the same scan rebuilt as a new 3-op pipeline** → all 0x0 | churn (rebuild) |
| 23–24 | ap_hourly_simple / ap_may_ok | ap guesses cols 39, 38; a `to_datetime('%Y-01-01')` variant | out of steps; **no answer** |

First semantic divergence: **step 0, twice independently** — the 3-line TLE stride (44/134 TLEs) and the bare-`read_fwf` OMNI2 load. Both predate any rendered difference between arms. 18 ops, 25 agent steps, $0.451 (4× the Delta arms).

Evidence during the churn (rendered, step-10 context — near-identical reflections appear in **22 of 25** step contexts):
> `Attempt reflection: you have edited operator `omni2_spec` 6 times; 5 prior variant(s) did not converge. | Latest result: produced output but the task is not yet solved. | Do NOT submit another near-variant. Change strategy: inspect this operator's UPSTREAM result/columns for a wrong assumption, try a DIFFERENT decomposition, or build the answer from a different operator.`

The engine explicitly rendered edit counts and the precise correct advice ("inspect UPSTREAM … wrong assumption"). The re-probes were therefore **not history-starved** — every probe's result (0x0) and the repetition warnings were in context each time. What was missing was the *diagnosis* (fwf DOY truncation), which no arm's render surfaces: under SchemaOnly there is no stats block, and a stats block (doy max = 9 over the "May" mask, or per-column min/max on the raw table) is the artifact that would have exposed it — that lever is equalized across both C3 arms.

## Pair verdicts

**C1 — Delta3k > Delta5k: CHRONIC-VARIANCE.**
The arms had identical rendered evidence at every aligned decision: same truncated spec cell (AP position rendered in neither), same plausible 8784×55 raw render, same loader hint. The winner's load choice (`read_csv` whitespace vs `read_fwf`) and correct `ap=49` came at steps 0–2 from priors, before any rendered difference existed — the reject condition of skill rule 5. The loser's hinge (step 4) was taken against an evidence-free `0 rows, 0 cols` render that the 5k budget could not enrich (more render budget than the winner, so not render starvation). Task is chronic; accept rules not met; default stands.

**C3 — Delta3k > Latest3k: CHRONIC-VARIANCE** (the levers-report "history-blindness churn" attribution is specifically **REJECTED-method-choice**).
The churn is real and confirmed (colscan resubmitted at steps 15–20 with a 0.995-similarity pair; the same probe rebuilt as new ops at step 22; 18 ops; $0.451). But the mechanism is not missing history: each probe's 0x0 result was rendered every time, and the engine additionally rendered explicit "edited N times / do NOT submit another near-variant / inspect UPSTREAM" reflections in 22/25 contexts — richer anti-repetition signal than DELTA's change-log adds. The Delta arm's rendered history contains nothing that would have prevented the re-probe, because the missing fact (why May is empty in a bare-fwf table) is never rendered under any history mode; the winner avoided the loop entirely through two step-0 method choices (whitespace-CSV load; 2-line TLE parse) made with zero rendered difference between arms — and the Latest arm's independent 44/134-TLE defect would plausibly have kept the answer wrong even had it escaped the AP trap. Divergence predates the arms' first rendered difference → attribution to the Latest-vs-Delta lever rejected; task chronic → CHRONIC-VARIANCE. (The lever did shape the failure *topology* — Latest burned 4× cost failing loudly where Delta5k failed cheaply and wrong — consistent with the known Latest-thrash tail, but shaping how a doomed run fails is not flip attribution.)

**Cross-cutting note for the deck:** the gold plan's duplicate-epoch item (groupby-mean per rounded hour) was NOT the discriminator on this task — all three arms handled or bypassed it consistently with gold on the TLE side (134→126, 134→125, 44→43). The discriminator was the OMNI2 load spec: bare `read_fwf` width-inference silently destroys the DOY column (first-100-row inference on a January-headed year file), an engine-invisible hazard that manifests only as evidence-free empty outputs downstream. A stats-bearing render (per-column min/max) is the cheapest lever that would have exposed it; SchemaOnly hid it in every arm here.
