# astronomy-hard-9 — deep dive (counter-intuitive: 3k-solo win — least render budget beat 5k and Latest)

The arm with the LEAST rendered evidence won. **Delta3kSchemaOnly** (mode Y: delta history,
3k char limit, no stats block) answered **24 = gold**; **Delta5kSchemaOnly** (more render
budget, same history mode) answered 19, and **Latest3kSchemaOnly** (full-DAG re-render each
step) never answered — it died at the 25-step cap after 20 steps of AP-column churn at 4×
the cost. `astronomy-hard-9` is in `chronic_flippers.json` (verified), so the default is
CHRONIC/VARIANCE unless the accept rules are met. Verdict up front: **C1 CHRONIC, C3
CHRONIC — and the levers-report "history-blindness churn" story for C3 is specifically
REJECTED: the churn happened despite rendered refutations and explicit engine anti-churn
reflections in 22/25 contexts (re-verified).** Every data claim below was re-executed on
the real files with `.venv/bin/python`; sweep-era traces have empty thoughts, so all labels
come from code + rendered observations.

One knob per pair (config.json diff verified): C1 = `max_operator_result_char_limit`
3000→5000; C3 = `context_mode` delta→latest. All arms: `agent_mode:code`,
`max_operator_edits:0`, SchemaOnly (no column stats), max_steps 25.

| Arm | role | context_mode | char_limit | steps | cost_usd | input_tokens | answer | verdict |
|---|---|---|---|---|---|---|---|---|
| **Delta3kSchemaOnly** (mode Y) | **WINNER** | delta | 3k | 9 | 0.114 | 107k | **24** | PASS |
| Delta5kSchemaOnly (X, C1) | loser | delta | **5k** | 7 | 0.118 | 103k | 19 | FAIL (wrong answer) |
| Latest3kSchemaOnly (X, C3) | loser | **latest** | 3k | 25 | 0.451 | 324k | (no response) | FAIL (step-exhausted) |

## Task

Q: "Determine the best lag (from 0 to 48 hours) between atmospheric drag--measured as
semi-major axis change (in km) from TLE data of SATCAT 43180--and the OMNI AP index, that
maximizes the r^2 correlation during May 1--30, 2024. TLE epoch times should be rounded to
the nearest hour to align with AP measurements. Use hourly OMNI2 data. omni2 data format
specification can be found at omni2.text file Use earth's gravitational paremeter mu =
398600.4418 km^3/s^2." Gold answer: **24** (numeric_exact).

D: three files under `data/astronomy/input/`. Path quirk zero: the prompt's machine
manifest lists `omni2_low_res/omni2.txt`, but the file on disk is `omni2.text` (the
question string names "omni2.text" verbatim). All three arms' first spec load 404'd on the
manifest path.

`omni2_low_res/omni2_2024.dat` — **8784 lines** (366 d × 24 h, 2024 is a leap year) × 55
fixed-width fields, no header, space-padded right-aligned columns. Real lines (file lines
1–2 = Jan 1, lines 2901–2902 = May 1 = DOY 121):

```
2024   1  0 2596 51 52  60  36   5.3   5.1  -7.3 322.9   4.0  -3.0  -0.6  -2.8 ...
2024   1  1 2596 51 52  53  33   5.4   5.2 -21.3 333.4   4.4  -2.2  -1.9  -1.6 ...
2024 121 20 2601 51 52  61  26   7.0   6.5  -8.7 224.8  -4.6  -4.5  -1.0  -3.7 ...
2024 121 21 2601 51 52  63  35   9.4   8.4 -37.7 214.5  -5.5  -3.8  -5.1  -1.1 ...
^^^^ year, I4, chars 0-3
    ^^^^ DOY, I4, chars 4-7 (right-aligned: Jan 1 = "   1", May 1 = " 121")
        ^^^ hour, I3, chars 8-10
```

Semantics: word 1 = year, word 2 = decimal day-of-year (Jan 1 = 1), word 3 = hour 0–23;
word 50 = **ap-index** (nT) = **0-based column 49**; word 51 = f10.7 = 0-based col 50.
Fill values 999.9 / 9999. / 999999.99 pepper the physics columns.

**THE decisive hazard (reproduced locally):** DOY is right-aligned in a 4-char field, and
the first 100 rows are Jan 1–5 (DOY 1–5 — a single digit sitting at char 7).
`pd.read_fwf(..., header=None)` infers colspecs from those first 100 rows and infers field
1 = **chars 7:8, one character wide** (inferred colspecs `[(0,4), (7,8), (9,11), (12,16)]`).
Every row with DOY ≥ 10 keeps only the LAST digit: raw row 2900 `2024 121 20` parses as
`[2024, 1, 20]`. Verified: max parsed DOY = **9**, NaT share **0.0** (all times "parse"),
May-window rows = **0**, and the table still renders a plausible **8784×55** — the defect
is engine-invisible under SchemaOnly. `read_csv(header=None, delim_whitespace=True)` is
immune (no width inference): 8784×55 with 744 true May hours. Gold is immune via 55
explicit widths.

`omni2_low_res/omni2.text` — the human-readable OMNI2 spec (~300 lines of prose + a
WORD/FORMAT table). The rows that matter (verbatim, lines 94–98 and 167–168):

```
WORD  FORMAT  Fill Value         MEANING                  UNITS/COMMENTS
 1      I4              Year                              1963, 1964, etc.
 2      I4              Decimal Day                       January 1 = Day 1
 3      I3              Hour                              0, 1,...,23
...
50       I4   999      ap-index                     nT
51       F6.1 999.9    f10.7_index                  ( sfu = 10-22W.m-2.Hz-1)
```

Render quirk: loaded as a 1×1 `content` cell, both Delta arms rendered it truncated to
~100 chars (`"O M N I 2  DATA SET\n … Th..."`) — the ap-index position (word 50) was
**never rendered in any arm**, 3k or 5k.

`TLE/43180.tle` — **268 non-blank lines = 134 strict 2-line element sets**, NO name/"0 "
lines (grep: 134 lines start `"1 "`, 134 start `"2 "`, 0 start `"0 "`). Real lines (first
pair):

```
1 43180U 18014A   24122.17811289  .00004675  00000-0  20621-3 0  9994
2 43180  97.3997  30.6612 0001844  94.0751 266.0696 15.22259301346483
```

Semantics: line 1 chars 18–32 = epoch `YYDDD.dddddddd` (`24122.17811289` = 2024, DOY
122.178 = 2024-05-01 04:16:28Z — the whole file is May 2024); line 2 chars 52–63 = mean
motion `no_kozai` in rev/day (15.22259301). **The 3-line-stride hazard (simulated
locally):** parsing the common name/l1/l2 3-line format (`range(0, n-2, 3)` + startswith
checks) over this 2-line file keeps only offsets i ≡ 3 (mod 6) → **44 of 134 TLEs**, and
puts a line-2 string in the `name` field — both visible in the render.

## Solution

Dataflow from `solutions/astronomy/astronomy-hard-9.py`:

```
read_fwf(omni2_2024.dat, widths=[4,4,3,5,3,3,4,4,6,...,5])   # 55 EXPLICIT widths — hazard-immune
  → t = Jan-1(year) + (doy−1) days + hour (UTC)
  → slice 2024-04-01..2024-06-30, keep ap = named col from position 49 ─────────────┐
                                                                                     ├─ for lag 0..47:
read_tle_pairs(43180.tle)          # strict 2-line pairs, asserts even line count    │    shift AP index +lag h
  → per pair: n = no_kozai rad/min ÷ 60;  a = (mu/n²)^(1/3),  mu=398600.4418         │    inner-join on hour
  → epoch per TLE field (skyfield)                                                   │    r = Pearson corr → r²
  → filter [2024-05-01, 2024-06-01), sort by epoch                                   │  argmax r² → 24
  → semi_major_change = diff(a), assigned to the SECOND epoch, drop leading NaN      │
  → epoch.round("h")  →  groupby(rounded hour).mean()   # dup hours AVERAGED ────────┘
```

Plan items: (1) OMNI2 load spec = explicit widths + year/doy/hour time + ap col 49 + wide
window; (2) TLE load spec = strict 2-line pairs, a from no_kozai; (3) May filter, sorted;
(4) diff grain = successive TLEs → second epoch; (5) duplicate-epoch handling = round to
hour then groupby-mean; (6) lag scan 0..47, shift AP forward, inner join, argmax r².

## What Delta5kSchemaOnly does (mode X, C1 — 19, FAIL)

- step 0 `omni2_spec`: read manifest path `omni2.txt` → FileNotFoundError (probe).
- --> step 0 `omni2_2024_raw`: **`pd.read_fwf(..., header=None, dtype=str)`** → 8784×55.
  DIVERGES plan-1 (load spec): DOY silently truncated to its last digit for doy ≥ 10; the
  May window is unreachable from this table forever after. Latent — the render looks
  normal.
- step 0 `tle_43180_raw`: raw lines → 268×1 ✓ (plan-2 deferred downstream).
- step 1 `omni2_spec`: `.txt`→`.text` fallback → loads; rendered truncated at ~100 chars.
- --> step 2 `omni2_2024_ap`: time from cols 0/1/2 ✓ but **ap = col 50 = f10.7** (spec
  word 51, off-by-one vs word 50). DIVERGES plan-1 (ap col) — died on `KeyError: 0`
  (string column labels), so the wrong pick never executed.
- step 3 `omni2_2024_ap`: same semantics via `cols[i]` (ap still 50) → **`Output Table: 0
  rows, 0 cols`** — the fwf hazard fires; no error, no sample, no diagnostic; the wrong-ap
  pick stays masked.
- --> step 4 `omni2_2024_ap`: **THE HINGE.** Reinterprets the time basis as
  year/month/day/hour = cols 0/1/2/3 (doy→month, hour→day, brn→hour) and fixes ap to 49 →
  **814×2 "May" rows**. DIVERGES plan-1 (time basis): those are Jan/Feb/… rows whose
  truncated DOY happens to be 5; the exact 814 reproduces locally. 814 > 720 possible
  May-1..30 hours — rendered, ignored.
- step 5 (6 ops in one call): `omni2_2024_ap_hourly` groupby-hour → **330×2** (a May
  hourly series has ≤720 unique hours — second rendered red flag); TLE side fully correct:
  134 TLEs → a = 6877.16 km → round → May filter → diff → groupby-mean → **125×2**
  (plans 2–5 ✓); `lag_scan` vs the garbage AP → **best 19, r² = 0.296, n = 49** (tiny
  join, rendered); `best_lag` → 19.
- step 6: Final Answer: 19 ✗.

First semantic divergence: step 0 (bare `read_fwf`) — where the data went bad; the
divergence that fixed the wrong answer in place: step 4's evidence-free time-basis flip.

## What Latest3kSchemaOnly does (mode X, C3 — no answer, FAIL at step cap)

- --> step 0 `tle_43180_raw`: **3-line-record parse (name/l1/l2, stride 3) over the 2-line
  file → 44/134 TLEs**. DIVERGES plan-2 (TLE load spec). The render shows the defect
  plainly — `Output 44x6 ... name | 0  2 43180  97.3997  30.9846 ...` (the `name` field is
  a line-2 string) — visible, never noticed.
- --> step 0 `omni2_2024_raw`: **bare `read_fwf(header=None)`** → 8784×55. DIVERGES plan-1
  — the same DOY-truncation trap as Delta5k, independently.
- steps 0–5 `omni2_spec`: `.txt` ×3, a path-replace variant, glob `omni2*.txt` — all fail;
  **`.text` never tried** although the question names "omni2.text" verbatim; final render
  `exists=false`. The spec never loads; ap stays a guessing game.
- steps 3–14 (`omni2_ap_may` / `omni2_ap_hourly` / debug / probe / timecheck): AP column
  guesses **26, 24, 24, 3, 24, 7, 8** on a correct year/doy/hour time build; every
  May-filtered execution → `0 rows, 0 cols`. Step 9 `omni2_timecheck` `head(48)` with NO
  May filter → **48×4 correct January times** (`2024 1 0 2024-01-01T00:00`) — the decisive
  rendered contrast (table parses fine; May doesn't exist) that was never acted on.
- steps 5/7 `tle_elements`/`tle_drag`: correct a-formula, floor(+30 min) hour rounding, May
  filter, groupby-hour mean-of-a then diff → **43×2** drag points (vs 125/126 in the Delta
  arms — the 44-TLE deficit, rendered; plan-4 variant: diff of hourly means).
- --> steps 15–20 `omni2_ap_colscan`: a per-column May min/median/q95/max scan to find AP,
  **resubmitted 6×**; consecutive-pair difflib similarity 0.666 / 0.885 / 0.603 / 0.803 /
  **0.995** (re-verified; last pair over the 0.92 identical-probe flag). Every run 0x0 /
  NameError / KeyError — the May mask itself is empty, so the scan can never return rows.
- step 21 `omni2_ap_hourly_fromscan`: robustified chooser on the empty scan → 0x0.
- --> step 22: **the same scan rebuilt as a new 3-op pipeline** (`omni2_may_window` →
  `omni2_ap_candidates` → `omni2_ap_hourly_fixed`) + `best_lag` → all 0x0 (churn-rebuild).
- steps 23–24: ap guesses cols 39, then 38 with a `to_datetime('%Y-01-01')` time variant →
  out of steps. **No answer.** 18 ops, 25 agent steps, $0.451, 324k input tokens (~4×
  either Delta arm).

First semantic divergence: step 0, twice independently (3-line TLE stride; bare
`read_fwf`) — both before any rendered observation existed.

## What Delta3kSchemaOnly does (mode Y, WINNER — 24, PASS)

- step 0 `tle_43180`: manual **strict 2-line pair parse** (epoch `l1[18:32]`, mean motion
  `l2[52:63]` rev/day) → **134×8** ✓ plan-2.
- step 0 `omni2_spec`: manifest path `omni2.txt` → FileNotFoundError (same trap as both
  losers; near-miss, recovered at step 1).
- step 0 `omni2_2024_raw`: **`pd.read_csv(path, header=None, delim_whitespace=True)`** →
  8784×55 clean numerics ✓ plan-1 — the hazard-immune load, chosen before anything had
  rendered.
- step 1 `omni2_spec`: `.txt`→`.text` fallback → loads (render truncated ~100 chars —
  identical to Delta5k's).
- step 2 `omni2_2024_ap`: time from cols 0/1/2, **ap = col 49**, May 1–31 slice →
  `KeyError: 0` (near-miss 2, mechanical: integer vs string column labels; fixed in 1
  edit).
- step 3: same semantics via `cols[i]` → **720×2** May AP ✓ plan-1.
- step 4 `tle_sma_change_hourly` (+ `lag_r2_scan`, `best_lag_value` queued): a =
  (mu/n²)^⅓, round epoch to hour, diff, May filter, **groupby(hour).mean()** → TypeError
  tz-aware vs str (near-miss 3).
- step 5: Timestamp compare → TypeError tz-aware vs naive (near-miss 4).
- step 6: tz-normalize → **126×2** (134 TLEs → 126 hourly rows: duplicate hours AVERAGED —
  gold's exact dedup grain) ✓ plans 3/4/5.
- step 7 `lag_r2_scan`: lag 0..48, AP shifted forward, inner join, corr² → **best 24,
  r² = 0.663, n = 121** ✓ plan-6.
- step 8: Final Answer: 24 ✓.

First semantic divergence vs gold: **none** (lag 0..48 vs gold's 0..47 and a May-only AP
slice vs gold's Apr–Jun window are harmless variants; same argmax). All four errors (spec
path, KeyError, 2× tz TypeError) were mechanical, loudly rendered, and each fixed in ≤2
edits with zero semantic wobble.

## Why Y succeeded but X failed

**The rendered evidence was identical at every choice that decided the outcome — this is a
method-choice/chronic-variance case, not a lever story.** The three decisive divergences
all happened at step 0, in contexts containing only the task prompt + file manifest
(nothing had rendered yet in ANY arm): the winner wrote `read_csv(delim_whitespace=True)`
where both losers wrote bare `read_fwf`, and the winner wrote a 2-line TLE parse where
Latest3k wrote a 3-line stride. The winner's `ap = col 49` (step 2) was likewise prior
knowledge, not rendered evidence: the ap position was never rendered in any arm — both
Delta arms saw the same ~100-char truncated spec cell,

> `[omni2_spec] Output 1x1: result: | Output Table: 1 rows, 1 cols | content | 0	                              O M N I 2  DATA SET\n … Th... | Schema (1 cols): content (str)`

and the same plausible raw render (`Output Table: 8784 rows, 55 cols`, first row
`2024 1 0 2596 51 52 …` — byte-similar between fwf and whitespace loads). Skill rule 5's
reject condition applies verbatim: divergence predates the arms' first rendered
difference.

**Delta5k (C1) did not fail for lack of render budget — it had MORE.** Everything rendered
before its step-4 hinge, in full:

> `[omni2_2024_ap] Output 0x0: Inputs: omni2_2024_raw (8784 rows, 55 cols) | result: | Inputs: omni2_2024_raw (8784 rows, 55 cols) | Output Table: 0 rows, 0 cols | Schema (2 cols): time (datetime), ap (numeric)`

Nothing in that line distinguishes "your DOY/hour reading is right but the fwf table is
corrupt" from "your column mapping is wrong" — the agent guessed the latter and
reinterpreted DOY as month, which un-emptied the table with garbage rows. The refutations
that WERE rendered afterward (814 rows > 720 possible May hours; 330 unique "hours";
n = 49 join at the lag scan) were ignored. A 5k budget renders this identically to 3k;
the missing artifact (a per-column min/max showing doy max = 9) is a stats-block artifact,
and SchemaOnly withheld it from BOTH arms equally.

**Latest3k (C3) churned DESPITE the evidence, not for lack of it — the levers-report
"history-blindness" attribution is rejected.** Every probe's 0x0 result re-rendered every
step (latest mode re-renders the full DAG), and the engine additionally rendered explicit
anti-churn reflections in **22 of 25** step contexts (re-verified in `react_steps.json`),
e.g. the step-10 context:

> `Attempt reflection: you have edited operator `omni2_spec` 6 times; 5 prior variant(s) did not converge. | Latest result: produced output but the task is not yet solved. | Do NOT submit another near-variant. Change strategy: inspect this operator's UPSTREAM result/columns for a wrong assumption, try a DIFFERENT decomposition, or build the answer from a different operator.`

That is precisely the correct advice (the wrong assumption WAS upstream, in
`omni2_2024_raw`), rendered alongside the step-9 `timecheck` contrast (January rows parse
perfectly; the May mask is empty) — richer anti-repetition signal than DELTA's change-log
provides. The colscan was still resubmitted six times (0.995 similarity at the last pair)
and then rebuilt as three new operators. Delta3k's rendered history contains nothing that
would have prevented this loop, because the one missing fact — WHY May is empty in a
bare-fwf table — is never rendered under any history mode at SchemaOnly. The winner simply
never entered the loop, on the strength of two step-0 method choices made with zero
rendered difference between arms. And Latest3k carried an independent second defect
(44/134 TLEs → 43 drag points) that would plausibly have kept its answer wrong even had it
escaped the AP trap. What the `latest` lever did do is shape the failure **topology**:
Latest burned $0.451 and 25 steps failing loudly where Delta5k failed cheaply and wrong at
$0.118 — consistent with the known Latest-thrash tail (identical-probe flag fired), but
shaping how a doomed run fails is not flip attribution.

## Pair verdicts

**C1 — Delta3k > Delta5k: CHRONIC.** Identical rendered evidence at every aligned decision
(same truncated spec cell, same plausible 8784×55 render, same loader hint); the winner's
whitespace-CSV load and ap=49 came from priors at steps 0–2, before any rendered
difference existed; the loser's hinge was a guess against an evidence-free `0 rows, 0
cols` render that more render budget could not enrich. Task chronic; accept rules not met.

**C3 — Delta3k > Latest3k: CHRONIC (history-blindness REJECTED-method-choice).** The churn
is real (6× colscan, 0.995-similarity resubmission, step-22 rebuild, 18 ops, $0.451,
step-cap death) but not history-starved: every probe result and 22/25 explicit "do NOT
submit another near-variant / inspect UPSTREAM" reflections were in context. Divergence
(3-line TLE stride + bare fwf, both step 0) predates the arms' first rendered difference;
task chronic.

**Cross-cutting note for the deck:** the 3k-solo win is not a render-budget effect at all.
The discriminator was the OMNI2 load method: bare `read_fwf` width-inference on a
January-headed year file silently destroys the DOY column and manifests only as
evidence-free empty outputs downstream. The cheapest lever that would have surfaced it is
a stats-bearing render (per-column min/max: doy max = 9 under the "May" mask) — withheld
by SchemaOnly in every arm here. The gold plan's duplicate-epoch item (groupby-mean per
rounded hour) discriminated nothing: all three arms handled it consistently with gold on
the TLE side (134→126, 134→125, 44→43).
