# Failure dive — archeology-hard-12 (all-arm common-core failure)

## Task
Q: Count the number of human conflicts between 800 and 1400 AD, and attribute
them as best you can to modern countries. Define a conflict as between two
actors that lasts at least a year.

D: two files.
- `conflict_brecke.csv` — the Brecke conflict catalog: `Conflict` (a name like
  `"Byzantine-Bulgarian War"`), `StartYear`, `EndYear`, region.
- `worldcities.csv` — used only for its `country` column (the set of modern
  country names to attribute actors to).

## Solution
```
conflict_brecke.csv
  → filter(800 ≤ StartYear, EndYear ≤ 1400)  [era window]
  → filter(EndYear - StartYear ≥ 1)          [lasts ≥ a year]
  → split Conflict name on "-" or "and" → actor_a, actor_b
  → get_matching_word(actor, countries): keep only conflicts where an actor
     name CONTAINS a modern country name (substring, case-insensitive)
  → count = 409
```
The load-bearing, idiosyncratic step is the **country-attribution filter**:
gold DROPS conflicts whose actors don't substring-match a
`worldcities.country` value.

## What DeltaStats3kD2 does (best arm, FAIL 447)
- STEP 0–1 load `conflict_brecke` + `worldcities`.
- STEP 2 `conflicts_filtered` --> **the problematic step.** Applies the era
  window (800–1400) and the ≥1-year duration rule, but attributes/counts
  **all** surviving conflicts — it does not require an actor to match a
  modern country, so nothing is dropped on attribution.
- STEP 3 `conflict_count` → 447.

## What the gold dataflow does at the missed step
Gold additionally runs `get_matching_word` on both actors and keeps a
conflict only if at least one actor name contains a modern country string.
That attribution filter removes 38 conflicts (447 → 409). "Attribute them as
best you can to modern countries" is read by gold as a *filter*, by the arms
as a *no-op annotation*.

## Why it fell short
**Convention misread, render-invariant.** The instruction "attribute … as
best you can" is genuinely ambiguous — gold operationalizes it as a
substring-match drop; the natural reading (all four arms took it) is "count
the conflicts, attribution is descriptive." Both `conflict` and `country`
data were fully rendered; nothing about rows/stats/history disambiguates a
verbal instruction. All four arms produce the **identical** 447.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 5 | **447** |
| Delta3kSchemaOnly | 5 | 447 |
| Delta5kSchemaOnly | 5 | 447 |
| Latest3kSchemaOnly | 5 | 447 |

**All four identical (447 vs gold 409)** — maximal render-invariance: a
convergent interpretation of an ambiguous attribution instruction. Same
family as archeology-easy-11. Not addressable by any context lever; only by
pinning the attribution semantics.
