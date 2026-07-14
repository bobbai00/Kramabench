# Failure dive — archeology-easy-11 (all-arm common-core failure)

## Task
Q: What is the average latitude of capital cities? If there're more than one
capital in a country, only count the lat of the capital with the largest
population. Round your answer to 4 decimal places.

D: one file.
- `data/archeology/input/worldcities.csv` — 44,691 rows × 11 cols. Sample:

  | city | lat | country | capital | population |
  |---|---|---|---|---|
  | Tokyo | 35.6897 | Japan | primary | 37,732,000 |
  | Delhi | 28.61 | India | admin | 32,226,000 |

  Semantic key: the `capital` column is a **rank enum**, not a flag —
  `{primary, admin, minor, "" }`. `primary` = the national capital;
  `admin` = first-order administrative-division seat; `minor` = lower seat;
  blank = not a capital of any kind. "Capital city" in the gold sense is
  `primary` only.

## Solution
Single-source pipeline:
```
worldcities.csv
  → filter(capital == "primary")          # national capitals only
  → group(country) keep argmax(population) # tie-break rule from the question
  → mean(lat) → round(4)                   # = 17.4274
```

## What DeltaStats3kD2 does (best arm, FAIL 17.1667)
- STEP 0 `worldcities` — load the CSV (44,691×11). Render shows the
  `capital` values `primary`/`admin` in the first rows; the stats arm's
  profile also carries `capital top_5={"minor"=…,"admin"=…,"primary"=…}`.
- STEP 1 `capital_maxpop` --> **the problematic step.**
  `caps = worldcities[worldcities['capital'].notna()]` then sort by
  `(country, population desc)` and `drop_duplicates('country', keep='first')`.
  This keeps **every capital-ranked row (primary + admin + minor)** and picks
  the largest-population one per country → 238 countries. Gold keeps only
  `capital == "primary"` first, so its country set and its per-country city
  differ (a country whose highest-population capital-tagged city is an `admin`
  seat, or which has no `primary` at all, is scored differently).
- STEP 2 `avg_capital_lat` — `mean(lat)` = 17.16669 → "17.1667".

## What the gold dataflow does at the missed step
Gold's very first filter is `df["capital"] == "primary"` (exact-equals the
enum value), not `notna()`. The per-country max-population tie-break then
operates only within primary capitals. The 0.26° latitude gap is entirely
this filter: `notna()` admits admin/minor seats that shift the mean.

## Why it fell short
**Mis-read evidence (task-semantics), render-invariant.** The deciding fact
— that `capital` is a 4-value rank and "capital city" means `primary` — was
fully visible to every arm (the enum values render in the very first sample;
the stats arm additionally got the explicit `top_5` breakdown naming all
three ranks). No arm lacked the information; all four interpreted "capital
cities" as "any capital-ranked row." This is an interpretation choice about
the question, not an information-delivery failure — exactly why more rows
(5k), the stats profile (D2), and history (Latest) all land in the same
place.

## Cross-arm failure shape
| arm | steps | answer | vs gold 17.4274 |
|---|---|---|---|
| DeltaStats3kD2 | 5 | 17.1667 | −0.26, `capital.notna()` |
| Delta3kSchemaOnly | 5 | 17.1667 | identical mechanism |
| Delta5kSchemaOnly | 6 | 17.1900 | same misread, minor per-country tie difference |
| Latest3kSchemaOnly | 6 | 17.1667 | identical mechanism |

**Same failure, same mechanism, all four arms** — a shared semantic misread
of the `capital` enum. Render-config-invariant; belongs to the
"convention/interpretation" family from SEMANTIC_WALKS.md, here as a
common-core (not flip) case. Fixable only by disambiguating "capital city"
(a task/prompt-semantics lever), not by any render parameter.
