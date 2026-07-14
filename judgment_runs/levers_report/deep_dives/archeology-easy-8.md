# Failure dive — archeology-easy-8 (all-arm common-core failure)

## Task
Q: How many unique sources were used in the Roman cities dataset?

D: one file.
- `data/archeology/input/roman_cities.csv` — 1,388 rows × 12 cols. The
  relevant column is `Select Bibliography`, a **semicolon-separated list of
  citation strings**, e.g.:
  - `BNP; Hansen 2006; Hansen and Nielsen 2004; PECS; Sear 2006.`
  - `BNP; DGRG; PECS; Sear 2006.`

  Semantic key: a "source" is one citation token *inside* the field, not the
  whole field. Tokens carry trailing punctuation (`.`, `?`) and some have a
  `page:` style suffix after a colon.

## Solution
```
roman_cities.csv
  → filter(Select Bibliography notna)
  → explode on ";"                         # each row → many citation tokens
  → normalize each token: strip; drop "?" and "."; cut at ":" (keep prefix)
  → set() dedup → count = 52
```
The normalization is the whole task: split, punctuation-strip, colon-prefix,
dedupe.

## What DeltaStats3kD2 does (best arm, FAIL 872)
- STEP 0 `roman_cities` — load (1,388×12); the render shows the raw
  `BNP; Hansen 2006; …` field contents.
- STEP 1 `unique_sources_count` --> **the problematic step.**
  `s.dropna().nunique()` — counts distinct **whole bibliography strings**
  (872 of them), never splitting on ";". Wrong granularity entirely: it
  answers "how many distinct bibliography lists" instead of "how many
  distinct sources."
- STEP 2 → "872".

## What the gold dataflow does at the missed step
Gold explodes each field on ";" into individual citation tokens, normalizes
each (strip, remove `?`/`.`, cut at `:`), and dedupes the token set → 52.
The step the arm skipped is the explode-and-normalize; it treated a
list-valued cell as an atom.

## Why it fell short
**Underspecified parse convention (hidden spec).** Two layers: (1) the arm
mis-read the field as atomic — the `;`-separated structure was visible in
the sample it rendered, so a careful read could have split it; (2) even
splitting correctly, the exact normalization (drop `?`/`.`, cut at `:`) is
not stated in the question nor derivable from schema/samples — it is gold's
private convention. No render parameter (rows, stats, history) encodes a
string-normalization spec, so this is render-invariant.

## Cross-arm failure shape
| arm | steps | answer | what it did |
|---|---|---|---|
| DeltaStats3kD2 | 4 | **872** | `nunique()` on whole strings (no split) |
| Delta3kSchemaOnly | 4 | 82 | split, partial cleaning |
| Delta5kSchemaOnly | 4 | 1 | degenerate parse (collapsed to one) |
| Latest3kSchemaOnly | 4 | 55 | split + strip, missing `:`-cut / punct → slight over-count |

Unlike archeology-easy-11 (all four converge on ONE wrong value), here the
four answers **scatter (872 / 82 / 1 / 55 vs 52)** — the signature of an
*evidence-underdetermined* task: the parse spec is not in the context, so
each arm improvises differently. High variance, render-invariant. The
`Latest3k` 55 is closest (correct split, missing only the final punctuation
normalization). Fixable only by specifying the normalization
(task/prompt-semantics), not by a render lever.
