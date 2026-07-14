# Failure dive — astronomy-hard-11 (all-arm common-core failure)

## Task
Q: Using OMNI2 data, run the **NRLMSISE-00** atmospheric model to predict
neutral density for Swarm-B throughout 2024. Derive model inputs (F10.7,
F10.7A, daily Ap, 3-hour Ap vector) from OMNI2. Compare against measured
neutral density from Swarm-B POD files and report RMSE over the year. OMNI2
format spec in `omni2.text`; 3-hour Ap per the MathWorks NRLMSISE-00 page.

D: OMNI2 fixed-width yearly file (needs the `omni2.text` column spec to parse),
`omni2.text` (the format spec itself), and Swarm-B POD neutral-density files
for all of 2024 (year-long series).

## Solution
Parse OMNI2 via its spec → derive the four NRLMSISE-00 driver inputs
(incl. the 3-hour Ap vector with the specific lag convention) → run the
NRLMSISE-00 physical model (`pymsis`) for every Swarm-B timestamp in 2024 →
RMSE vs measured = 4.638e-13.

## What the arms do
All four arms **step-cap (26 steps) with no answer.** The pipeline requires:
(a) parsing a fixed-width file against an external spec, (b) constructing the
3-hour Ap vector to a precise lag convention, (c) invoking a domain physics
model over a full year of timestamps. This does not fit in the 25-step
budget, and the density render never stabilizes.

## What the gold dataflow does
Gold runs the full physical-model pipeline end-to-end in a script (no step
budget). The agent cannot reach the model-invocation stage before exhausting
its steps.

## Why it fell short
**Task-intrinsic complexity + execution-limit, render-invariant.** The
blocker is pipeline length and physics-model orchestration, not context: no
sampling/stats/history parameter shortens a year-long NRLMSISE-00 run or
supplies the Ap-vector convention. This is a step-budget/tooling ceiling.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 26 | (no response) |
| Delta3kSchemaOnly | 26 | (no response) |
| Delta5kSchemaOnly | 26 | (no response) |
| Latest3kSchemaOnly | 26 | (no response) |

**All four step-capped, no answer** — pure execution-limit. Not
context-addressable; would need a larger step budget and/or a physics-model
tool. Common-core by construction (every arm hits the same wall).
