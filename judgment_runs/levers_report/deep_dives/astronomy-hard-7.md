# Failure dive — astronomy-hard-7 (all-arm common-core failure)

## Task
Q: Train a density-prediction model using OMNI2 variables (f10.7, Kp, Dst)
and GOES variables (xrsb, xrsa) to forecast Swarm-Alpha density 4 hours
ahead. Use a 16-hour context window, project inputs forward with a **VAR(1)**
model, then fit a **linear regression** to predict the next 4 hours of
density. Train on wu334 windows, evaluate on wu335, report RMSE over the
4-hour forecast.

D: OMNI2 fixed-width series (f10.7_index, Kp_index, Dst_index_nT), GOES flux
CSVs (xrsb/xrsa observed), Swarm-Alpha density series, and the wu334/wu335
window definitions — five aligned time series with a single-timestamp
overlap rule between the input window end and the density start.

## Solution
A multi-stage modeling pipeline: align 5 series on the specified windows →
VAR(1) forward-projection of the 16h context → linear regression density
head → RMSE on the 4h forecast = 1.211e-13. Every stage (window alignment,
VAR order, the "overlap at a single timestamp" boundary, RMSE horizon) is a
precise spec.

## What the arms do
- DeltaStats3kD2 (18 steps) → **1.588e+10** (units/scale blow-up — a density
  ~1e-12 vs a 1e10 output means the regression target or feature scaling is
  wrong by ~22 orders).
- Delta3kSchemaOnly (18) → 5.125e-14; Delta5kSchemaOnly (20) → 7.105e-13 —
  right order of magnitude, wrong window/VAR handling.
- Latest3kSchemaOnly (26) → step-capped, no answer.
--> No arm reproduces the VAR(1)-then-regression pipeline on the exact
train/eval windows; each breaks at a different modeling stage.

## What the gold dataflow does
Gold implements the literal spec: VAR(1) projection of the 16h window, then
a linear regression head for the 4h horizon, on the wu334/wu335 windows with
the single-timestamp overlap. RMSE = 1.211e-13.

## Why it fell short
**Task-intrinsic modeling complexity, render-invariant.** This is a
multi-model time-series pipeline with ~6 stacked precise specifications; the
failure is in executing that pipeline correctly (window alignment, VAR order,
scaling), none of which is an information-delivery problem. More rows / the
stats profile / history cannot supply a correct VAR(1)+regression
implementation. The Latest arm's step-cap death additionally shows the
25-step budget is too tight for a pipeline this long.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 18 | 1.588e+10 (scale blow-up) |
| Delta3kSchemaOnly | 18 | 5.125e-14 |
| Delta5kSchemaOnly | 20 | 7.105e-13 |
| Latest3kSchemaOnly | 26 | (no response — step-capped) |

Gold 1.211e-13. **Four different failure stages** — a task-intrinsic hard
modeling problem, not a context-lever problem. Belongs to the "task
complexity / execution-limit" family that dominates the astronomy portion of
the common core.
