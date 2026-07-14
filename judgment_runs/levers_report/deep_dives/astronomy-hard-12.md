# Failure dive — astronomy-hard-12 (all-arm common-core failure)

## Task
Q: Estimate the mean geopotential energy per unit mass (J/kg) experienced by
Swarm-A from Sep 2–29, 2019 using precise orbital data. Use **SP3 files** to
determine the satellite's geodetic position, interpolate a mock geopotential
field defined over (lat, lon, alt) to compute PE at each timepoint, average.
Earth radius 6371.0 km, g = 9.80665. Round to 2 dp.

D: SP3 precise-orbit files (a specialized GNSS ephemeris format — ECEF
position at fixed epochs) for the Sep 2019 window, plus the mock geopotential
field definition over (lat, lon, alt).

## Solution
Parse SP3 → convert ECEF → geodetic (lat, lon, alt) → interpolate the mock
geopotential field at each orbital point → PE per unit mass → mean over the
window = 66,822,738.84.

## What the arms do
All four arms **step-cap (26 steps) with no answer.** Blockers: (a) parsing
the SP3 ephemeris format, (b) ECEF→geodetic conversion, (c) 3-D field
interpolation at every epoch over a ~4-week window. Too many specialized
stages for the 25-step budget.

## What the gold dataflow does
Gold runs the full geodesy pipeline in a script. The agent cannot get through
SP3 parsing + coordinate conversion + interpolation before the step cap.

## Why it fell short
**Task-intrinsic complexity + execution-limit, render-invariant.** The
failure is specialized-format parsing (SP3) plus a multi-stage geodesy
computation — a pipeline-length and domain-tooling ceiling, not an
information-delivery gap. No render parameter helps.

## Cross-arm failure shape
| arm | steps | answer |
|---|---|---|
| DeltaStats3kD2 | 26 | (no response) |
| Delta3kSchemaOnly | 26 | (no response) |
| Delta5kSchemaOnly | 26 | (no response) |
| Latest3kSchemaOnly | 26 | (no response) |

**All four step-capped, no answer** — pure execution-limit, identical to
astronomy-hard-11. Not context-addressable.
