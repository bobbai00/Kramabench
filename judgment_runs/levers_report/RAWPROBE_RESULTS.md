# RawProbe pilot — retrying the format-blinded failures (2026-07-16)

One-knob prompt intervention (protocol prepended to the task prompt; agent
service untouched): raw-preview before loaders (all formats incl. CSV), read
named spec files, post-load sanity ranges, join-key null/overlap check,
delete probes after verification. SUTs: `Delta5kSchemaOnlyRawProbe`,
`Latest3kSchemaOnlyRawProbe`, `DeltaStats3kD2RawProbe`. Targets = the
format-blinded failures from the deep dives; judged at the MECHANISM level
in traces (targets are chronic flippers — single-run pass/fail alone is
noise).

## Results (10 runs)

| task @ base arm | base outcome | RawProbe outcome | mechanism verdict |
|---|---|---|---|
| wildfire-hard-17 @Latest3k | no answer (25 ops, 20 sinks, churn) | **4830.9 = GOLD, score 1.0** (14 steps, $0.101) | **FIXED, confirmed**: trace shows raw previews + a null-audit op over ALL candidate ID columns (`raws_station_id_nulls`, `nwsid_example`) → picked and cleaned `NWS ID` → exact join. Protocol step 3 executed verbatim. |
| archeology-hard-1 @Stats3kD2 | 380.82 (window-mean, wrong regime) | **8477.86** — within **1.2%** of gold 8577.53 (13 steps) | **Mechanism fixed** (probe → correct header/units → correct interpolation regime); scores 0 only because `numeric_exact`. 22× closer. |
| environment-hard-9 @Latest3k | f1 0.36 (rain column as enterococcus) | **f1 0.80** (6 steps) | Improved: raw preview drove a correct column mapping for most beaches. |
| environment-hard-9 @Delta5k | f1 0 (silent concat mis-alignment) | f1 0.40 (7 steps) | Improved, partial. |
| astronomy-hard-9 @Delta5k | 19 (silent `read_fwf` DOY corruption) | no answer (26-step cap) | **Parse FIXED — failure converted, not cured**: it raw-probed and chose the whitespace loader (the exact trap avoided), but spent 10 probe ops + 6 spec-mining ops (`omni2_text_find_columns/_snip/_search_ap/_grep`) and step-capped before the 49-lag sweep. Silent-wrong → loud-budget-death. |
| astronomy-hard-9 @Latest3k | no answer (churn) | no answer (26 steps) | Same conversion; probe cost + Latest op-minting still exhausts the budget. |
| wildfire-hard-17 @Delta5k | 3317.4 (all-station mean) | 3317.4 (6 steps) | Unchanged — probed the files but NEVER PLANNED A JOIN: it read "stations used for fire monitoring" as "all RAWS stations", so the key-check clause never triggered. Residual failure is interpretation, outside the protocol's reach. |
| archeology-hard-2 @Stats3kD2 | 47.29 | 49.60 (8 steps) | Unchanged — the chronological-re-sort bug is downstream of loading, exactly as predicted. |
| CONTROL wildfire-easy-9 @Delta5k | pass, 4 steps | pass (0.91), 6 steps | +2 steps probe overhead on a clean task. |
| CONTROL astronomy-easy-2 @Stats3kD2 | pass, 4 steps | pass (1.0), 4 steps | zero overhead. |

Tally on the 8 targets: **1 exact fix (mechanism confirmed) + 1 near-fix
(1.2% off) + 2 partial improvements + 2 unchanged-as-predicted
(downstream/interpretation) + 2 converted (parse fixed, budget died).**
Controls: 0–2 steps overhead, no accuracy loss. Chronic caveat applies to
pass/fail; the mechanism evidence (probes executed in every target trace;
load/key decisions demonstrably changed in 4) is the durable result.

## What the pilot teaches (feeds DYNAMIC_KNOBS)

1. **The protocol works exactly where the failure lives at the data edge**
   (loader choice, key choice, column mapping) — and nowhere else
   (interpretation, downstream method): clean confirmation of the deep-dive
   taxonomy.
2. **Probe cost is the real tradeoff — and it argues for ENGINE-side S6.**
   astronomy-hard-9 shows prompt-driven probing competes with the step
   budget (10 preview ops where the protocol said one per file). If the
   engine attached 2–3 raw lines of the source file to every load-op
   observation (render-level S6), the same information would cost ZERO
   steps and no cache churn (write-time, append-only). Same for S5: a key
   null/overlap micro-profile rendered at join time beats asking the agent
   to build audit ops.
3. **Failure-mode conversion is progress**: silent-wrong → loud-budget-death
   is strictly better (visible, recoverable with a bigger step budget or
   cheaper probes; astronomy-hard-9 would likely land with engine-side
   raw-lines + its old 10-step trajectory).
4. **The interpretation residue** (wildfire-hard-17@5k "used stations") is
   the convention family again — no probe protocol reaches it.

## Next (not run yet)

- Engine-side S6/S5 (raw-lines on load observations; key micro-profile on
  joins) as flag-gated agent-service features — the zero-step version of
  this pilot.
- Step-budget interaction: rerun astronomy-hard-9 RawProbe at max_steps 35
  to confirm the converted failure lands.
- Full-benchmark RawProbe arm (with recovery + twin pair) before any
  aggregate claim.

## Addendum (2026-07-17)

The harness-side `_RawProbeMixin` and the three `*RawProbe` SUT classes were
REMOVED per Bob's direction: the guidance belongs in the agent, not the
benchmark prompt. It has been promoted into agent-service — the probe
protocol as principles in `prompts/code-mode.md` and the demonstrated beat
(raw preview → explicit-parameter loader → verify → delete probes) in the
worked e2e examples. This is a PERMANENT prompt change: every subsequent run
of ANY SUT carries it, so comparisons against pre-change scratch dirs are
cross-vintage (use fresh controls). The pilot scratch dirs
(`system_scratch/*RawProbe/`) remain as provenance for the results above.
