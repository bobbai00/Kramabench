# Astronomy Hard 7: Frontier-Decay Mechanism Audit

Audited 2026-07-10 from the historical Latest 3k control and the three saved
treatment watchdog traces.

## Verdict

Frontier decay made **no visible context change** in any treatment trace. The
full treatment and recovery 2 never produced a healthy operator. Recovery 1
first showed healthy loader results in agent step A22, but no consumer existed
in the input to that step; A22 then created the first consumer and the trace
ended before a subsequent prompt. Therefore no treatment prompt contained an
eligible decayed operator.

The treatment failures are the same loader-recovery/operator-churn pathology,
not evidence of an accuracy regression caused by frontier decay. The control
has the same initial pathology but stochastically escapes it at A11, then fails
accuracy later for a separate modeling reason.

Confidence is **very high (above 0.98)** that decay did not affect the visible
context. Recovery 2 strengthens this non-exposure conclusion but does not
change the causal verdict. Confidence about why independent model samples
escaped at different times remains **medium-high**: the deterministic bad-file
trigger is clear, but the runs are not checkpointed continuations with
identical random state.

## Artifacts

| Arm | Artifact | State / completeness |
| --- | --- | --- |
| Control | `system_scratch/DataflowSystemGPT52LatestStats3kD2/astronomy-hard-7` | Complete, 17 trace records (user + A1-A16) |
| Treatment full | `watchdog_traces/astronomy-hard-7_treatment_react_steps.json` and `_workflow.json` | Saved watchdog trace, 26 records (user + A1-A25) |
| Treatment recovery 1 | `watchdog_traces/astronomy-hard-7_treatment_recovery1_react_steps.json` and `_workflow.json` | Saved in `GENERATING`, 23 records (user + A1-A22) |
| Treatment recovery 2 | `watchdog_traces/astronomy-hard-7_treatment_recovery2_react_steps.json` and `_workflow.json` | Saved watchdog trace, 26 records (user + A1-A25) |
| Current treatment scratch | `system_scratch/DataflowSystemGPT52LatestStats3kD2FrontierDecay/astronomy-hard-7` | Incomplete overwrite: config, prompt, and zero-result evaluation only; not an additional usable trace |

Paths under `watchdog_traces/` are relative to this audit directory.

## Deterministic Trigger

The six exact filenames supplied in the task do not exist. The corresponding
repository files have broader date ranges:

| Input | Prompt filename suffix | Actual filename suffix |
| --- | --- | --- |
| OMNI2 wu334 | `20161022_to_20161024.csv` | `20160824_to_20161023.csv` |
| GOES wu334 | `20161022_to_20161024.csv` | `20160824_to_20161023.csv` |
| Density wu334 | `20161022_to_20161024.csv` | `20161023_to_20161026.csv` |
| OMNI2 wu335 | `20161025_to_20161029.csv` | `20160827_to_20161026.csv` |
| GOES wu335 | `20161025_to_20161029.csv` | `20160827_to_20161026.csv` |
| Density wu335 | `20161025_to_20161029.csv` | `20161026_to_20161029.csv` |

All four trajectories consequently begin with the same six parallel exact-path
loaders and receive the same `FileNotFoundError` class at A2. A sufficiently
broad `*wu334-*.csv` / `*wu335-*.csv` fallback is required to escape.

## Step-Level Comparison

### Control

- A1 creates `omni_wu334`, `omni_wu335`, `goes_wu334`, `goes_wu335`,
  `dens_wu334`, and `dens_wu335` with the nonexistent exact filenames.
- A2-A10 repeatedly modify the same six IDs. Even A7's recursive fallback is
  date-constrained, for example `**/OMNI2/*wu334*20161022*20161024*.csv`, and
  still cannot match the actual file.
- A11 is the first successful strategy change: it adds broad fallbacks such as
  `**/omni2-wu334-*.csv`, with analogous GOES and density patterns.
- A12 first shows all six healthy results: OMNI2 `1441 x 57`, GOES
  `86401 x 13`, and density `433 x 2`. It then creates
  `train_features_targets`.
- A13 modifies `train_features_targets`; A14 creates
  `eval_features_targets`; A15 creates `rmse_var1_linreg`; A16 answers
  `3.268e-13` versus gold `1.211e-13`.

### Treatment full

- A1 creates the same six logical loader IDs as the control:
  `omni_wu334`, `goes_wu334`, `dens_wu334`, `omni_wu335`, `goes_wu335`, and
  `dens_wu335`.
- A2-A25 modify all six on every step while every visible result remains an
  error. The attempts oscillate among relative, `/data`, `/mnt/data/data`,
  exact-name, and over-specific glob variants.
- A24 uses `/data/<exact-filename>`; A25 returns to
  `data/<exact-filename>`. Neither can match the actual filenames.
- No downstream operator is created. The saved workflow is six isolated
  loaders and zero links.

Because every result is erroring and every node is still an active source with
no consumer, frontier decay is categorically ineligible throughout this trace.

### Treatment recovery 1

- A1 creates six equivalent loaders under IDs `omni334`, `goes334`, `dens334`,
  `omni335`, `goes335`, and `dens335` using the same nonexistent exact names.
- A2-A20 repeatedly rewrite those same six IDs and keep receiving errors.
- A21 finally adds broad directory-local patterns such as
  `data/.../OMNI2/omni2-wu334-*.csv`, with analogous patterns for all inputs.
- A22 first shows all six healthy, with the same table sizes as the control and
  with `Column Schema and stats:` still present for every loader. A22 then
  creates `rmse_var_lr`, consuming all six loaders.
- There is no A23 input prompt. Thus there is no context render after the
  consumer becomes healthy and no opportunity for the grace period to elapse.

### Treatment recovery 2

- A1 creates `omni_wu334`, `goes_wu334`, `swarm_wu334`, `omni_wu335`,
  `goes_wu335`, and `swarm_wu335` with the nonexistent exact filenames.
- A2-A25 resubmit those same six exact-path loaders on every step. For each
  operator, the code has only two byte-level variants: with or without a final
  newline. A few summaries add `(exact path from task)`; there is no substantive
  path or recovery-strategy change.
- Every prompt from A2 onward contains errors for all six operators and no
  `Output Table` block. No downstream operator is created; the saved workflow
  is six isolated loaders and zero links.

This trace never has a healthy candidate, a consumer, or a post-consumer grace
turn. Frontier decay is ineligible throughout A1-A25.

## Visible-Context Check

The scan treated a successful result as an `Output Table` block and looked for
the frontier-decay signature: schema retained, stats absent, and only the first
and last three rows for a table larger than three rows.

| Trace | Agent prompts | Successful operator blocks | Stats-off / decay signatures |
| --- | ---: | ---: | ---: |
| Treatment full | 25 | 0 | 0 |
| Treatment recovery 1 | 22 | 6, all in A22 | 0 |
| Treatment recovery 2 | 25 | 0 | 0 |
| Control | 16 | 36 repeated renderings | 0 (overlay disabled) |

This is stronger than merely observing similar model behavior: the treatment
feature's output representation is absent from every model input in all three
failed treatment trajectories.

## Dataflow Shape

The runs share the same six-loader prefix but do not reach the same final
topology:

```text
control:             6 loaders -> train/eval feature operators -> RMSE
treatment full:      6 loaders (all erroring; no links)
treatment recovery1: 6 loaders -> one monolithic RMSE/model operator
treatment recovery2: 6 loaders (all erroring; no links)
```

The broad semantic intent is similar, but only the control completes. This
shape difference cannot be attributed to decay because the first behavioral
divergence occurs before any prompt could contain a decayed operator.

## Mechanism Conclusion

The first meaningful divergence is the model's choice of recovery strategy:
control broadens its glob at A11, treatment recovery 1 does so at A21, the full
treatment never does by A25, and recovery 2 repeats its exact path through A25.
Frontier decay is not active at any of those decision points. Classify all
three treatment failures as **same pathology / no visible treatment
exposure**, and do not count this task as evidence that the rule harms
accuracy. Recovery 2 increases confidence in that classification; it does not
alter it.
