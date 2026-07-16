# TODO — raw-probe-first prompt change (dataflow agent) — NOT YET IMPLEMENTED

Planned agent-service prompt experiment, motivated by the deep-dive evidence.
Parked here 2026-07-16; execute later.

## Evidence that motivates it

- **astronomy-hard-9**: all arms chose their OMNI2 loader at step 0, blind
  (~1.1k-char context, zero data rendered). Loser used bare `read_fwf`
  (width inference) → silent day-of-year truncation → May window
  unreachable; the corrupted parse LOOKS valid, so the step-1 render
  triggers no repair. The task even named the format-spec file; no arm read
  it as a spec.
- **Cross-architecture control**: the PLAIN code agent (no custom prompt)
  passed in 4 steps by native REPL habit — read the spec, chose a robust
  loader deliberately, printed head/shape to verify. The GUIDED code-agent
  prompt contains the verbatim proactive rule ("Before loading a file with
  `pd.read_csv()`, inspect the raw content first" + plausibility checks).
- **The dataflow prompt today** (`code-mode.md` "Handle messy data files")
  is only REACTIVE: probe raw content *after* loading and only *if*
  symptoms appear ("Unnamed:" columns / data-as-header). Two holes:
  (a) silent truncation shows neither symptom, so the rule never fires;
  (b) "Load data files directly in a single operator" actively discourages
  a pre-probe op (it was written to prevent probe sprawl).
- **environment-hard-9**: same law from the other side — the winner was
  saved by a LOUD parse failure (0 rows → forced re-parse); losers accepted
  silent plausible mis-parses (rain column as enterococcus).

## The change (three parts — Bob's spec)

1. **Prompt principle (proactive, NOT gated to non-CSV).** Standard formats
   are also dirty (beach datasheets are .csv with 7–11 unnamed cols; legal
   CSVs carry metadata rows; climateMeasurements.xlsx hides its header at
   row 6). Rule: before writing a loader for ANY file whose structure is
   not yet verified, first create a raw preview op (read the first N lines
   as text; also mid/end lines for large files — early rows can be
   unrepresentative, e.g. 1-digit day-of-year in January); if the task
   names a format-spec file, READ IT and derive explicit parse params
   (widths/sep/header/skiprows); after loading, plausibility-check key
   columns (ranges, dtypes, row count vs file size). Soften the
   "load directly in a single operator" clause accordingly.
2. **Demonstrate in the worked e2e examples** (per the standing rule:
   principles stay ad-hoc-example-free; demonstrations live in the existing
   example sections `example.latest*.md` / `example.delta*.md`). Add the
   beat: `raw_preview` op → read spec/head → informed loader with explicit
   params → verify → proceed.
3. **Deletion hygiene for probes.** Instruct (principle + demonstrated in
   the same examples) that once the full load is verified, the raw-preview
   /sample probe ops are DELETED. Otherwise this change reintroduces
   orphan-sink accumulation — the churn signature (sink-share ≥50% flag)
   and render cost we measured. Pairs with the existing "Dataflow hygiene"
   delete principle and the delete-unused-investigation-operator examples.

## Design cautions

- **Cost tension**: +1 probe op/step per unverified source. The delete rule
  offsets DAG/render growth; consider one combined preview op for several
  files of the same family. Measure, don't assume.
- **Where it lands**: agent-service `prompts/code-mode.md` + example files,
  flag-gated like render-prefs (orthogonality-test pattern) so the baseline
  stays byte-clean.

## Measurement plan (when we do it)

- New SUT on the DeltaStats3kD2 base (+ a fresh same-vintage control),
  full run + symmetric 2× recovery, `kb.py venn` + `case-metrics`,
  chronic-flipper gate.
- Accuracy targets: astronomy-hard-9, environment-hard-9 family (the two
  render-invariant silent-parse cases the code agent beats us on);
  secondary: archeology dirty-header trio (should shorten steps even if
  answers stay wrong).
- Guardrails: steps/cost on clean-CSV tasks must not inflate (the
  anti-churn intent of "load directly" must survive); sink-share flag rate
  must not rise (delete rule working).
