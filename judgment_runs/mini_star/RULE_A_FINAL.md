# Rule A final verdict — 8 reps per arm

**Runs:** A1RolePolicyReplicate1-8 vs A0ControlReplicate1-8, 20 hard tasks each = 320 runs.
Reps 1-3 (A1, A0) @ `4af1e98da` (14:25 pool); reps 4-8 + A0r4 @ `9d60d01dc` (17:30 pool, ORCHA3).
Cross-vintage is safe: golden parity holds for these configs (render byte-identical) and the A0r4
sentinel (60.0) sat inside the A0 spread. All 16 arm-reps: instant-fails 0, quota 0, `dirty=False`.

## Result

| arm | acc (avg ± rep std) | $/task | steps | reasoning/task | reps |
|---|---|---|---|---|---|
| A1 rich-source/lean-interior | **64.4 ± 8.4** | **0.0239** | 7.3 | 7,277 | 8 |
| A0 control (uniform render) | 59.1 ± 12.3 | 0.0241 | 7.4 | 7,596 | 8 |

A1 − A0: **+5.3pt acc, −0.9% cost, −0.1 steps, −319 reasoning**.
Per-rep: A1 `79/64/59/65/55/77/57.5/59`, A0 `69/66.5/51.5/60/75/32.5/64/54.5`.

## Verdict: NOT ESTABLISHED as an accuracy rule — but adopt A1 anyway

Pre-registered rule: call REAL only if |Δ| ≥ ~2×SEM and not carried by ≤2 task flips.

- SE of the difference at 8 reps = √(8.4²+12.3²)/√8 ≈ **5.3** — the +5.3 delta is exactly 1 SE.
  The control's variance *doubled* with more reps (one 32.5 rep, one 75.0 rep), so more data made
  the significance worse, not better. The measured run-level floor was honest.
- Flip structure confirms dice-reshuffling: the 3-rep flips (`biomedical-hard-8`,
  `archeology-hard-7`) **washed out** at 8 reps; a different task (`legal-hard-16`: A1 0.62 vs
  A0 0.12) now carries the delta alone.
- Yet the point estimate is stubborn: +5.0 (3 reps) → +5.6 (4 controls) → +5.3 (8 reps). Direction
  never flipped. If Rule A has a true effect it is ~+5 and this benchmark's variance cannot
  certify it at any rep count we can afford.

**Decision: A_win = A1.** At 8 reps it is point-positive on accuracy AND now point-cheaper
(−0.9%, flipped from +2.9% at 3 reps) with fewer steps and less reasoning. Weak dominance:
nothing about A1 costs anything, so the render policy ships on cost grounds with a possible
free ~+5 upside. The stats-density pool (see STATS_DENSITY_REPORT.md) independently fixed A1's
configuration: full-density source stats — every trim backfired monotonically.

One honest observation for the research write-up: the lone surviving flip, `legal-hard-16`, is a
**provenance-gold multi-file task** — the same failure family the trace deep-dive identified. If
A1's rich-source render helps anywhere specific, it is exactly where per-file structure matters.
That is the A4 arm's hypothesis, tested directly next.

## Next (in flight)

A4 = A1 + `sourceProvenanceHint` (prompt principle: multi-file loaders must carry a
`__source_file` column). Falsifiable: must lift `legal-hard-29` (44% base) and `legal-hard-16`
(28% base across all arms; A0 8-rep mean 0.12). If legal-29 doesn't move, the prompt form is
insufficient → data-layer injection.
