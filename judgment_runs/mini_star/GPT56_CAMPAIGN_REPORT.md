# gpt-5.6 (luna / terra) context-knob campaign — close-out

**2026-07-31 to 08-01.** Three rounds, 3,120 runs, all KramaBench-scored, every
comparison against a co-run control on one engine. Models route via litellm
`openai/responses/` at reasoning_effort=medium (both reject function tools on
/v1/chat/completions with any effort other than "none").

## Round 1 — the anchor/C1-C4 factorial (both models, 3 reps, full 104)

| arm | config | mini | luna | terra |
|---|---|---|---|---|
| Anchor | 1K DELTA | 63.3 | 68.6 | 73.3 |
| C1 +sampling | 5K DELTA | +3.3 | −0.2 | +1.7 |
| C2 +stats | 1K DELTA stats+hints | +1.2 | +1.5 | −0.4 |
| C3 +latest | 1K LATEST +code | **+5.4** | −0.0 | −1.4 |
| C4 all three | 5K LATEST stats+hints+code | +5.3 | −1.1 | **+2.8** |
| factorial span | | 5.6 pt | **2.6 pt** | 4.1 pt |

Terra's bare anchor (73.3) beats gpt-5-mini's best-ever arm (D8F 71.2). Terra C4
hit 76.0 — the program record — but no 5.6 delta reached 2× SE, and the knobs do
not transfer across models (LATEST: mini +5.4, terra −1.4).

## Round 2 — was 76.0 the ceiling? (terra, co-run control)

| arm | acc | vs T0 |
|---|---|---|
| T0 = C4 control | 71.5±1.8 | — |
| T1 5K DELTA stats | 72.8±1.7 | +1.3 (0.93×) |
| T2 C4 @ 10K | 71.7±2.3 | +0.2 (0.12×) |

Both null. Sampling saturates at 5K for terra like every other model; DELTA-vs-
LATEST is a wash. The control also exposed −4.5 pt era drift vs round 1 (same
config, 76.0 → 71.5), reconfirming that cross-pool comparisons are unusable.

## Round 3 — reasoning effort (the only axis untested)

Motivation: failing runs burn ~2× the reasoning tokens of passing runs on both
models (terra 551 vs 1,222; luna 691 vs 1,391).

| arm | acc | hard | $ | reasoning |
|---|---|---|---|---|
| E0 medium (T1 config) | 72.0±1.3 | **62.6±0.5** | 0.0115 | 587 |
| E1 high | 70.1±0.7 | 59.8±0.9 | 0.0145 | 1,069 |

**E1−E0 = −2.0 pt at 2.26× SE; hard −2.7 at 4.60× SE; cost +25.8%.** The single
strongest result of the campaign, and it is negative: forcing more reasoning
makes terra WORSE, most sharply on hard tasks. The 2× reasoning signature on
failing runs was overthinking as a symptom; raising the ceiling deepens it —
consistent with failures being method-choice errors, where extra deliberation
lets the model talk itself into a more complicated wrong method.

## Conclusions

1. **Context knobs are a weak-model phenomenon.** Knob sensitivity: mini 5.6 pt
   span, terra 4.1, luna 2.6 — inversely ordered with model strength, and no 5.6
   delta cleared 2× SE in any round.
2. **Effort is not a rescue lever either** — high is strictly worse (first ≥2× SE
   result on 5.6, and it points down).
3. **Terra's operating point: ~72-73 ± 2 at ~$0.011/task** (5K, stats, either
   mode, effort medium). Cheaper AND better than gpt-5-mini's best (71.2 at
   $0.0153). Luna: ~69 at ~$0.010.
4. **What remains is not tunable from the render**: 14 tasks fail on both models
   in all 15 runs each with the same wrong answers (method/interpretation);
   failures show 0 mechanical signatures.
5. Open engine defect: `Files read:` never renders on 5.6+DELTA (0 across 6
   arms × 3 reps) while LATEST renders 11-19/104 and mini+DELTA renders ~19.
   Immaterial to these conclusions; worth an engine-side instrumentation fix.

---

## POST-REPAIR CORRECTION (2026-08-02) — read this before any number above

Everything above was computed from pools that silently contained **infra zeros**:
when the memory watchdog recycled the engine, in-flight runs had
`Error: Unable to connect ...` written as their *response*, `evaluate.py` still
printed `Total score is: 0.0`, and the score-only resume check treated the dead
run as complete. 272 such runs were found across the campaign (~6.5 per arm);
228 were re-run against a live engine. The residual 44 are all the same task
(`wildfire-hard-19`, a ~1 GB load that kept getting killed mid-flight); they cost
each arm at most 0.4 pt and are flagged in the final table.

Three claims above are **retracted**:

1. **"−4.5 / −5.3 pt era drift" does not exist.** It was the infra zeros, not a
   day-to-day endpoint difference. Cross-pool comparison discipline still holds
   for other reasons, but terra's round-1 edge was never a drift artifact —
   `TerraC4 = 76.0` reproduces at n=5 (76.0 ± 2.4).
2. **Every dollar figure above uses gpt-5-mini pricing** and is wrong for these
   models. True litellm registry prices per M tokens (in / cached / out):
   luna $0.20 / $0.02 / $1.20, terra $2.00 / $0.20 / $12.00 — terra costs ~9×
   luna, which reverses the value recommendation.
3. **C5's effect size shrinks.** With repaired data: luna +2.0 (2.44× SE),
   terra +1.5 (1.39×), inverse-variance combined **+1.8 at ~2.8× SE** — still the
   one surviving 5.6 render knob and still an interaction (stats and sampling are
   each null alone), but not the +2.4 / 3.8× first reported. It also halves
   rep-to-rep std (±0.8-0.9 vs ±2.4).

**Authoritative post-repair numbers: `LUNA_TERRA_FINAL_TABLE.md`** (n=5, Anchor
through C5, both models, official KramaBench scores, true pricing, ± = population
std across per-rep means). Headlines:

| operating point | acc All | $/task |
|---|---|---|
| luna C5 (5K DELTA + stats) — best value | 73.4 ± 0.8 | 0.0098 |
| luna C3 (1K LATEST) — cheapest | 70.3 ± 1.5 | 0.0072 |
| terra C4 (5K LATEST + stats + code) — best absolute | **76.0 ± 2.4** | 0.0811 |

Conclusions 1, 2, 4 and 5 above survive the repair unchanged: knob sensitivity
still orders inversely with model strength, effort=high is still negative
(E1−E0 = −1.6 post-repair), the same tasks still fail deterministically, and the
`Files read:` DELTA defect is still open. Conclusion 3's operating point is
superseded by the table.
