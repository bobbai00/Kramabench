# E1 demand-paging analysis: full-push vs lean-push vs lean-push+pull

Shared tasks: 104 (control 104, lean 104, pull 104)

## Accuracy (pass = metric >= 0.9)

| Arm | Passes | Rate |
| --- | ---: | ---: |
| control (full push) | 80/104 | 76.9% |
| lean (rows=3) | 69/104 | 66.3% |
| lean+pull | 73/104 | 70.2% |

lean vs control flips: control-only 14 ['astronomy-easy-1', 'astronomy-easy-4', 'astronomy-hard-10', 'astronomy-hard-9', 'biomedical-hard-7', 'environment-hard-13', 'environment-hard-8', 'legal-easy-26', 'legal-hard-18', 'legal-hard-22', 'legal-hard-23', 'wildfire-hard-12', 'wildfire-hard-17', 'wildfire-hard-18']; lean-only 3 ['biomedical-hard-5', 'legal-hard-15', 'wildfire-easy-3']

pull vs control flips: control-only 12 ['astronomy-easy-1', 'astronomy-hard-10', 'environment-hard-10', 'environment-hard-13', 'environment-hard-7', 'environment-hard-8', 'legal-hard-18', 'legal-hard-22', 'legal-hard-23', 'wildfire-hard-12', 'wildfire-hard-16', 'wildfire-hard-17']; pull-only 5 ['biomedical-hard-5', 'environment-hard-9', 'legal-easy-19', 'legal-hard-15', 'wildfire-easy-3']

## Pull usage (treatment arm)

Tasks with >=1 pull: 33 / 104; total pulls: 129

Argument usage: {'maxRows': 124, 'where:distinct': 38, 'columns': 46, 'stats': 16, 'where:nonnumeric': 1}

| Task | Pulls | Score lean | Score pull |
| --- | ---: | ---: | ---: |
| `archeology-easy-4` | 5 | 1.00 | 1.00 |
| `archeology-hard-1` | 4 | 0.00 | 0.00 |
| `archeology-hard-2` | 3 | 0.00 | 0.00 |
| `archeology-hard-5` | 9 | 0.00 | 0.00 |
| `astronomy-easy-1` | 31 | 0.00 | 0.00 |
| `astronomy-easy-5` | 2 | 1.00 | 1.00 |
| `astronomy-hard-10` | 3 | 0.00 | 0.00 |
| `astronomy-hard-12` | 5 | 0.00 | 0.00 |
| `astronomy-hard-8` | 4 | 1.00 | 1.00 |
| `astronomy-hard-9` | 3 | 0.00 | 1.00 |
| `biomedical-hard-5` | 1 | 1.00 | 1.00 |
| `biomedical-hard-7` | 2 | 0.00 | 1.00 |
| `environment-easy-4` | 5 | 1.00 | 1.00 |
| `environment-easy-5` | 1 | 1.00 | 1.00 |
| `environment-hard-12` | 1 | 1.00 | 1.00 |
| `environment-hard-17` | 12 | 0.00 | 0.00 |
| `environment-hard-8` | 9 | 0.00 | 0.00 |
| `legal-easy-19` | 1 | 0.00 | 1.00 |
| `legal-easy-25` | 1 | 1.00 | 1.00 |
| `legal-easy-26` | 1 | 0.50 | 1.00 |
| `legal-easy-3` | 1 | 1.00 | 1.00 |
| `legal-easy-5` | 1 | 1.00 | 1.00 |
| `legal-hard-14` | 1 | 1.00 | 1.00 |
| `legal-hard-15` | 3 | 1.00 | 1.00 |
| `legal-hard-17` | 2 | 1.00 | 1.00 |
| `legal-hard-22` | 2 | 0.00 | 0.00 |
| `legal-hard-28` | 6 | 1.00 | 1.00 |
| `legal-hard-6` | 1 | 1.00 | 1.00 |
| `legal-hard-7` | 1 | 1.00 | 1.00 |
| `legal-hard-8` | 3 | 1.00 | 1.00 |
| `wildfire-easy-1` | 1 | 1.00 | 1.00 |
| `wildfire-hard-17` | 2 | 0.63 | 0.76 |
| `wildfire-hard-5` | 2 | 1.00 | 1.00 |

## Paired cache-aware usage

### lean vs control (103 paired tasks)

| Measure | control | lean | Δ |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $5.0351 | $5.6482 | $+0.6131 (+12.18%) |
| input_tokens | 6,997,884 | 7,014,864 | +16,980 (+0.24%) |
| cached_tokens | 6,079,104 | 6,142,464 | +63,360 (+1.04%) |
| output_tokens | 168,813 | 217,614 | +48,801 (+28.91%) |
| num_steps | 774 | 905 | +131 (+16.93%) |
| Uncached input | 918,780 | 872,400 | -46,380 (-5.05%) |

### pull vs lean (103 paired tasks)

| Measure | lean | pull | Δ |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $5.6482 | $5.5772 | $-0.0710 (-1.26%) |
| input_tokens | 7,014,864 | 6,990,425 | -24,439 (-0.35%) |
| cached_tokens | 6,142,464 | 5,888,000 | -254,464 (-4.14%) |
| output_tokens | 217,614 | 186,972 | -30,642 (-14.08%) |
| num_steps | 905 | 855 | -50 (-5.52%) |
| Uncached input | 872,400 | 1,102,425 | +230,025 (+26.37%) |

### pull vs control (103 paired tasks)

| Measure | control | pull | Δ |
| --- | ---: | ---: | ---: |
| Cache-aware cost | $5.0351 | $5.5772 | $+0.5422 (+10.77%) |
| input_tokens | 6,997,884 | 6,990,425 | -7,459 (-0.11%) |
| cached_tokens | 6,079,104 | 5,888,000 | -191,104 (-3.14%) |
| output_tokens | 168,813 | 186,972 | +18,159 (+10.76%) |
| num_steps | 774 | 855 | +81 (+10.47%) |
| Uncached input | 918,780 | 1,102,425 | +183,645 (+19.99%) |

## Verdict

**The dose-response is now measured.** Cutting the default render to 3 rows
starves the agent: −10.6 accuracy points AND +12.2% cost — the starvation tax
is paid in trajectory length (+131 steps, +29% output tokens), not saved
tokens. "Under-render thrashes" is now a benchmark-scale number, not a trace
anecdote.

**The pull mechanism works as designed.** 33/104 tasks pulled (129 pulls) with
sophisticated usage — column projections (46), distinct-value slices (38),
stats blocks (16). Pull recovers 3.9 of the 10.6 lost points (~37%) and
reduces thrash vs lean (−50 steps, −14% output). The poster case is
`biomedical-hard-7` (the README-names-the-data-sheet task): lean fails it,
the agent pulls the hidden sheet content back and passes. `astronomy-hard-9`,
`legal-easy-19`, `legal-easy-26` similarly recover.

**But the operating point loses.** rows=3 is too far below the information
water line: pull recovers only partially, costs +10.8% vs the full-push
control, and the cache pays for every pull (uncached +26% vs lean — tool
results append per turn and the tail section grows). At KramaBench trajectory
lengths, full-push Latest-3k remains the Pareto point.

**What survives for the paper:** (1) the push-axis dose-response curve
(full → rows=3 quantified at −10.6 acc / +12% cost); (2) demonstrated
competent demand-paging by the agent (arg-usage histogram + named recovery
cases); (3) the design lesson that pull is a SAFETY VALVE on a well-fed
default, not a substitute for one. The natural follow-up (E1b) is pull on top
of a mildly-lean or full default; the natural home for the pull story is
LakeQA Phase B, where search/download IS the pull loop and no push default
can contain a 9.5TB lake.
