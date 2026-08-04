# gpt-5.6 luna / terra — Anchor to C5 (n=5, official KramaBench scores)

Post-repair (infra-zero runs re-run; each arm retains 1-5 unrepaired `wildfire-hard-19` zeros, <=0.4 pt).
Cost = true per-model pricing; all ± are population std ACROSS THE 5 REPS (per-rep means).

## gpt-5.6-luna ($0.20 / $0.02 / $1.20 per M tok)

| arm | config | acc Easy | acc Hard | acc All | $ Easy | $ Hard | $ All |
|---|---|---|---|---|---|---|---|
| Anchor | 1K, DELTA, wo stats | 80.7±3.4 | 62.0±1.8 | 69.6±1.9 | 0.0049±0.0002 | 0.0099±0.0008 | 0.0078±0.0005 |
| C1 | 5K, DELTA, wo stats | 81.7±3.9 | 63.8±4.4 | 71.0±4.0 | 0.0058±0.0005 | 0.0120±0.0007 | 0.0094±0.0005 |
| C2 | 1K, DELTA, w stats | 83.2±2.8 | 62.3±2.1 | 70.7±0.9 | 0.0056±0.0005 | 0.0115±0.0008 | 0.0091±0.0005 |
| C3 | 1K, LATEST, wo stats, +code | 79.8±3.0 | 63.8±2.8 | 70.3±1.5 | 0.0043±0.0003 | 0.0092±0.0006 | 0.0072±0.0003 |
| C4 | 5K, LATEST, w stats, +code | 78.4±0.0 | 63.7±4.3 | 69.6±2.5 | 0.0059±0.0008 | 0.0114±0.0005 | 0.0091±0.0005 |
| C5 | 5K, DELTA, w stats | 83.1±2.2 | 66.9±0.7 | 73.4±0.8 | 0.0063±0.0002 | 0.0122±0.0007 | 0.0098±0.0004 |

## gpt-5.6-terra ($2.00 / $0.20 / $12.00 per M tok)

| arm | config | acc Easy | acc Hard | acc All | $ Easy | $ Hard | $ All |
|---|---|---|---|---|---|---|---|
| Anchor | 1K, DELTA, wo stats | 84.8±1.8 | 66.8±2.1 | 74.1±1.5 | 0.0465±0.0044 | 0.0863±0.0074 | 0.0701±0.0059 |
| C1 | 5K, DELTA, wo stats | 82.2±1.9 | 68.2±4.4 | 73.8±3.0 | 0.0551±0.0036 | 0.1091±0.0077 | 0.0872±0.0057 |
| C2 | 1K, DELTA, w stats | 83.6±1.8 | 66.6±2.5 | 73.5±0.9 | 0.0539±0.0048 | 0.1016±0.0048 | 0.0822±0.0021 |
| C3 | 1K, LATEST, wo stats, +code | 83.0±0.3 | 66.6±3.1 | 73.2±1.8 | 0.0407±0.0006 | 0.0745±0.0050 | 0.0606±0.0029 |
| C4 | 5K, LATEST, w stats, +code | 83.7±3.6 | 70.8±1.7 | 76.0±2.4 | 0.0520±0.0037 | 0.1010±0.0045 | 0.0811±0.0021 |
| C5 | 5K, DELTA, w stats | 85.8±2.8 | 67.5±1.2 | 74.9±0.9 | 0.0573±0.0021 | 0.1103±0.0048 | 0.0887±0.0030 |

