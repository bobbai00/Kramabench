#!/usr/bin/env python3
"""Combine round0/1/2 snapshots -> per-(arm,task) observation sequence, classify
flaky vs stable, and produce the recovery-equalized (best-of-rounds) matrix."""
import json, os
D = os.path.dirname(__file__)
ARMS = ["anchor", "C1_5k", "C2_stats", "C3_latest_code"]
TASKS = [l.strip() for l in open(os.path.join(D, "tasks10.txt")) if l.strip()]
PASS = 0.9
snaps = {r: json.load(open(os.path.join(D, f"snap_round{r}.json"))) for r in (0, 1, 2)}

def seq(arm, t):
    # round0 always present; rounds 1/2 present only for tasks that were rerun
    # (initially-failed). If absent in a round, the task passed r0 -> carry r0.
    out = []
    for r in (0, 1, 2):
        v = snaps[r].get(arm, {}).get(t)
        out.append(v)
    return out

def pf(v):
    return "?" if v is None else ("P" if v >= PASS else ".")

print("=== per-arm observation sequences (round0/1/2) ; * = flaky (mixed P/.) ===")
flaky = {a: [] for a in ARMS}
best = {a: {} for a in ARMS}
for a in ARMS:
    print(f"\n[{a}]")
    for t in TASKS:
        s = seq(a, t)
        obs = [v for v in s if v is not None]
        marks = "".join(pf(v) for v in s)
        passes = sum(1 for v in obs if v >= PASS)
        best[a][t] = max((v for v in obs), default=0.0)
        is_flaky = 0 < passes < len(obs)  # mixed outcomes across observed rounds
        if is_flaky:
            flaky[a].append(t)
        print(f"  {t:<22} {marks:<3} obs={['%.2f'%v for v in obs]} {'FLAKY' if is_flaky else ''}")

print("\n=== recovery-equalized matrix (best of observed rounds) ===")
print(f"{'task':<22}" + "".join(f"{a:>16}" for a in ARMS))
passc = {a: 0 for a in ARMS}
for t in TASKS:
    row = ""
    for a in ARMS:
        v = best[a][t]
        if v >= PASS:
            passc[a] += 1
        row += f"{v:>13.2f}{pf(v):>3}"
    print(f"{t:<22}{row}")
print(f"{'PASS/10':<22}" + "".join(f"{passc[a]:>15}P" for a in ARMS))

print("\n=== flaky (transient) tasks per arm ===")
allflaky = set()
for a in ARMS:
    print(f"  {a}: {flaky[a]}")
    allflaky.update(flaky[a])
print(f"  UNION flaky: {sorted(allflaky)}")

# stable-fail: failed in all observed rounds for that arm
print("\n=== stable-fail (0 in every observed round) per arm ===")
for a in ARMS:
    sf = [t for t in TASKS if all((v is None or v < PASS) for v in seq(a, t)) and best[a][t] < PASS]
    print(f"  {a}: {sf}")

json.dump({"flaky": flaky, "best": best, "passc": passc}, open(os.path.join(D, "flaky_result.json"), "w"), indent=1)
