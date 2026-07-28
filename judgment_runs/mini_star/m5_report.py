"""M5 (task-completion) per-arm report: mean +- std across replicates.

M5 = fraction of a task's ground-truth steps that the agent's final dataflow
performs (scripts/judge_m5.py, version 1: computed / absent).

  mean = per-task M5 averaged over tasks, then averaged over the arm's reps
  std  = population std across the per-rep means (the replicate spread)
Tasks are the intersection across an arm's reps so every rep is scored on the
same set.
"""
import json
import os
import statistics
import sys

KB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = [
    ("anchor  Delta 1k  schema", "DataflowSystemGPT5MiniDelta1kSchemaOnly"),
    ("C1      Delta 5k  schema", "DataflowSystemGPT5MiniDelta5kSchemaOnly"),
    ("C2      Delta 1k  stats",  "DataflowSystemGPT5MiniDeltaStats1kD2"),
    ("C3      Latest1k  code",   "DataflowSystemGPT5MiniLatest1kCodeInSnap"),
    ("C4      Delta 5k  stats",  "DataflowSystemGPT5MiniDeltaStats5kD2"),
]


def rep_scores(arm):
    """{task: m5} for one replicate arm."""
    base = os.path.join(KB, "system_scratch", arm)
    out = {}
    if not os.path.isdir(base):
        return out
    for t in sorted(os.listdir(base)):
        p = os.path.join(base, t, "judge_m5.json")
        if not os.path.exists(p):
            continue
        try:
            j = json.load(open(p))
        except Exception:
            continue
        if j.get("version") == 1 and isinstance(j.get("m5"), (int, float)):
            out[t] = j["m5"]
    return out


def mean(xs):
    return sum(xs) / len(xs)


rows = []
for label, base in CONFIGS:
    per = [rep_scores(f"{base}Replicate{i}") for i in range(5)]
    per = [p for p in per if p]
    if not per:
        continue
    tasks = sorted(set.intersection(*[set(p) for p in per]))
    rep_means = [mean([p[t] for t in tasks]) for p in per]
    # step-pooled view: every step weighted equally rather than every task
    rows.append(dict(label=label, reps=len(per), n=len(tasks),
                     mean=mean(rep_means), std=statistics.pstdev(rep_means),
                     lo=min(rep_means), hi=max(rep_means), rep_means=rep_means))

print(f"{'config':30s} {'reps':>4s} {'n':>4s}   {'M5 (mean ± std)':>18s}   {'range':>15s}")
print("-" * 82)
for r in rows:
    print(f"{r['label']:30s} {r['reps']:4d} {r['n']:4d}   "
          f"{r['mean']:.3f} ± {r['std']:.3f}{'':>6s}   {r['lo']:.3f}–{r['hi']:.3f}")

json.dump(rows, open(os.path.join(KB, "judgment_runs/mini_star/m5_table.json"), "w"), indent=1)
