"""Final headline table: accuracy mean±std, floor, flipped%, cost mean±std.

Conventions (Bob's, 2026-07-27):
  score      = KramaBench-native answer-type metric, continuous (kb.answer_scores)
  mean       = per-task mean across reps, then mean over tasks
  std        = population std across the per-rep aggregate means
  floor      = mean over tasks of MIN score across reps
  flipped%   = fraction of tasks with ANY score difference across reps
  cost       = per rep: 5% two-sided trimmed mean of that rep's per-task cost_usd;
               then mean (and std) across reps.  Trim per rep, then average.
Tasks are the intersection across an arm's reps.
"""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import kb

ROOT = kb.KB_ROOT / "system_scratch"

CONFIGS = [
    ("anchor  Delta 1k  schema",      "DataflowSystemGPT5MiniDelta1kSchemaOnly",  "df"),
    ("C1      Delta 5k  schema",      "DataflowSystemGPT5MiniDelta5kSchemaOnly",  "df"),
    ("C2      Delta 1k  stats",       "DataflowSystemGPT5MiniDeltaStats1kD2",     "df"),
    ("C3      Latest1k  code",        "DataflowSystemGPT5MiniLatest1kCodeInSnap", "df"),
    ("C4      Delta 5k  stats",       "DataflowSystemGPT5MiniDeltaStats5kD2",     "df"),
    ("C5      Delta 2k  stats",       "DataflowSystemGPT5MiniDeltaStats2kD2",     "df"),
    ("C6      Latest1k  stats",       "DataflowSystemGPT5MiniLatestStats1kD2",    "df"),
    ("C7      Delta 2k  schema",      "DataflowSystemGPT5MiniDelta2kSchemaOnly",  "df"),
    ("CA-1k   code agent guided",     "CodeAgentSystemGpt5MiniProxyChars1kGuided", "ca"),
    ("CA-5k   code agent guided",     "CodeAgentSystemGpt5MiniProxyChars5kGuided", "ca"),
]


def arms_of(base, kind):
    if kind == "df":
        names = [f"{base}Replicate{i}" for i in range(5)]
    else:  # code-agent convention: base is replicate 0
        names = [base] + [f"{base}Replicate{i}" for i in (1, 2, 3, 4)]
    return [a for a in names if (ROOT / a).is_dir()]


def mean(xs):
    return sum(xs) / len(xs)


def trimmed(xs, pct=0.05):
    """5% two-sided trimmed mean."""
    s = sorted(xs)
    k = int(round(len(s) * pct))
    keep = s[k:len(s) - k] or s
    return mean(keep)


def row(label, base, kind):
    arms = arms_of(base, kind)
    per = [kb.answer_scores(a) for a in arms]
    tasks = sorted(set.intersection(*[set(p) for p in per]))
    detail = {t: [p[t] for p in per] for t in tasks}
    n = len(tasks)

    acc_mean = mean([mean(v) for v in detail.values()])
    rep_means = [mean([p[t] for t in tasks]) for p in per]
    acc_std = statistics.pstdev(rep_means)
    floor = mean([min(v) for v in detail.values()])
    flipped = sum(1 for v in detail.values() if len(set(v)) > 1) / n

    costs = [{r["task_id"]: r["cost"] for r in kb.load_cost_stats(a)} for a in arms]
    rep_cost = [trimmed([c[t] for t in tasks if t in c]) for c in costs]
    return dict(label=label, reps=len(arms), n=n, acc=acc_mean, acc_sd=acc_std,
                floor=floor, flipped=flipped,
                cost=mean(rep_cost), cost_sd=statistics.pstdev(rep_cost))


rows = [row(*c) for c in CONFIGS]

# Scores reported on the 0-100 scale (KramaBench leaderboard convention).
print(f"{'config':30s} {'reps':>4s} {'n':>4s}  {'score (mean ± std)':>19s}  {'floor':>6s}  "
      f"{'flipped':>8s}  {'$/task (mean ± std)':>22s}")
print("-" * 104)
for r in rows:
    print(f"{r['label']:30s} {r['reps']:4d} {r['n']:4d}  "
          f"{100*r['acc']:5.1f} ± {100*r['acc_sd']:4.1f}{'':>6s}  {100*r['floor']:6.1f}  {r['flipped']:7.1%}  "
          f"{r['cost']:.4f} ± {r['cost_sd']:.4f}{'':>5s}")

json.dump(rows, open(kb.KB_ROOT / "judgment_runs/mini_star/final_table.json", "w"), indent=1)
