#!/usr/bin/env python3
"""
Conditional-cut analysis — does a knob's metric gap widen in the task category
where its mechanism can act, and go flat in the control category?

Categories are derived from GOLD/TASK properties only (never from which arm won,
so the split is non-circular):
  table_size    : max (rows * cols) of any gold-subtask table the pipeline
                  touches, inferred from the reference solution's rendered
                  intermediates — proxy: # subtasks whose answer is a list, and
                  the task's data_sources count. We use a robust proxy below.
  pipeline_len  : # gold subtasks (proxy for operator count / history depth)
  needs_distrib : any gold subtask step mentions dedup/distribution/threshold/
                  outlier/duplicate/unique  (where a stats line would inform)
  difficulty    : 'easy' / 'hard' from the task id

Each knob is scored on (trigger subset) vs (control subset); we print the metric
means for the anchor and ray arm in each bucket, and the gap. A knob is
"mechanistically confirmed" when gap(trigger) >> gap(control).

Run: .venv/bin/python scripts/conditional_cut.py
"""
import json, glob, re, statistics as st, os
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
REPS = ["Replicate0", "Replicate1", "Replicate2"]


def load_workload():
    W = {}
    for f in glob.glob(str(KB / "workload/*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")):
            continue
        for t in json.load(open(f)):
            if t.get("id"):
                W[t["id"]] = t
    return W


DIST_RE = re.compile(r"\b(dedup|duplicat|distinct|unique|distribution|threshold|"
                     r"outlier|most common|frequency|value_counts|top[\s-]?\d|median|quantile)\b", re.I)


def big_table_proxy(task):
    """Does the reference solution touch a large table (where a 1k char cap would
    truncate)? Proxy from gold: a subtask whose answer is a LIST with many items,
    OR >=3 data_sources (multi-file merge -> wide intermediate), OR a subtask step
    mentioning 'all rows'/'each row'. Deterministic from gold only."""
    for s in task.get("subtasks", []):
        a = s.get("answer")
        if isinstance(a, list) and len(a) >= 6:
            return True
    if len(task.get("data_sources", []) or []) >= 3:
        return True
    for s in task.get("subtasks", []):
        if re.search(r"\ball rows\b|\beach row\b|\bevery (row|record)\b|\bfull (table|dataset)\b",
                     s.get("step", ""), re.I):
            return True
    return False


def categorize(task):
    subs = task.get("subtasks", []) or []
    return dict(
        big_table=big_table_proxy(task),
        long_pipeline=len(subs) >= 5,
        needs_distrib=any(DIST_RE.search(s.get("step", "") or "") for s in subs),
        hard="hard" in task["id"],
    )


def metric_by_task(suf, fname, field):
    """task_id -> mean(field) over the 3 replicate arms of this config."""
    per = {}
    for r in REPS:
        arm = f"DataflowSystemGPT5Mini{suf}{r}"
        for p in glob.glob(str(KB / "system_scratch" / arm / "*" / fname)):
            d = json.load(open(p))
            v = d.get(field)
            if isinstance(v, (int, float)):
                per.setdefault(Path(p).parent.name, []).append(v)
    return {t: st.mean(v) for t, v in per.items()}


def gap_by_bucket(anchor_suf, ray_suf, fname, field, cat_key, W, cats):
    A = metric_by_task(anchor_suf, fname, field)
    R = metric_by_task(ray_suf, fname, field)
    common = [t for t in A if t in R and t in cats]
    buckets = {}
    for want in (True, False):
        ts = [t for t in common if cats[t][cat_key] == want]
        if not ts:
            buckets[want] = None
            continue
        a = st.mean(A[t] for t in ts); r = st.mean(R[t] for t in ts)
        buckets[want] = (len(ts), a, r, r - a)
    return buckets


KNOBS = [
    # label, anchor_suf, ray_suf, category, metric(fname, field), higher_is_better
    ("rows 1k->5k  [M5]",   "Delta1kSchemaOnly", "Delta5kSchemaOnly", "big_table",
     "judge_m5m6.json", "m5", True),
    ("rows 1k->5k  [waste]", "Delta1kSchemaOnly", "Delta5kSchemaOnly", "big_table",
     "judge_m9react.json", "waste_frac", False),
    ("history D->L [waste]", "DeltaStats1kD2", "LatestStats1kD2", "long_pipeline",
     "judge_m9react.json", "waste_frac", False),
    ("history D->L [M5]",    "DeltaStats1kD2", "LatestStats1kD2", "long_pipeline",
     "judge_m5m6.json", "m5", True),
    ("stats off->on [M5]",  "Delta1kSchemaOnly", "DeltaStats1kD2", "needs_distrib",
     "judge_m5m6.json", "m5", True),
    ("stats off->on [ground]", "Delta1kSchemaOnly", "DeltaStats1kD2", "needs_distrib",
     "judge_m9react.json", "grounding", True),
]


def main():
    W = load_workload()
    cats = {t: categorize(td) for t, td in W.items()}
    # category sizes
    print("Category sizes (of 104 tasks):")
    for k in ("big_table", "long_pipeline", "needs_distrib", "hard"):
        n = sum(1 for c in cats.values() if c[k])
        print(f"  {k:14s}: {n} trigger / {len(cats)-n} control")
    print()
    print(f"{'knob [metric]':22s} {'bucket':9s} {'n':>3s} {'anchor':>7s} {'ray':>7s} {'gap':>7s}")
    print("-" * 62)
    for lab, asuf, rsuf, cat, fname, field, hib in KNOBS:
        b = gap_by_bucket(asuf, rsuf, fname, field, cat, W, cats)
        for want, name in ((True, "TRIGGER"), (False, "control")):
            if not b[want]:
                continue
            n, a, r, g = b[want]
            star = ""
            if want:  # flag when trigger gap is in the beneficial direction and >2x control
                cg = b[False][3] if b[False] else 0
                good = (g > 0) if hib else (g < 0)
                if good and abs(g) > 2 * abs(cg) + 0.02:
                    star = "  <== widened"
            print(f"{lab if want else '':22s} {name:9s} {n:3d} {a:7.3f} {r:7.3f} {g:+7.3f}{star}")
        print()


if __name__ == "__main__":
    main()
