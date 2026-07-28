#!/usr/bin/env python3
"""
M9 (decomposition / grounding / waste) per task-division, for the 7 gpt-5-mini
dataflow configs (Rep0/1/2 averaged, react-grain M9). Three divisions, each from
task-intrinsic properties (non-circular):

  difficulty : easy | hard        (task id)
  file_size  : small | mid | large tertiles  (raw bytes of gold data_sources)
  gold_steps : few | some | many  tertiles    (# gold subtasks)

For each division bucket we print every config's M9 triple, so knob separation
WITHIN a category is visible. Reads existing judge_m9react.json caches — no runs.

Run: .venv/bin/python scripts/m9_by_division.py
"""
import json, glob, os, statistics as st
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
REPS = ["Replicate0", "Replicate1", "Replicate2"]
CONFIGS = [("anchor", "Delta1kSchemaOnly"), ("C1 rows5k", "Delta5kSchemaOnly"),
           ("C2 stats1k", "DeltaStats1kD2"), ("C3 lat+code", "Latest1kCodeInSnap"),
           ("C4 stats5k", "DeltaStats5kD2"), ("C5 stats2k", "DeltaStats2kD2"),
           ("C6 lat+stat", "LatestStats1kD2")]
FIELDS = [("decomp", "decomposition"), ("ground", "grounding"), ("waste", "waste_frac")]


def load_workload():
    W = {}
    for f in glob.glob(str(KB / "workload/*.json")):
        if any(x in f for x in ("tiny", "quick", "lakeqa")):
            continue
        for t in json.load(open(f)):
            if t.get("id"):
                W[t["id"]] = t
    return W


def filebytes(task, dom):
    tot = 0
    for ds in task.get("data_sources", []) or []:
        for p in set(glob.glob(str(KB / f"data/{dom}/input/{ds}")) +
                     glob.glob(str(KB / f"data/{dom}/input/**/{os.path.basename(ds)}"), recursive=True)):
            if os.path.isfile(p):
                tot += os.path.getsize(p)
    return tot


def m9_by_task(suf, field):
    per = {}
    for r in REPS:
        arm = f"DataflowSystemGPT5Mini{suf}{r}"
        for p in glob.glob(str(KB / "system_scratch" / arm / "*" / "judge_m9react.json")):
            v = json.load(open(p)).get(field)
            if isinstance(v, (int, float)):
                per.setdefault(Path(p).parent.name, []).append(v)
    return {t: st.mean(v) for t, v in per.items()}


def tertile_split(vals):
    """dict task->value -> {task: 'small'|'mid'|'large'} by tertile."""
    srt = sorted(vals.items(), key=lambda x: x[1])
    n = len(srt)
    lab = {}
    for i, (t, _) in enumerate(srt):
        lab[t] = "lo" if i < n // 3 else ("mid" if i < 2 * n // 3 else "hi")
    return lab


def main():
    W = load_workload()
    # precompute per-config per-field task scores
    M = {suf: {f: m9_by_task(suf, fld) for f, fld in FIELDS} for _, suf in CONFIGS}

    fsize = {t: filebytes(td, t.rsplit("-", 2)[0]) for t, td in W.items()}
    fsize = {t: b for t, b in fsize.items() if b}
    gsteps = {t: len(W[t].get("subtasks", []) or []) for t in W}

    divisions = {
        "difficulty": {t: ("hard" if "hard" in t else "easy") for t in W},
        "file_size": tertile_split(fsize),
        "gold_steps": tertile_split(gsteps),
    }
    bucket_order = {"difficulty": ["easy", "hard"],
                    "file_size": ["lo", "mid", "hi"],
                    "gold_steps": ["lo", "mid", "hi"]}
    bucket_name = {"lo": "small/few", "mid": "mid", "hi": "large/many", "easy": "easy", "hard": "hard"}

    for dim, assign in divisions.items():
        print(f"\n{'='*78}\nDIVISION: {dim}\n{'='*78}")
        for b in bucket_order[dim]:
            tasks = [t for t, lab in assign.items() if lab == b]
            n0 = len(tasks)
            print(f"\n  [{bucket_name[b]}]  ({n0} tasks)")
            print(f"    {'config':12s} {'decomp':>7s} {'ground':>7s} {'waste':>7s}  {'n':>3s}")
            for lab, suf in CONFIGS:
                row = f"    {lab:12s}"
                nn = 0
                for f, _ in FIELDS:
                    sc = M[suf][f]
                    vs = [sc[t] for t in tasks if t in sc]
                    nn = max(nn, len(vs))
                    row += f" {st.mean(vs):7.3f}" if vs else f" {'—':>7s}"
                print(row + f"  {nn:3d}")


if __name__ == "__main__":
    main()
