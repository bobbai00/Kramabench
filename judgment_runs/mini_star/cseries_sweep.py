#!/usr/bin/env python3
"""Full C-series sweep: accuracy + cost with std, all reps, full 104 tasks.

Accuracy = KramaBench's own metric values (native measures CSVs) reduced with
compute_scores.py's formula — validated 12/12 against `compute_scores.py --sut`.
A rep is INCLUDED only if all 6 workloads have a FULL measures CSV; partial
--task_id CSVs and abandoned reps are excluded and reported, never averaged in.
"""
import glob, json, os, statistics as st
import pandas as pd

KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
SM = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]
WL_N = {"archeology": 12, "astronomy": 12, "biomedical": 9, "environment": 20,
        "legal": 30, "wildfire": 21}
PIN, PCACHE, POUT = 0.25e-6, 0.025e-6, 2e-6

# label -> (arm base, reps to try, knob description)
ARMS = [
    ("anchor", "Delta1kSchemaOnly",     range(8), "DELTA 1k schema-only"),
    ("C1",     "Delta5kSchemaOnly",     range(8), "DELTA 5k schema-only        (sampling 1k->5k)"),
    ("C7",     "Delta2kSchemaOnly",     range(5), "DELTA 2k schema-only        (sampling 1k->2k)"),
    ("C2",     "DeltaStats1kD2",        range(8), "DELTA 1k + stats/D2         (stats ON)"),
    ("C5",     "DeltaStats2kD2",        range(5), "DELTA 2k + stats/D2"),
    ("C4",     "DeltaStats5kD2",        range(5), "DELTA 5k + stats/D2"),
    ("C3",     "Latest1kCodeInSnap",    range(5), "LATEST 1k + code            (delta->latest)"),
    ("C8",     "Latest5kCodeInSnap",    range(5), "LATEST 5k + code"),
    ("C6",     "LatestStats1kD2",       range(5), "LATEST 1k + stats/D2"),
    ("C8s",    "LatestStats5kD2Code",   range(5), "LATEST 5k + code + stats/D2"),
    ("C9",     "C9SourceRichLatest",    range(1, 4), "LATEST src 5k+stats / down 1k no-stats"),
    ("C10",    "C10SourceRichDelta",    range(1, 4), "DELTA  src 5k+stats / down 1k no-stats"),
    ("C11",    "C11UniformRichLatest",  range(1, 4), "LATEST 5k + stats ALL ops"),
    ("C12",    "C12LatestStats1kCode",  range(1, 4), "LATEST 1k + code + stats/D2  (missing cell)"),
    ("A7",     "A7FileIOFact",          range(1, 4), "= C11 + engine `files read:` fact"),
    # --- NEW ENGINE ERA (post-restart 2026-07-29 ~13:2x). Paired arms; do NOT
    # compare these against the rows above, only against each other.
    ("D8",     "D8Latest5kCode",        range(1, 4), "[era2] LATEST 5k + code            (control)"),
    ("D8F",    "D8FileIO",              range(1, 4), "[era2] LATEST 5k + code + files-read"),
    ("D12",    "D12LatestStats1kCode",  range(1, 4), "[era2] LATEST 1k + code + stats    (control)"),
    ("D12F",   "D12FileIO",             range(1, 4), "[era2] LATEST 1k + code + stats + files-read"),
    ("N1",     "N1Latest5kStats",       range(1, 4), "[era2] LATEST 5k + code + stats (+fact)"),
    ("N2",     "N2Delta5kStats",        range(1, 4), "[era2] DELTA  5k + stats        (+fact)"),
    ("N3",     "N3SrcRich5k2k",         range(1, 4), "[era2] LATEST src 5k / down 2k, stats both (+fact)"),
]


def full_rows(sut):
    dfs = []
    for wl, n in WL_N.items():
        for f in sorted(glob.glob(f"{KB}/results/{sut}/{wl}_measures_*.csv"), reverse=True):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if d[d["metric"].isin(SM)]["task_id"].nunique() >= n:
                dfs.append(d)
                break
    if len(dfs) < len(WL_N):
        return None  # incomplete: refuse to score
    d = pd.concat(dfs, ignore_index=True)
    return d[d["metric"].isin(SM)]


def rep_stats(sut):
    d = full_rows(sut)
    if d is None:
        return None
    easy = d[d["task_id"].str.contains("-easy-")]
    hard = d[d["task_id"].str.contains("-hard-")]
    tin = ca = out = steps = 0.0
    n = 0
    for f in glob.glob(f"{KB}/system_scratch/{sut}/*/stats.json"):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        tin += s.get("input_tokens", 0); ca += s.get("cached_tokens", 0)
        out += s.get("output_tokens", 0); steps += s.get("num_steps", 0); n += 1
    cost = ((tin - ca) * PIN + ca * PCACHE + out * POUT) / n if n else None
    return dict(acc=d["value"].sum() / len(d) * 100,
                easy=easy["value"].sum() / len(easy) * 100 if len(easy) else float("nan"),
                hard=hard["value"].sum() / len(hard) * 100 if len(hard) else float("nan"),
                cost=cost, steps=steps / n if n else 0)


print("C-SERIES SWEEP — full 104 tasks. Accuracy = KramaBench's own score.")
print("A rep counts only if all 6 workloads have a FULL measures CSV.\n")
print(f"{'':<5}{'knobs':<40}{'reps':>5}{'accuracy':>15}{'easy':>13}{'hard':>13}{'$/task':>16}{'steps':>7}")
print("-" * 116)
out_rows = {}
skipped = []
for label, base, reps, desc in ARMS:
    accs, easys, hards, costs, steps = [], [], [], [], []
    for r in reps:
        sut = f"DataflowSystemGPT5Mini{base}Replicate{r}"
        if not os.path.isdir(f"{KB}/system_scratch/{sut}"):
            continue
        rs = rep_stats(sut)
        if rs is None:
            n_ans = len(glob.glob(f"{KB}/system_scratch/{sut}/*/response.txt"))
            skipped.append(f"{label} rep{r} ({base}): incomplete — {n_ans}/104 answers, no full CSVs")
            continue
        accs.append(rs["acc"]); easys.append(rs["easy"]); hards.append(rs["hard"])
        if rs["cost"]:
            costs.append(rs["cost"])
        steps.append(rs["steps"])
    if not accs:
        continue
    sd = lambda v: st.pstdev(v) if len(v) > 1 else 0.0
    out_rows[label] = dict(desc=desc, n=len(accs), acc=st.mean(accs), accsd=sd(accs),
                           easy=st.mean(easys), easysd=sd(easys), hard=st.mean(hards), hardsd=sd(hards),
                           cost=st.mean(costs) if costs else float("nan"), costsd=sd(costs) if costs else 0,
                           steps=st.mean(steps), accs=accs)
    r = out_rows[label]
    print(f"{label:<5}{desc:<40}{r['n']:>5}{r['acc']:>9.1f} ±{r['accsd']:4.1f}"
          f"{r['easy']:>8.1f} ±{r['easysd']:3.1f}{r['hard']:>8.1f} ±{r['hardsd']:3.1f}"
          f"{r['cost']:>11.4f} ±{r['costsd']:.4f}{r['steps']:>7.1f}")

anc = out_rows.get("anchor")
if anc:
    print(f"\n{'':<5}{'':<40}{'':>5}{'Δacc vs anchor':>18}{'Δcost':>10}{'accuracy per rep':>34}")
    print("-" * 116)
    for label, r in out_rows.items():
        if label == "anchor":
            print(f"{label:<5}{r['desc']:<40}{r['n']:>5}{'—':>18}{'—':>10}   {['%.1f' % a for a in r['accs']]}")
        else:
            print(f"{label:<5}{r['desc']:<40}{r['n']:>5}{r['acc']-anc['acc']:>+13.1f} pt"
                  f"{(r['cost']/anc['cost']-1)*100:>+9.1f}%   {['%.1f' % a for a in r['accs']]}")

if skipped:
    print("\nEXCLUDED (not averaged in):")
    for s in skipped:
        print(f"  {s}")
