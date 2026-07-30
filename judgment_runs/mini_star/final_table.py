#!/usr/bin/env python3
"""Master table: arm, config, accuracy(all/easy/hard) ±std, cost(all/easy/hard) ±std.

Accuracy = KramaBench's own metric values from the per-SUT measures CSVs, reduced
with compute_scores.py's formula (verified to match `compute_scores.py --sut`
exactly on every arm). Per-SUT CSVs are read deliberately: the shared
results/aggregated_results.csv that compute_scores.py reads is a concurrency
hazard and was corrupted by overlapping `kb.py reeval` writes.
"""
import glob, json, os, statistics as st
import pandas as pd

KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
SM = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]
WL_N = {"archeology": 12, "astronomy": 12, "biomedical": 9, "environment": 20, "legal": 30, "wildfire": 21}
PIN, PCACHE, POUT = 0.25e-6, 0.025e-6, 2e-6

# (label, arm base, reps, sampling, stats, mode, era)
ARMS = [
    ("anchor", "Delta1kSchemaOnly",      range(8),    "1K",     "wo stats", "delta",  1),
    ("C1",     "Delta5kSchemaOnly",      range(8),    "5K",     "wo stats", "delta",  1),
    ("C7",     "Delta2kSchemaOnly",      range(5),    "2K",     "wo stats", "delta",  1),
    ("C2",     "DeltaStats1kD2",         range(5),    "1K",     "w stats",  "delta",  1),
    ("C5",     "DeltaStats2kD2",         range(5),    "2K",     "w stats",  "delta",  1),
    ("C4",     "DeltaStats5kD2",         range(5),    "5K",     "w stats",  "delta",  1),
    ("C6",     "LatestStats1kD2",        range(5),    "1K",     "w stats",  "latest", 1),
    ("C3",     "Latest1kCodeInSnap",     range(5),    "1K",     "wo stats", "latest", 1),
    ("C8",     "Latest5kCodeInSnap",     range(5),    "5K",     "wo stats", "latest", 1),
    ("C8s",    "LatestStats5kD2Code",    range(5),    "5K",     "w stats",  "latest", 1),
    ("C11",    "C11UniformRichLatest",   range(1, 4), "5K",     "w stats",  "latest", 1),
    ("C9",     "C9SourceRichLatest",     range(1, 4), "5K-1K",  "w stats",  "latest", 1),
    ("C10",    "C10SourceRichDelta",     range(1, 4), "5K-1K",  "w stats",  "delta",  1),
    ("A7",     "A7FileIOFact",           range(1, 4), "5K",     "w stats",  "latest", 1),
    ("D8",     "D8Latest5kCode",         range(1, 4), "5K",     "wo stats", "latest", 2),
    ("D8F",    "D8FileIO",               range(1, 4), "5K",     "wo stats", "latest", 2),
    ("D12",    "D12LatestStats1kCode",   range(1, 4), "1K",     "w stats",  "latest", 2),
    ("D12F",   "D12FileIO",              range(1, 4), "1K",     "w stats",  "latest", 2),
    ("N1",     "N1Latest5kStats",        range(1, 4), "5K",     "w stats",  "latest", 2),
    ("N2",     "N2Delta5kStats",         range(1, 4), "5K",     "w stats",  "delta",  2),
    ("N3",     "N3SrcRich5k2k",          range(1, 4), "5K-2K",  "w stats",  "latest", 2),
    ("N4",     "N4Latest2kStats",        range(1, 4), "2K",     "w stats",  "latest", 2),
    ("N5",     "N5SrcRich2k1k",          range(1, 4), "2K-1K",  "w stats",  "latest", 2),
    ("N6",     "N6Latest3kStats",        range(1, 4), "3K",     "w stats",  "latest", 2),
    # D8F reps 4-5 ran AFTER the `Files read:` layout move (now grouped with
    # `Inputs:` above `Code:`); reps 1-3 ran before it. Deliberately a separate
    # row — pooling across a render change would hide the layout effect, and
    # this split doubles as a free A/B on it.
    ("D8F'",   "D8FileIO",               range(4, 6), "5K",     "wo stats", "latest", 3),
    # LAYOUT A/B (2026-07-30): same engine, both arms interleaved task-major.
    # Settles the D8F reps-4-5 scare: -5.6 pt there was engine senescence, not the
    # `Files read:` reposition. Paired, the layout is +1.2 pt at 0.43x SE and -11% cost.
    ("LOld",   "LayoutOld",              range(1, 4), "5K",     "wo stats", "latest", 4),
    ("LNew",   "LayoutNew",              range(1, 4), "5K",     "wo stats", "latest", 4),
    # P-SERIES code budget (chained, same engine as the layout pool).
    ("P0",     "P0CodeControl",          range(1, 4), "5K",     "wo stats", "latest", 4),
    ("P1",     "P1Code800",              range(1, 4), "5K/c800","wo stats", "latest", 4),
    ("P2",     "P2Code400",              range(1, 4), "5K/c400","wo stats", "latest", 4),
]


def rep(sut):
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
    if len(dfs) < 6:
        return None
    d = pd.concat(dfs, ignore_index=True)
    d = d[d["metric"].isin(SM)]
    e, h = d[d.task_id.str.contains("-easy-")], d[d.task_id.str.contains("-hard-")]
    ce, ch = [], []
    for f in glob.glob(f"{KB}/system_scratch/{sut}/*/stats.json"):
        t = os.path.basename(os.path.dirname(f))
        try:
            s = json.load(open(f))
        except Exception:
            continue
        c = ((s.get("input_tokens", 0) - s.get("cached_tokens", 0)) * PIN
             + s.get("cached_tokens", 0) * PCACHE + s.get("output_tokens", 0) * POUT)
        (ce if "-easy-" in t else ch if "-hard-" in t else []).append(c)
    if not ce or not ch:
        return None
    n_all = len(ce) + len(ch)
    return dict(a=d["value"].sum() / len(d) * 100,
                e=e["value"].sum() / len(e) * 100, h=h["value"].sum() / len(h) * 100,
                ca=(sum(ce) + sum(ch)) / n_all, ce=st.mean(ce), ch=st.mean(ch))


sd = lambda v: st.pstdev(v) if len(v) > 1 else 0.0
print("arm    config                              r   acc all      acc easy     acc hard     $ all           $ easy          $ hard")
print("-" * 142)
for lab, base, reps, samp, stats, mode, era in ARMS:
    rs = [rep(f"DataflowSystemGPT5Mini{base}Replicate{r}") for r in reps]
    rs = [x for x in rs if x]
    if not rs:
        continue
    g = lambda k: [x[k] for x in rs]
    cfg = f"{samp}, {stats}, {mode}" + {1: "", 2: " [e2]", 3: " [e2, post-layout]", 4: " [e3 paired]"}[era]
    print(f"{lab:<7}{cfg:<36}{len(rs):>2}  "
          f"{st.mean(g('a')):>5.1f}±{sd(g('a')):<4.1f}  {st.mean(g('e')):>5.1f}±{sd(g('e')):<4.1f}  {st.mean(g('h')):>5.1f}±{sd(g('h')):<4.1f}  "
          f"{st.mean(g('ca')):.4f}±{sd(g('ca')):.4f}  {st.mean(g('ce')):.4f}±{sd(g('ce')):.4f}  {st.mean(g('ch')):.4f}±{sd(g('ch')):.4f}")
