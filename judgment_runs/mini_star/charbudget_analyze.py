#!/usr/bin/env python3
"""C9 / C10 / C11 — full-104 three-way comparison of the per-operator CHAR budget.

  C9  LATEST+code : sources 5k + stats, every downstream op 1k + no stats
  C10 DELTA       : same split (the char-budget leg binds on DELTA event renders)
  C11 LATEST+code : 5k + stats for ALL ops (the uniform-rich reference)

Accuracy comes from KramaBench's OWN scores — the native measures CSVs written by
evaluate.py's metric pass, reduced with compute_scores.py's exact formula
(sum(value)/n over SCORE_METRICS) — never a self-invented metric. Cost/token
figures come from each task's stats.json.
"""
import glob, json, os, statistics as st
import pandas as pd

KB = os.path.expanduser("~/Desktop/bobflow/Kramabench")
SCORE_METRICS = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]
WLS = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]
WL_N = {"archeology": 12, "astronomy": 12, "biomedical": 9, "environment": 20,
        "legal": 30, "wildfire": 21}
PIN, PCACHE, POUT = 0.25e-6, 0.025e-6, 2e-6

ARMS = {
    "ANCHOR delta1k schema-only": [f"DataflowSystemGPT5MiniDelta1kSchemaOnlyReplicate{r}" for r in (5, 6, 7)],
    "C9  source-rich LATEST": [f"DataflowSystemGPT5MiniC9SourceRichLatestReplicate{r}" for r in (1, 2, 3)],
    "C10 source-rich DELTA": [f"DataflowSystemGPT5MiniC10SourceRichDeltaReplicate{r}" for r in (1, 2, 3)],
    "C11 uniform-rich LATEST": [f"DataflowSystemGPT5MiniC11UniformRichLatestReplicate{r}" for r in (1, 2, 3)],
}


def native_rows(sut):
    """Freshest FULL-workload measures CSV per workload (skip partial --task_id CSVs)."""
    dfs = []
    for wl in WLS:
        for f in sorted(glob.glob(f"{KB}/results/{sut}/{wl}_measures_*.csv"), reverse=True):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if d[d["metric"].isin(SCORE_METRICS)]["task_id"].nunique() >= WL_N[wl]:
                dfs.append(d)
                break
    if not dfs:
        return None
    d = pd.concat(dfs, ignore_index=True)
    return d[d["metric"].isin(SCORE_METRICS)]


def official(sut):
    d = native_rows(sut)
    if d is None or len(d) == 0:
        return None
    return dict(
        overall=d["value"].sum() / len(d) * 100,
        n=len(d),
        easy=(lambda s: s["value"].sum() / len(s) * 100 if len(s) else float("nan"))(d[d["task_id"].str.contains("-easy-")]),
        n_easy=len(d[d["task_id"].str.contains("-easy-")]),
        hard=(lambda s: s["value"].sum() / len(s) * 100 if len(s) else float("nan"))(d[d["task_id"].str.contains("-hard-")]),
        n_hard=len(d[d["task_id"].str.contains("-hard-")]),
        by_wl={wl: (lambda s: s["value"].sum() / len(s) * 100 if len(s) else float("nan"))(
            d[d["task_id"].str.startswith(wl)]) for wl in WLS},
    )


def usage(sut):
    tin = ca = out = re_ = steps = 0.0
    n = 0
    for f in glob.glob(f"{KB}/system_scratch/{sut}/*/stats.json"):
        try:
            s = json.load(open(f))
        except Exception:
            continue
        tin += s.get("input_tokens", 0); ca += s.get("cached_tokens", 0)
        out += s.get("output_tokens", 0); re_ += s.get("reasoning_tokens", 0)
        steps += s.get("num_steps", 0); n += 1
    if not n:
        return None
    return dict(n=n, tin=tin / n, cached=ca / n, out=out / n, reas=re_ / n, steps=steps / n,
                cachepct=ca / tin * 100 if tin else 0,
                cost=((tin - ca) * PIN + ca * PCACHE + out * POUT) / n)


rows = {}
print("FULL 104 tasks x 3 reps. Accuracy = KramaBench's own score "
      "(native measures CSVs, compute_scores.py formula).\n")
print(f"{'arm':<26}{'acc mean±std':>15}{'easy':>8}{'hard':>8}{'$/task':>9}{'in tok':>9}{'out tok':>9}{'cache%':>8}{'reason':>8}{'steps':>7}")
print("-" * 107)
for label, suts in ARMS.items():
    offs = [official(s) for s in suts]
    offs = [o for o in offs if o]
    us = [usage(s) for s in suts]
    us = [u for u in us if u]
    if not offs or not us:
        print(f"{label:<26} no data")
        continue
    accs = [o["overall"] for o in offs]
    rows[label] = dict(
        accs=accs, acc=st.mean(accs), sd=st.pstdev(accs) if len(accs) > 1 else 0,
        easy=st.mean([o["easy"] for o in offs]), hard=st.mean([o["hard"] for o in offs]),
        by_wl={wl: st.mean([o["by_wl"][wl] for o in offs]) for wl in WLS},
        cost=st.mean([u["cost"] for u in us]), tin=st.mean([u["tin"] for u in us]),
        out=st.mean([u["out"] for u in us]), cache=st.mean([u["cachepct"] for u in us]),
        reas=st.mean([u["reas"] for u in us]), steps=st.mean([u["steps"] for u in us]),
        reps=len(accs), n=offs[0]["n"])
    r = rows[label]
    print(f"{label:<26}{r['acc']:>9.1f} ±{r['sd']:4.1f}{r['easy']:>7.1f}%{r['hard']:>7.1f}%"
          f"{r['cost']:>9.4f}{r['tin']:>9,.0f}{r['out']:>9,.0f}{r['cache']:>7.1f}%{r['reas']:>8,.0f}{r['steps']:>7.1f}")

print(f"\nper-rep official OVERALL:")
for label, r in rows.items():
    print(f"  {label:<26}{['%.1f' % a for a in r['accs']]}   (n={r['n']} metric rows/rep)")

print(f"\nper-workload (avg of 3 reps, KramaBench score):")
print(f"{'arm':<26}" + "".join(f"{w[:11]:>12}" for w in WLS))
for label, r in rows.items():
    print(f"{label:<26}" + "".join(f"{r['by_wl'][w]:>11.1f}%" for w in WLS))

c9, c10, c11 = rows.get("C9  source-rich LATEST"), rows.get("C10 source-rich DELTA"), rows.get("C11 uniform-rich LATEST")
if c9 and c11:
    print(f"\n=== Q1  C9 vs C11 (does lean-downstream beat uniform-rich on COST at equal accuracy?) ===")
    print(f"  accuracy {c9['acc']-c11['acc']:+6.1f} pt   cost {(c9['cost']/c11['cost']-1)*100:+6.1f}%   "
          f"input {c9['tin']-c11['tin']:+8,.0f}   output {c9['out']-c11['out']:+7,.0f}   steps {c9['steps']-c11['steps']:+5.1f}")
    import math
    se = math.sqrt(c9['sd']**2 + c11['sd']**2) / math.sqrt(3)
    d = c9['acc'] - c11['acc']
    print(f"  SE-of-diff {se:.2f} -> |delta| {abs(d):.1f} = {abs(d)/se if se else 0:.2f}x SE "
          f"({'OUTSIDE' if se and abs(d) >= 2*se else 'INSIDE'} noise)")
if c9 and c10:
    print(f"\n=== Q2  C10 vs C9 (same split under DELTA vs LATEST) ===")
    print(f"  accuracy {c10['acc']-c9['acc']:+6.1f} pt   cost {(c10['cost']/c9['cost']-1)*100:+6.1f}%   "
          f"input {c10['tin']-c9['tin']:+8,.0f}   cache {c10['cache']-c9['cache']:+5.1f}pp   steps {c10['steps']-c9['steps']:+5.1f}")

anc = rows.get("ANCHOR delta1k schema-only")
if anc:
    print(f"\n=== vs ANCHOR (delta 1k schema-only, reps 5-7) ===")
    print(f"{'arm':<26}{'acc':>8}{'d acc':>8}{'easy':>8}{'d easy':>8}{'hard':>8}{'d hard':>8}{'$/task':>9}{'d cost':>9}")
    for label, r in rows.items():
        if label.startswith("ANCHOR"):
            print(f"{label:<26}{r['acc']:>7.1f}{'—':>8}{r['easy']:>7.1f}{'—':>8}{r['hard']:>7.1f}{'—':>8}{r['cost']:>9.4f}{'—':>9}")
        else:
            print(f"{label:<26}{r['acc']:>7.1f}{r['acc']-anc['acc']:>+8.1f}{r['easy']:>7.1f}{r['easy']-anc['easy']:>+8.1f}"
                  f"{r['hard']:>7.1f}{r['hard']-anc['hard']:>+8.1f}{r['cost']:>9.4f}{(r['cost']/anc['cost']-1)*100:>+8.1f}%")
