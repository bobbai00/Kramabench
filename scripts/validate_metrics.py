#!/usr/bin/env python3
"""
Validation: do M1 / M2 flip-gaps distinguish hand-labeled ATTRIBUTED flips
(a knob mechanistically caused the win) from CHRONIC coins?

Ground truth = my probe-star semantic-walk verdicts. For each exclusive-win
flip we know winner arm, loser arm, and verdict. A good attribution metric
should give ATTRIBUTED flips a large winner-minus-loser gap and coins ~0.

Run: .venv/bin/python scripts/validate_metrics.py [--with-m2]
"""
import argparse, importlib.util
from pathlib import Path
KB = Path(__file__).resolve().parent.parent

def load(name, path):
    s = importlib.util.spec_from_file_location(name, KB / path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

ARM = {
    "1k":    "DataflowSystemGPT52Delta1kSchemaOnlyProbePrompt",
    "5k":    "DataflowSystemGPT52Delta5kSchemaOnlyProbePrompt",
    "stats": "DataflowSystemGPT52DeltaStats1kD2ProbePrompt",
    "latest":"DataflowSystemGPT52Latest5kSchemaOnlyProbePrompt",
}
# (task, comparison, winner_arm, loser_arm, verdict) — from probe_star/REPORT.md + venn_C*p.txt
GT = [
    # C1  1k(A) vs 5k(B)
    ("legal-hard-2",       "C1", "5k", "1k", "ATTRIBUTED"),
    ("archeology-hard-7",  "C1", "1k", "5k", "chronic"),
    ("legal-easy-19",      "C1", "1k", "5k", "chronic"),
    ("environment-hard-8", "C1", "5k", "1k", "chronic"),
    ("legal-easy-9",       "C1", "5k", "1k", "chronic"),
    ("legal-hard-22",      "C1", "5k", "1k", "chronic"),
    ("wildfire-hard-17",   "C1", "5k", "1k", "chronic"),
    ("wildfire-hard-18",   "C1", "5k", "1k", "chronic"),
    # C2  schema/1k(A) vs stats(B)
    ("biomedical-hard-5",  "C2", "stats", "1k", "ATTRIBUTED"),
    ("archeology-hard-7",  "C2", "1k", "stats", "chronic"),
    ("legal-easy-19",      "C2", "1k", "stats", "chronic"),
    ("wildfire-hard-12",   "C2", "1k", "stats", "chronic"),
    ("environment-hard-7", "C2", "stats", "1k", "chronic"),
    ("environment-hard-8", "C2", "stats", "1k", "chronic"),
    ("legal-easy-9",       "C2", "stats", "1k", "chronic"),
    ("legal-hard-22",      "C2", "stats", "1k", "chronic"),
    ("wildfire-hard-17",   "C2", "stats", "1k", "chronic"),
    # C3  delta/5k(A) vs latest(B)
    ("environment-hard-12","C3", "5k", "latest", "ATTRIBUTED"),
    ("biomedical-easy-2",  "C3", "5k", "latest", "chronic"),
    ("environment-hard-20","C3", "5k", "latest", "chronic"),
    ("legal-hard-18",      "C3", "5k", "latest", "chronic"),
    ("legal-hard-22",      "C3", "5k", "latest", "chronic"),
    ("wildfire-easy-9",    "C3", "5k", "latest", "chronic"),
    ("biomedical-hard-5",  "C3", "latest", "5k", "chronic"),
    ("environment-hard-11","C3", "latest", "5k", "chronic"),
    ("environment-hard-7", "C3", "latest", "5k", "chronic"),
    ("legal-easy-19",      "C3", "latest", "5k", "chronic"),
]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--with-m2", action="store_true"); a = ap.parse_args()
    m1 = load("m1", "scripts/m1_plan_coverage.py")
    W = m1.load_workload()
    m2 = load("m2", "scripts/m2_intermediate_coverage.py") if a.with_m2 else None

    def m1cov(arm, task):
        r = m1.score_task(ARM[arm], task, W[task]); return r["coverage"] if r else None
    m2cache = {}
    def m2cov(arm, task):
        key = (arm, task)
        if key not in m2cache:
            r = m2.score_task_timeout(ARM[arm], task, W[task], timeout=20)
            m2cache[key] = r.get("coverage") if r else None
        return m2cache[key]

    hdr = f"{'flip':30s} {'verdict':11s} {'M1 w':>6s} {'M1 l':>6s} {'M1 gap':>7s}"
    if a.with_m2: hdr += f" {'M2 w':>6s} {'M2 l':>6s} {'M2 gap':>7s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for task, comp, win, lose, verdict in GT:
        if task not in W: continue
        w1, l1 = m1cov(win, task), m1cov(lose, task)
        if w1 is None or l1 is None: continue
        g1 = w1 - l1
        line = f"{comp+' '+task:30s} {verdict:11s} {w1:6.2f} {l1:6.2f} {g1:+7.2f}"
        rec = dict(task=task, comp=comp, verdict=verdict, m1gap=g1)
        if a.with_m2:
            w2, l2 = m2cov(win, task), m2cov(lose, task)
            if w2 is not None and l2 is not None:
                g2 = w2 - l2; rec["m2gap"] = g2
                line += f" {w2:6.2f} {l2:6.2f} {g2:+7.2f}"
            else:
                line += f" {'—':>6s} {'—':>6s} {'—':>7s}"
        rows.append(rec); print(line)

    import statistics as st
    def summ(key):
        att = [r[key] for r in rows if r["verdict"] == "ATTRIBUTED" and key in r]
        chr_ = [r[key] for r in rows if r["verdict"] == "chronic" and key in r]
        print(f"\n  {key}: ATTRIBUTED gaps={[round(x,2) for x in att]} (mean {st.mean(att):+.3f})")
        print(f"      chronic gaps: mean {st.mean(chr_):+.3f}, mean|gap| {st.mean([abs(x) for x in chr_]):.3f}, "
              f"max|gap| {max(abs(x) for x in chr_):.3f}")
        # can a threshold separate ATTRIBUTED from chronic?
        thr_ok = all(a_ > max(abs(x) for x in chr_) for a_ in att) if att and chr_ else False
        print(f"      -> all ATTRIBUTED gaps exceed max chronic |gap|? {thr_ok}")
    print("\n=== attribution separability ===")
    summ("m1gap")
    if a.with_m2: summ("m2gap")

if __name__ == "__main__":
    main()
