#!/usr/bin/env python3
"""
Common-failure analysis across the star arms (C1/C2/C3 + anchor).

Question: the arms sit at ~the same aggregate accuracy — do they fail the
SAME tasks or different ones? Prints per-arm fail sets, the all-arm common
core (the future-focus set), pairwise overlap (Jaccard), and the
uniquely-failed tasks per arm; tags chronic flippers. Saves JSON next to the
levers report.

Run: .venv/bin/python scripts/analyze_common_failures.py [--th 0.9]
"""

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)

ARMS = {
    "Delta3kSchemaOnly": "DataflowSystemGPT52Delta3kSchemaOnly",
    "Delta5kSchemaOnly": "DataflowSystemGPT52Delta5kSchemaOnly",
    "DeltaStats3kD2": "DataflowSystemGPT52DeltaStats3kD2",
    "Latest3kSchemaOnly": "DataflowSystemGPT52Latest3kSchemaOnly",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--th", type=float, default=0.9)
    a = ap.parse_args()

    chron = set(json.load(open(KB / "judgment_runs/levers_report/chronic_flippers.json")))
    tag = lambda t: t + ("*" if t in chron else "")
    fails, passes = {}, {}
    for nm, sut in ARMS.items():
        sc = kb.answer_scores(sut)
        fails[nm] = {t for t, v in sc.items() if v < a.th}
        passes[nm] = {t for t, v in sc.items() if v >= a.th}
        print(f"{nm:20s} fails {len(fails[nm]):3d} / {len(sc)}")

    common = set.intersection(*fails.values())
    union = set.union(*fails.values())
    print(f"\nALL-ARM COMMON FAILURES: {len(common)} of union {len(union)} "
          f"({100*len(common)/max(1,len(union)):.0f}% of any-fail is everybody-fails)")
    print(f"  chronic among common: {sum(1 for t in common if t in chron)}/{len(common)}")
    for t in sorted(common):
        print(f"   {tag(t)}")

    print("\npairwise fail-set overlap (|A∩B| / |A∪B| Jaccard):")
    for x, y in combinations(ARMS, 2):
        i, u = fails[x] & fails[y], fails[x] | fails[y]
        print(f"  {x:20s} vs {y:20s}  ∩={len(i):3d}  J={len(i)/max(1,len(u)):.2f}")

    print("\nuniquely-failed per arm (this arm fails, ALL other arms pass):")
    uniq = {}
    for nm in ARMS:
        others = set.union(*(passes[o] for o in ARMS if o != nm))
        strict_others = set.intersection(*(passes[o] for o in ARMS if o != nm))
        uniq[nm] = sorted(fails[nm] & strict_others)
        print(f"  {nm:20s} ({len(uniq[nm])}): {[tag(t) for t in uniq[nm]]}")

    out = {
        "th": a.th,
        "fails": {k: sorted(v) for k, v in fails.items()},
        "common_failures": sorted(common),
        "common_chronic": sorted(t for t in common if t in chron),
        "unique_failures": uniq,
    }
    path = KB / "judgment_runs/levers_report/common_failures.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n[json] {path.relative_to(KB)}")


if __name__ == "__main__":
    main()
