#!/usr/bin/env python3
"""
M5/M6 knob report — C1 (1k->5k), C2 (schema->stats), C3 (delta->latest)
contrasts per model, matched tasks only, with M3/M4 (chunked-judge) and the
native answer score alongside for the same matched sets.

Run: .venv/bin/python scripts/m5m6_report.py
"""
import json, glob, sys
from pathlib import Path

KB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KB))

MODELS = {
    "gpt-5-mini": "DataflowSystemGPT5Mini",
    "gpt-5.2": "DataflowSystemGPT52",
}
KNOBS = [
    ("anchor", "Delta1kSchemaOnly"),
    ("C1 rows 1k->5k", "Delta5kSchemaOnly"),
    ("C2 schema->stats", "DeltaStats1kD2"),
    ("C3 delta->latest", "Latest1kCodeInSnap"),
]


def load_metric(arm, fname, keys):
    out = {}
    for p in glob.glob(str(KB / "system_scratch" / arm / "*" / fname)):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        t = Path(p).parent.name
        out[t] = {k: d.get(k) for k in keys}
    return out


def answer_scores(arm):
    """Primary continuous score per task from latest measures CSVs (via kb.py)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kb", KB / "kb.py")
        kb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kb)
        return kb.answer_scores(arm)
    except Exception:
        return {}


def fmt(x):
    return f"{x:+.3f}" if x is not None else "    —"


def main():
    for model, prefix in MODELS.items():
        arms = {label: prefix + suffix for label, suffix in KNOBS}
        m56 = {l: load_metric(a, "judge_m5m6.json", ["m5", "m6", "m7"]) for l, a in arms.items()}
        m34 = {l: load_metric(a, "judge_m3m4.json", ["m3", "m4_process"]) for l, a in arms.items()}
        ans = {l: answer_scores(a) for l, a in arms.items()}

        print(f"\n{'='*100}\n{model}\n{'='*100}")
        print(f"{'arm':22s} {'n':>4s} {'M5 value':>9s} {'M7 v+fus':>9s} {'M6 step':>9s} {'M3':>7s} {'M4':>7s} {'answer':>7s}")
        for l, a in arms.items():
            ts = m56[l]
            if not ts:
                print(f"{l:22s}    0 (no judge_m5m6 caches)")
                continue
            n = len(ts)
            m5 = sum(v["m5"] for v in ts.values()) / n
            m6 = sum(v["m6"] for v in ts.values()) / n
            m7 = sum(v.get("m7") if v.get("m7") is not None else v["m5"] for v in ts.values()) / n
            j34 = [m34[l][t] for t in ts if t in m34[l]]
            a3 = sum(x["m3"] for x in j34) / len(j34) if j34 else None
            a4 = sum(x["m4_process"] for x in j34) / len(j34) if j34 else None
            av = [ans[l][t] for t in ts if t in ans[l] and ans[l][t] is not None]
            aa = sum(av) / len(av) if av else None
            print(f"{l:22s} {n:4d} {m5:9.3f} {m7:9.3f} {m6:9.3f} "
                  f"{a3 if a3 is not None else float('nan'):7.3f} {a4 if a4 is not None else float('nan'):7.3f} "
                  f"{aa if aa is not None else float('nan'):7.3f}")

        anchor = m56["anchor"]
        print(f"\n  matched-task knob contrasts (ray - anchor):")
        print(f"  {'knob':22s} {'n':>4s} {'dM5':>7s} {'up/dn':>7s} {'dM7':>7s} {'up/dn':>7s} {'dM6':>7s} {'up/dn':>7s} {'dM3':>7s} {'dM4':>7s} {'dANS':>7s}")
        for l, a in arms.items():
            if l == "anchor":
                continue
            ray = m56[l]
            common = sorted(set(anchor) & set(ray))
            if not common:
                print(f"  {l:22s}    0")
                continue
            d5 = [ray[t]["m5"] - anchor[t]["m5"] for t in common]
            d6 = [ray[t]["m6"] - anchor[t]["m6"] for t in common]
            d7 = [(ray[t].get("m7") if ray[t].get("m7") is not None else ray[t]["m5"])
                  - (anchor[t].get("m7") if anchor[t].get("m7") is not None else anchor[t]["m5"])
                  for t in common]
            u5, n5 = sum(1 for x in d5 if x > 1e-9), sum(1 for x in d5 if x < -1e-9)
            u6, n6 = sum(1 for x in d6 if x > 1e-9), sum(1 for x in d6 if x < -1e-9)
            u7, n7 = sum(1 for x in d7 if x > 1e-9), sum(1 for x in d7 if x < -1e-9)
            c34 = [t for t in common if t in m34[l] and t in m34["anchor"]]
            d3 = sum(m34[l][t]["m3"] - m34["anchor"][t]["m3"] for t in c34) / len(c34) if c34 else None
            d4 = sum(m34[l][t]["m4_process"] - m34["anchor"][t]["m4_process"] for t in c34) / len(c34) if c34 else None
            ca = [t for t in common if ans[l].get(t) is not None and ans["anchor"].get(t) is not None]
            da = sum(ans[l][t] - ans["anchor"][t] for t in ca) / len(ca) if ca else None
            print(f"  {l:22s} {len(common):4d} {sum(d5)/len(d5):+7.3f} {f'{u5}/{n5}':>7s} "
                  f"{sum(d7)/len(d7):+7.3f} {f'{u7}/{n7}':>7s} "
                  f"{sum(d6)/len(d6):+7.3f} {f'{u6}/{n6}':>7s} {fmt(d3):>7s} {fmt(d4):>7s} {fmt(da):>7s}")


if __name__ == "__main__":
    main()
