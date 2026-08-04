#!/usr/bin/env python
"""Numeric parity gate: do LongDS's published answers reproduce in OUR venv?

The gold answers were computed against the benchmark's frozen 181-package Python
stack, and the LongDS judge demands exact numeric equality with no rounding. If
our pandas/numpy produce different digits, a correct method grades 0 and the
benchmark measures our library versions instead of our agent. So: run their gold
task.py here, compare every turn's `answers[str(turn)]` against task.json.

task.py has no imports (they lived in a notebook cell) and resolves data through
a relative ROOT, so both are injected before exec.
"""
import json
import os
import sys
from pathlib import Path

TASK = Path(
    "/home/bob/Desktop/bobflow/DataMind/longds/dataset/task/longds/"
    "sports/nfl_big_data_bowl_2023/task1"
)
DATA_ROOT = Path(
    "/home/bob/Desktop/bobflow/DataMind/longds/dataset/data/longds/"
    "sports/nfl_big_data_bowl_2023/task1"
)

PRELUDE = """
import json, re, math, warnings, itertools, collections
from pathlib import Path
import numpy as np
import pandas as pd
"""


def canon(x, places=None):
    """Judge-equivalent normalization: trailing zeros are insignificant."""
    if isinstance(x, dict):
        return {k: canon(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        if x == int(x):
            return int(x)
        return round(x, 10)
    if isinstance(x, int):
        return x
    return x


def diff(gold, got, path=""):
    """Yield (path, gold, got) for every leaf mismatch."""
    if isinstance(gold, dict) and isinstance(got, dict):
        for k in gold:
            if k not in got:
                yield (f"{path}.{k}", gold[k], "<MISSING>")
            else:
                yield from diff(gold[k], got[k], f"{path}.{k}")
        return
    if isinstance(gold, list) and isinstance(got, list):
        if len(gold) != len(got):
            yield (f"{path}[len]", len(gold), len(got))
        for i, (a, b) in enumerate(zip(gold, got)):
            yield from diff(a, b, f"{path}[{i}]")
        return
    if canon(gold) != canon(got):
        yield (path or ".", gold, got)


def main():
    src = (TASK / "task.py").read_text()
    src = src.replace(
        'ROOT =  Path("../../../../../data/longds/sports/nfl_big_data_bowl_2023/task1")',
        f'ROOT = Path("{DATA_ROOT}")',
    )
    if str(DATA_ROOT) not in src:
        print("FATAL: ROOT rewrite did not apply — inspect the literal in task.py")
        return 2

    ns = {"__name__": "__gold__"}
    devnull = open(os.devnull, "w")
    real_stdout = sys.stdout
    sys.stdout = devnull  # task.py prints every answer; we want the dict, not the noise
    try:
        exec(compile(PRELUDE + src, "task.py", "exec"), ns)
    except Exception as exc:
        sys.stdout = real_stdout
        import traceback

        traceback.print_exc()
        print(f"\nFATAL: gold code raised {type(exc).__name__}: {exc}")
        return 2
    finally:
        sys.stdout = real_stdout

    produced = ns.get("answers")
    if not isinstance(produced, dict):
        print("FATAL: gold code produced no `answers` dict")
        return 2

    turns = json.loads((TASK / "task.json").read_text())
    print(f"turns in task.json: {len(turns)}   answers produced: {len(produced)}")
    print(f"pandas {ns['pd'].__version__}  numpy {ns['np'].__version__}")
    print()

    exact, mismatched, missing = [], [], []
    for t in turns:
        tid = str(t["turn_id"])
        if tid not in produced:
            missing.append(tid)
            continue
        bad = list(diff(t["answer"], produced[tid]))
        if bad:
            mismatched.append((tid, bad))
        else:
            exact.append(tid)

    print(f"EXACT      {len(exact)}/{len(turns)}")
    print(f"MISMATCH   {len(mismatched)}")
    print(f"NOT BUILT  {len(missing)}  {missing if missing else ''}")
    for tid, bad in mismatched:
        print(f"\n--- turn {tid}: {len(bad)} leaf mismatches (first 6) ---")
        for p, g, o in bad[:6]:
            print(f"    {p}\n      gold {g!r}\n      ours {o!r}")
    return 0 if not mismatched and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
