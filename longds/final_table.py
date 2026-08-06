#!/usr/bin/env python3
"""Baseline vs ours across every judged LongDS task: accuracy, cost, and cost per
correct answer.

Cost alone is misleading when accuracy differs by 30 points: an arm that answers
in two steps and gets 6% right looks "cheap". Cost per correct turn is the honest
efficiency number, so it is reported alongside both raw axes.
"""
import glob
import json
import statistics
from pathlib import Path

KB = Path(__file__).resolve().parents[1]
SCRATCH = KB / "system_scratch"
BASE, OURS = "LongDS_LongDSLunaDelta1k", "LongDS_LongDSLunaRecall"


def load(arm: str, task: str) -> dict | None:
    run = SCRATCH / arm / task
    ev = run / "results_eval.json"
    if not ev.exists():
        return None
    turns = json.loads(ev.read_text())
    scored = [t["judge"]["score"] for t in turns if t.get("judge", {}).get("score") is not None]
    if not scored:
        return None
    stats = [json.loads(p.read_text()) for p in sorted(run.glob("t[0-9][0-9]/stats.json"))]
    cost = sum(s["cost_usd"] for s in stats)
    steps = sum(s["agent_steps"] for s in stats)
    tokens = sum(s["input_tokens"] for s in stats)
    return {
        "n": len(scored),
        "acc": statistics.mean(scored) * 100,
        "correct": sum(scored),
        "cost": cost,
        "steps": steps / max(1, len(stats)),
        "tok_step": tokens / max(1, steps),
    }


def main() -> int:
    tasks = sorted({p.name for p in (SCRATCH / OURS).iterdir()} & {p.name for p in (SCRATCH / BASE).iterdir()})
    rows = []
    for task in tasks:
        b, o = load(BASE, task), load(OURS, task)
        if b and o:
            rows.append((task, b, o))
    if not rows:
        print("no task has both arms judged")
        return 1

    print(f"{'task':<34}{'turns':>6}{'base acc':>10}{'our acc':>9}{'base $':>9}{'our $':>8}"
          f"{'base $/ok':>11}{'our $/ok':>10}{'our steps':>11}")
    print("-" * 108)
    for task, b, o in rows:
        short = task.replace("business__", "").replace("geoscience__", "")[:32]
        bpc = b["cost"] / b["correct"] if b["correct"] else float("inf")
        opc = o["cost"] / o["correct"] if o["correct"] else float("inf")
        print(f"{short:<34}{o['n']:>6}{b['acc']:>9.1f}%{o['acc']:>8.1f}%{b['cost']:>9.2f}{o['cost']:>8.2f}"
              f"{bpc:>11.3f}{opc:>10.3f}{o['steps']:>11.1f}")

    tb = sum(b["n"] for _, b, _ in rows)
    acc_b = sum(b["acc"] * b["n"] for _, b, _ in rows) / tb
    acc_o = sum(o["acc"] * o["n"] for _, _, o in rows) / tb
    cost_b = sum(b["cost"] for _, b, _ in rows)
    cost_o = sum(o["cost"] for _, _, o in rows)
    ok_b = sum(b["correct"] for _, b, _ in rows)
    ok_o = sum(o["correct"] for _, _, o in rows)
    print("-" * 108)
    print(f"{'TURN-WEIGHTED (' + str(tb) + ' turns)':<34}{'':>6}{acc_b:>9.1f}%{acc_o:>8.1f}%"
          f"{cost_b:>9.2f}{cost_o:>8.2f}{cost_b / max(1, ok_b):>11.3f}{cost_o / max(1, ok_o):>10.3f}")
    print(f"\ncorrect turns: baseline {ok_b:.0f}, ours {ok_o:.0f} of {tb}")
    print(f"cost per correct turn: baseline ${cost_b / max(1, ok_b):.3f}, ours ${cost_o / max(1, ok_o):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
