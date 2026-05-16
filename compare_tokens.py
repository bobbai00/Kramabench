#!/usr/bin/env python3
"""Compare token usage between two DataflowSystem variants by workload and difficulty.

Usage:
    python compare_tokens.py                        # use the SYSTEMS list below
    python compare_tokens.py <SUT1> <SUT2> ...      # override SYSTEMS from argv
"""

import json
import os
import sys
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import trim_mean

import pandas as pd

SCRATCH_DIR = Path("system_scratch")

# Pricing per million tokens. Keys are the substring used to identify the model
# from a system name (substring match, longest key wins).
MODEL_PRICING = {
    "Gpt52":   {"input": 1.75, "output": 14.00},
    "Gpt5Mini":{"input": 0.25, "output":  2.00},
    "GPT5Mini":{"input": 0.25, "output":  2.00},
    "Haiku45": {"input": 1.00, "output":  5.00},
}

# Default pricing (GPT-5.2)
DEFAULT_PRICING = MODEL_PRICING["Gpt52"]


def get_pricing(system_name: str) -> dict:
    """Determine pricing based on the model identifier in the system name."""
    # Check longest keys first to avoid partial matches (e.g. "Gpt5Mini" before "Gpt5")
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if key in system_name:
            return MODEL_PRICING[key]
    return DEFAULT_PRICING


def cost(input_tokens, output_tokens, pricing=None):
    """Calculate cost in USD."""
    if pricing is None:
        pricing = DEFAULT_PRICING
    return input_tokens / 1_000_000 * pricing["input"] + output_tokens / 1_000_000 * pricing["output"]

SYSTEMS = [
    "CodeAgentSystemGpt52",
    "DataflowSystemGpt52ResultChars5000",
    "DataflowSystemGpt52ResultChars5000Hamilton",
    "DataflowSystemGpt52ResultChars5000LatestOnly",
    "DataflowSystemGpt52ResultChars5000OptionalResultRetrieval",
    "DataflowSystemGpt52ResultChars5000ParallelToolCalls",
    "DataflowSystemGpt52ResultChars5000ParallelOptionalRetrieval"
]

# SYSTEMS = [
#     "CodeAgentSystemGpt5MiniMedium",
#     "DataflowSystemGpt5MiniMedium",
#     "DataflowSystemGpt5MiniMediumResultChars5000",
#     "DataflowSystemGpt5MiniMediumResultChars5000LatestOnly" 
# ]

RESULTS_DIR = Path("results")
SCORE_METRICS = ["success", "llm_paraphrase", "rae_score", "f1", "f1_approximate"]


def load_stats(system_name: str) -> list[dict]:
    """Load stats.json from every task directory for a given system."""
    base = SCRATCH_DIR / system_name
    records = []
    for task_dir in sorted(base.iterdir()):
        if not task_dir.is_dir():
            continue
        stats_file = task_dir / "stats.json"
        if not stats_file.exists():
            continue
        with open(stats_file) as f:
            stats = json.load(f)
        task_id = task_dir.name  # e.g. "archeology-easy-10"
        parts = task_id.rsplit("-", 2)  # ["archeology", "easy", "10"]
        if len(parts) >= 3:
            workload = parts[0]
            difficulty = parts[1]
        else:
            workload = parts[0]
            difficulty = "unknown"
        records.append({
            "task_id": task_id,
            "workload": workload,
            "difficulty": difficulty,
            "input_tokens": stats.get("input_tokens", 0),
            "output_tokens": stats.get("output_tokens", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "num_steps": stats.get("num_steps", 0),
            "elapsed_seconds": stats.get("elapsed_seconds", 0),
        })
    return records


def fmt(n):
    """Format number with commas."""
    return f"{n:>12,}"


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for r in rows:
                w = max(w, len(str(r[i])))
            col_widths.append(w + 2)

    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += str(h).rjust(w)
    print(header_line)
    print("-" * sum(col_widths))
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            line += str(val).rjust(w)
        print(line)


def analyze_by_group(records, group_key):
    """Aggregate token stats by a grouping key."""
    groups = defaultdict(lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "num_steps": 0, "elapsed_seconds": 0, "count": 0
    })
    for r in records:
        g = groups[r[group_key]]
        g["input_tokens"] += r["input_tokens"]
        g["output_tokens"] += r["output_tokens"]
        g["total_tokens"] += r["total_tokens"]
        g["num_steps"] += r["num_steps"]
        g["elapsed_seconds"] += r["elapsed_seconds"]
        g["count"] += 1
    return dict(sorted(groups.items()))


def load_task_success(system_name: str) -> dict[str, float]:
    """Load per-task success from the latest results CSV for each workload.

    Returns {task_id: max_score} where max_score is the max value across
    SCORE_METRICS for that task (1.0 = success, <1.0 = fail).
    """
    rdir = RESULTS_DIR / system_name
    if not rdir.exists():
        return {}
    # Pick the latest CSV per workload (sorted by timestamp in filename)
    files = sorted(rdir.glob("*_measures_*.csv"))
    latest: dict[str, Path] = {}
    for f in files:
        wl = f.name.split("_measures_")[0]
        latest[wl] = f  # last one wins
    dfs = [pd.read_csv(f) for f in latest.values()]
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    score_df = df[df["metric"].isin(SCORE_METRICS)]
    return score_df.groupby("task_id")["value"].max().to_dict()


def is_infra_error(system_name: str, task_id: str, record: dict) -> bool:
    """Detect infrastructure-level errors (null answers, zero tokens)."""
    # Zero input tokens means the LLM was never called
    if record["input_tokens"] == 0:
        return True
    # Check answer.json for null
    answer_file = SCRATCH_DIR / system_name / task_id / "answer.json"
    if answer_file.exists():
        try:
            ans = json.load(open(answer_file))["answer"]
            if ans is None or str(ans).strip().lower() in ("null", "none", ""):
                return True
        except Exception:
            return True
    else:
        return True
    return False


TRIM_PROPORTION = 0.1  # 10% trim from each side (scipy default-style)


def safe_trim_mean(values: list[float]) -> float:
    """Compute 10% trimmed mean. Falls back to regular mean for small samples."""
    if not values:
        return 0.0
    if len(values) < 5:
        return statistics.mean(values)
    return trim_mean(values, TRIM_PROPORTION)


def main():
    global SYSTEMS
    # Allow ad-hoc system lists via argv without editing the module header.
    if len(sys.argv) > 1:
        SYSTEMS = sys.argv[1:]

    all_data = {}
    for sys_name in SYSTEMS:
        all_data[sys_name] = load_stats(sys_name)
        print(f"Loaded {len(all_data[sys_name])} tasks for {sys_name}")

    # ── Grand totals ──
    print_section("GRAND TOTALS")
    headers = ["System", "Tasks", "Input Tokens", "Output Tokens", "Total Tokens", "Steps", "Time (s)"]
    rows = []
    for sys_name in SYSTEMS:
        recs = all_data[sys_name]
        rows.append([
            sys_name,
            fmt(len(recs)),
            fmt(sum(r["input_tokens"] for r in recs)),
            fmt(sum(r["output_tokens"] for r in recs)),
            fmt(sum(r["total_tokens"] for r in recs)),
            fmt(sum(r["num_steps"] for r in recs)),
            fmt(round(sum(r["elapsed_seconds"] for r in recs))),
        ])
    print_table(headers, rows)

    # ── Cost Analysis ──
    print_section("COST ANALYSIS (model-specific pricing)")
    headers = ["System", "Model", "Total Cost", "Avg Cost/Task", "Input Cost", "Output Cost"]
    rows = []
    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        recs = all_data[sys_name]
        tot_i = sum(r["input_tokens"] for r in recs)
        tot_o = sum(r["output_tokens"] for r in recs)
        total_cost = cost(tot_i, tot_o, p)
        input_cost = tot_i / 1_000_000 * p["input"]
        output_cost = tot_o / 1_000_000 * p["output"]
        avg_cost = total_cost / len(recs)
        pricing_label = [k for k, v in MODEL_PRICING.items() if v == p][0]
        rows.append([
            sys_name,
            f"{pricing_label} (${p['input']}/{p['output']})",
            f"${total_cost:>9.2f}",
            f"${avg_cost:>7.4f}",
            f"${input_cost:>9.2f}",
            f"${output_cost:>8.2f}",
        ])
    print_table(headers, rows)

    # ── Cost by Workload ──
    print_section("COST BY WORKLOAD")
    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        print(f"\n  --- {sys_name} ---")
        by_wl = analyze_by_group(all_data[sys_name], "workload")
        headers = ["Workload", "Tasks", "Total Cost", "Avg Cost/Task", "Input Cost", "Output Cost"]
        rows = []
        for wl, g in by_wl.items():
            tc = cost(g["input_tokens"], g["output_tokens"], p)
            ic = g["input_tokens"] / 1_000_000 * p["input"]
            oc = g["output_tokens"] / 1_000_000 * p["output"]
            rows.append([
                wl,
                fmt(g["count"]),
                f"${tc:>8.2f}",
                f"${tc / g['count']:>7.4f}",
                f"${ic:>8.2f}",
                f"${oc:>7.2f}",
            ])
        print_table(headers, rows)

    # ── Cost by Difficulty ──
    print_section("COST BY DIFFICULTY")
    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        print(f"\n  --- {sys_name} ---")
        by_diff = analyze_by_group(all_data[sys_name], "difficulty")
        headers = ["Difficulty", "Tasks", "Total Cost", "Avg Cost/Task", "Input Cost", "Output Cost"]
        rows = []
        for d, g in by_diff.items():
            tc = cost(g["input_tokens"], g["output_tokens"], p)
            ic = g["input_tokens"] / 1_000_000 * p["input"]
            oc = g["output_tokens"] / 1_000_000 * p["output"]
            rows.append([
                d,
                fmt(g["count"]),
                f"${tc:>8.2f}",
                f"${tc / g['count']:>7.4f}",
                f"${ic:>8.2f}",
                f"${oc:>7.2f}",
            ])
        print_table(headers, rows)

    # Savings
    t0 = sum(r["total_tokens"] for r in all_data[SYSTEMS[0]])
    t1 = sum(r["total_tokens"] for r in all_data[SYSTEMS[1]])
    i0 = sum(r["input_tokens"] for r in all_data[SYSTEMS[0]])
    i1 = sum(r["input_tokens"] for r in all_data[SYSTEMS[1]])
    o0 = sum(r["output_tokens"] for r in all_data[SYSTEMS[0]])
    o1 = sum(r["output_tokens"] for r in all_data[SYSTEMS[1]])
    s0 = sum(r["num_steps"] for r in all_data[SYSTEMS[0]])
    s1 = sum(r["num_steps"] for r in all_data[SYSTEMS[1]])
    print_section(f"TOKEN CHANGE ({SYSTEMS[1]} vs {SYSTEMS[0]})")
    print(f"    Input tokens:  {i1 - i0:+,} ({(i1 - i0) / i0 * 100:+.1f}%)")
    print(f"    Output tokens: {o1 - o0:+,} ({(o1 - o0) / o0 * 100:+.1f}%)")
    print(f"    Total tokens:  {t1 - t0:+,} ({(t1 - t0) / t0 * 100:+.1f}%)")
    print(f"    Steps:         {s1 - s0:+,} ({(s1 - s0) / s0 * 100:+.1f}%)")
    print(f"    Avg input/step:  {i0 // s0:,} -> {i1 // s1:,}  ({(i1 / s1 - i0 / s0) / (i0 / s0) * 100:+.1f}%)")
    print(f"    Avg output/step: {o0 // s0:,} -> {o1 // s1:,}  ({(o1 / s1 - o0 / s0) / (o0 / s0) * 100:+.1f}%)")
    p0 = get_pricing(SYSTEMS[0])
    p1 = get_pricing(SYSTEMS[1])
    c0 = cost(i0, o0, p0)
    c1 = cost(i1, o1, p1)
    print(f"    Cost:            ${c0:.2f} -> ${c1:.2f}  ({(c1 - c0) / c0 * 100:+.1f}%)")

    # ── Per-Step Analysis: Grand ──
    print_section("PER-STEP TOKEN ANALYSIS — GRAND TOTALS")
    headers = ["System", "Total Steps", "Avg Steps/Task", "Avg Input/Step", "Avg Output/Step", "Avg Total/Step"]
    rows = []
    for sys_name in SYSTEMS:
        recs = all_data[sys_name]
        tot_i = sum(r["input_tokens"] for r in recs)
        tot_o = sum(r["output_tokens"] for r in recs)
        tot_t = sum(r["total_tokens"] for r in recs)
        tot_s = sum(r["num_steps"] for r in recs)
        rows.append([
            sys_name,
            fmt(tot_s),
            f"{tot_s / len(recs):.1f}".rjust(12),
            fmt(tot_i // tot_s),
            fmt(tot_o // tot_s),
            fmt(tot_t // tot_s),
        ])
    print_table(headers, rows)

    # ── Per-Step Analysis: By Workload ──
    print_section("PER-STEP TOKEN ANALYSIS — BY WORKLOAD")
    for sys_name in SYSTEMS:
        print(f"\n  --- {sys_name} ---")
        by_wl = analyze_by_group(all_data[sys_name], "workload")
        headers = ["Workload", "Steps", "Avg Steps/Task", "Avg Input/Step", "Avg Output/Step", "Avg Total/Step"]
        rows = []
        for wl, g in by_wl.items():
            rows.append([
                wl,
                fmt(g["num_steps"]),
                f"{g['num_steps'] / g['count']:.1f}".rjust(12),
                fmt(g["input_tokens"] // g["num_steps"]) if g["num_steps"] else "N/A",
                fmt(g["output_tokens"] // g["num_steps"]) if g["num_steps"] else "N/A",
                fmt(g["total_tokens"] // g["num_steps"]) if g["num_steps"] else "N/A",
            ])
        print_table(headers, rows)

    # ── Per-Step Comparison by Workload ──
    print_section("PER-STEP COMPARISON BY WORKLOAD (CtxOptD2 vs Gpt52)")
    by_wl_0 = analyze_by_group(all_data[SYSTEMS[0]], "workload")
    by_wl_1 = analyze_by_group(all_data[SYSTEMS[1]], "workload")
    headers = ["Workload", "Steps Δ", "Steps Δ%", "Avg In/Step Δ", "Avg In/Step Δ%", "Avg Out/Step Δ", "Avg Out/Step Δ%"]
    rows = []
    for wl in sorted(set(by_wl_0.keys()) | set(by_wl_1.keys())):
        g0 = by_wl_0.get(wl)
        g1 = by_wl_1.get(wl)
        if not g0 or not g1 or not g0["num_steps"] or not g1["num_steps"]:
            continue
        ds = g1["num_steps"] - g0["num_steps"]
        dsp = ds / g0["num_steps"] * 100
        ais0 = g0["input_tokens"] / g0["num_steps"]
        ais1 = g1["input_tokens"] / g1["num_steps"]
        aos0 = g0["output_tokens"] / g0["num_steps"]
        aos1 = g1["output_tokens"] / g1["num_steps"]
        rows.append([
            wl,
            fmt(ds), f"{dsp:+.1f}%",
            fmt(round(ais1 - ais0)), f"{(ais1 - ais0) / ais0 * 100:+.1f}%",
            fmt(round(aos1 - aos0)), f"{(aos1 - aos0) / aos0 * 100:+.1f}%",
        ])
    print_table(headers, rows)

    # ── Per-Step Analysis: By Difficulty ──
    print_section("PER-STEP TOKEN ANALYSIS — BY DIFFICULTY")
    for sys_name in SYSTEMS:
        print(f"\n  --- {sys_name} ---")
        by_diff = analyze_by_group(all_data[sys_name], "difficulty")
        headers = ["Difficulty", "Steps", "Avg Steps/Task", "Avg Input/Step", "Avg Output/Step", "Avg Total/Step"]
        rows = []
        for d, g in by_diff.items():
            rows.append([
                d,
                fmt(g["num_steps"]),
                f"{g['num_steps'] / g['count']:.1f}".rjust(12),
                fmt(g["input_tokens"] // g["num_steps"]) if g["num_steps"] else "N/A",
                fmt(g["output_tokens"] // g["num_steps"]) if g["num_steps"] else "N/A",
                fmt(g["total_tokens"] // g["num_steps"]) if g["num_steps"] else "N/A",
            ])
        print_table(headers, rows)

    # ── Per-Step Comparison by Difficulty ──
    print_section("PER-STEP COMPARISON BY DIFFICULTY (CtxOptD2 vs Gpt52)")
    by_d_0 = analyze_by_group(all_data[SYSTEMS[0]], "difficulty")
    by_d_1 = analyze_by_group(all_data[SYSTEMS[1]], "difficulty")
    headers = ["Difficulty", "Steps Δ", "Steps Δ%", "Avg In/Step Δ", "Avg In/Step Δ%", "Avg Out/Step Δ", "Avg Out/Step Δ%"]
    rows = []
    for d in sorted(set(by_d_0.keys()) | set(by_d_1.keys())):
        g0 = by_d_0.get(d)
        g1 = by_d_1.get(d)
        if not g0 or not g1 or not g0["num_steps"] or not g1["num_steps"]:
            continue
        ds = g1["num_steps"] - g0["num_steps"]
        dsp = ds / g0["num_steps"] * 100
        ais0 = g0["input_tokens"] / g0["num_steps"]
        ais1 = g1["input_tokens"] / g1["num_steps"]
        aos0 = g0["output_tokens"] / g0["num_steps"]
        aos1 = g1["output_tokens"] / g1["num_steps"]
        rows.append([
            d,
            fmt(ds), f"{dsp:+.1f}%",
            fmt(round(ais1 - ais0)), f"{(ais1 - ais0) / ais0 * 100:+.1f}%",
            fmt(round(aos1 - aos0)), f"{(aos1 - aos0) / aos0 * 100:+.1f}%",
        ])
    print_table(headers, rows)

    # ── Per-Task Step & Per-Step Token comparison ──
    print_section("TOP 20 TASKS BY PER-STEP INPUT TOKEN CHANGE")
    task_map_0 = {r["task_id"]: r for r in all_data[SYSTEMS[0]]}
    task_map_1 = {r["task_id"]: r for r in all_data[SYSTEMS[1]]}
    per_step_diffs = []
    for tid in set(task_map_0.keys()) & set(task_map_1.keys()):
        r0 = task_map_0[tid]
        r1 = task_map_1[tid]
        if r0["num_steps"] == 0 or r1["num_steps"] == 0:
            continue
        ips0 = r0["input_tokens"] / r0["num_steps"]
        ips1 = r1["input_tokens"] / r1["num_steps"]
        per_step_diffs.append((tid, r0["num_steps"], r1["num_steps"], round(ips0), round(ips1), round(ips1 - ips0)))
    per_step_diffs.sort(key=lambda x: x[5])  # most negative first

    headers = ["Task ID", "Steps(G52)", "Steps(Ctx)", "In/Step(G52)", "In/Step(Ctx)", "Δ In/Step", "Δ%"]
    rows = []
    for tid, s0, s1, ips0, ips1, d in per_step_diffs[:10]:
        pct = (d / ips0 * 100) if ips0 else 0
        rows.append([tid, fmt(s0), fmt(s1), fmt(ips0), fmt(ips1), fmt(d), f"{pct:+.1f}%"])
    rows.append(["...", "", "", "", "", "", ""])
    for tid, s0, s1, ips0, ips1, d in per_step_diffs[-10:]:
        pct = (d / ips0 * 100) if ips0 else 0
        rows.append([tid, fmt(s0), fmt(s1), fmt(ips0), fmt(ips1), fmt(d), f"{pct:+.1f}%"])
    print_table(headers, rows)

    # ── By Workload ──
    print_section("BY WORKLOAD")
    for sys_name in SYSTEMS:
        print(f"\n  --- {sys_name} ---")
        by_wl = analyze_by_group(all_data[sys_name], "workload")
        headers = ["Workload", "Tasks", "Input Tokens", "Output Tokens", "Total Tokens", "Avg Input/Task", "Avg Output/Task"]
        rows = []
        for wl, g in by_wl.items():
            rows.append([
                wl,
                fmt(g["count"]),
                fmt(g["input_tokens"]),
                fmt(g["output_tokens"]),
                fmt(g["total_tokens"]),
                fmt(g["input_tokens"] // g["count"]),
                fmt(g["output_tokens"] // g["count"]),
            ])
        print_table(headers, rows)

    # ── By Workload comparison ──
    print_section("BY WORKLOAD — COMPARISON (CtxOptD2 vs Gpt52)")
    by_wl_0 = analyze_by_group(all_data[SYSTEMS[0]], "workload")
    by_wl_1 = analyze_by_group(all_data[SYSTEMS[1]], "workload")
    headers = ["Workload", "Input Δ", "Input Δ%", "Output Δ", "Output Δ%", "Total Δ", "Total Δ%"]
    rows = []
    for wl in sorted(set(by_wl_0.keys()) | set(by_wl_1.keys())):
        g0 = by_wl_0.get(wl, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        g1 = by_wl_1.get(wl, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        di = g1["input_tokens"] - g0["input_tokens"]
        do = g1["output_tokens"] - g0["output_tokens"]
        dt = g1["total_tokens"] - g0["total_tokens"]
        dip = (di / g0["input_tokens"] * 100) if g0["input_tokens"] else 0
        dop = (do / g0["output_tokens"] * 100) if g0["output_tokens"] else 0
        dtp = (dt / g0["total_tokens"] * 100) if g0["total_tokens"] else 0
        rows.append([wl, fmt(di), f"{dip:+.1f}%", fmt(do), f"{dop:+.1f}%", fmt(dt), f"{dtp:+.1f}%"])
    print_table(headers, rows)

    # ── By Difficulty ──
    print_section("BY DIFFICULTY")
    for sys_name in SYSTEMS:
        print(f"\n  --- {sys_name} ---")
        by_diff = analyze_by_group(all_data[sys_name], "difficulty")
        headers = ["Difficulty", "Tasks", "Input Tokens", "Output Tokens", "Total Tokens", "Avg Input/Task", "Avg Output/Task"]
        rows = []
        for d, g in by_diff.items():
            rows.append([
                d,
                fmt(g["count"]),
                fmt(g["input_tokens"]),
                fmt(g["output_tokens"]),
                fmt(g["total_tokens"]),
                fmt(g["input_tokens"] // g["count"]),
                fmt(g["output_tokens"] // g["count"]),
            ])
        print_table(headers, rows)

    # ── By Difficulty comparison ──
    print_section("BY DIFFICULTY — COMPARISON (CtxOptD2 vs Gpt52)")
    by_d_0 = analyze_by_group(all_data[SYSTEMS[0]], "difficulty")
    by_d_1 = analyze_by_group(all_data[SYSTEMS[1]], "difficulty")
    headers = ["Difficulty", "Input Δ", "Input Δ%", "Output Δ", "Output Δ%", "Total Δ", "Total Δ%"]
    rows = []
    for d in sorted(set(by_d_0.keys()) | set(by_d_1.keys())):
        g0 = by_d_0.get(d, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        g1 = by_d_1.get(d, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        di = g1["input_tokens"] - g0["input_tokens"]
        do = g1["output_tokens"] - g0["output_tokens"]
        dt = g1["total_tokens"] - g0["total_tokens"]
        dip = (di / g0["input_tokens"] * 100) if g0["input_tokens"] else 0
        dop = (do / g0["output_tokens"] * 100) if g0["output_tokens"] else 0
        dtp = (dt / g0["total_tokens"] * 100) if g0["total_tokens"] else 0
        rows.append([d, fmt(di), f"{dip:+.1f}%", fmt(do), f"{dop:+.1f}%", fmt(dt), f"{dtp:+.1f}%"])
    print_table(headers, rows)

    # ── By Workload × Difficulty ──
    print_section("BY WORKLOAD × DIFFICULTY")
    for sys_name in SYSTEMS:
        print(f"\n  --- {sys_name} ---")
        groups = defaultdict(lambda: {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "count": 0
        })
        for r in all_data[sys_name]:
            key = (r["workload"], r["difficulty"])
            g = groups[key]
            g["input_tokens"] += r["input_tokens"]
            g["output_tokens"] += r["output_tokens"]
            g["total_tokens"] += r["total_tokens"]
            g["count"] += 1
        headers = ["Workload", "Difficulty", "Tasks", "Input Tokens", "Output Tokens", "Total Tokens"]
        rows = []
        for (wl, diff), g in sorted(groups.items()):
            rows.append([wl, diff, fmt(g["count"]), fmt(g["input_tokens"]), fmt(g["output_tokens"]), fmt(g["total_tokens"])])
        print_table(headers, rows)

    # ── Per-task detail (sorted by biggest token difference) ──
    print_section("TOP 20 TASKS BY TOTAL TOKEN DIFFERENCE (CtxOptD2 - Gpt52)")
    task_map_0 = {r["task_id"]: r for r in all_data[SYSTEMS[0]]}
    task_map_1 = {r["task_id"]: r for r in all_data[SYSTEMS[1]]}
    diffs = []
    for tid in set(task_map_0.keys()) | set(task_map_1.keys()):
        r0 = task_map_0.get(tid, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        r1 = task_map_1.get(tid, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        diffs.append((tid, r1["total_tokens"] - r0["total_tokens"], r0["total_tokens"], r1["total_tokens"]))
    diffs.sort(key=lambda x: x[1])  # most negative first (biggest savings)

    headers = ["Task ID", "Gpt52 Total", "CtxOptD2 Total", "Δ Total", "Δ%"]
    rows = []
    for tid, dt, t0, t1 in diffs[:10]:
        pct = (dt / t0 * 100) if t0 else 0
        rows.append([tid, fmt(t0), fmt(t1), fmt(dt), f"{pct:+.1f}%"])
    rows.append(["...", "", "", "", ""])
    for tid, dt, t0, t1 in diffs[-10:]:
        pct = (dt / t0 * 100) if t0 else 0
        rows.append([tid, fmt(t0), fmt(t1), fmt(dt), f"{pct:+.1f}%"])
    print_table(headers, rows)

    # ── Avg Cost Per Task: All / Successful / Failed (filtered) ──
    print_section(f"AVG COST PER TASK — ALL / SUCCESSFUL / FAILED ({TRIM_PROPORTION:.0%} trimmed mean, infra errors removed)")

    # Load task success for each system
    task_success = {}
    for sys_name in SYSTEMS:
        task_success[sys_name] = load_task_success(sys_name)

    # Build per-task cost lists, split by success/fail, excluding infra errors
    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        success_map = task_success[sys_name]
        costs_all = []
        costs_success = []
        costs_fail = []
        infra_errors = []
        no_result = []

        for r in all_data[sys_name]:
            tid = r["task_id"]
            c = cost(r["input_tokens"], r["output_tokens"], p)

            # Skip infrastructure errors
            if is_infra_error(sys_name, tid, r):
                infra_errors.append(tid)
                continue

            # Skip tasks not in results (no evaluation was run)
            if tid not in success_map:
                no_result.append(tid)
                continue

            costs_all.append(c)
            if success_map[tid] >= 1.0:
                costs_success.append(c)
            else:
                costs_fail.append(c)

        print(f"\n  --- {sys_name} ---")
        if infra_errors:
            print(f"    Infrastructure errors removed: {len(infra_errors)} tasks {infra_errors}")
        if no_result:
            print(f"    No evaluation result: {len(no_result)} tasks {no_result}")

        headers = ["Category", "Tasks", "Trimmed Mean", "Median", "Mean (raw)", "Stdev (raw)", "Total"]
        rows = []
        for label, lst in [("All", costs_all),
                            ("Successful", costs_success),
                            ("Failed", costs_fail)]:
            if lst:
                tm = safe_trim_mean(lst)
                med = statistics.median(lst)
                raw_mean = statistics.mean(lst)
                sd = statistics.stdev(lst) if len(lst) > 1 else 0
                total = sum(lst)
            else:
                tm = med = raw_mean = sd = total = 0
            rows.append([
                label,
                str(len(lst)),
                f"${tm:.4f}",
                f"${med:.4f}",
                f"${raw_mean:.4f}",
                f"${sd:.4f}",
                f"${total:.4f}",
            ])
        print_table(headers, rows)

    # ── Cross-system comparison table ──
    print_section(f"AVG COST PER TASK — CROSS-SYSTEM COMPARISON ({TRIM_PROPORTION:.0%} trimmed mean)")
    headers = ["System", "All TrMean", "All Med", "Succ TrMean", "Succ Med", "Fail TrMean", "Fail Med", "#Succ", "#Fail"]
    rows = []
    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        success_map = task_success[sys_name]
        costs_all = []
        costs_success = []
        costs_fail = []
        for r in all_data[sys_name]:
            tid = r["task_id"]
            if is_infra_error(sys_name, tid, r):
                continue
            if tid not in success_map:
                continue
            c = cost(r["input_tokens"], r["output_tokens"], p)
            costs_all.append(c)
            if success_map[tid] >= 1.0:
                costs_success.append(c)
            else:
                costs_fail.append(c)

        def safe_median(lst): return statistics.median(lst) if lst else 0

        rows.append([
            sys_name,
            f"${safe_trim_mean(costs_all):.4f}",
            f"${safe_median(costs_all):.4f}",
            f"${safe_trim_mean(costs_success):.4f}",
            f"${safe_median(costs_success):.4f}",
            f"${safe_trim_mean(costs_fail):.4f}",
            f"${safe_median(costs_fail):.4f}",
            str(len(costs_success)),
            str(len(costs_fail)),
        ])
    print_table(headers, rows)

    # ── Avg Cost Per Task: All / Successful / Failed (unfiltered, infra errors only removed) ──
    print_section("AVG COST PER TASK — ALL / SUCCESSFUL / FAILED (no outlier removal, infra errors removed)")

    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        success_map = task_success[sys_name]
        costs_all = []
        costs_success = []
        costs_fail = []
        infra_errors = []
        no_result = []

        for r in all_data[sys_name]:
            tid = r["task_id"]
            c = cost(r["input_tokens"], r["output_tokens"], p)

            if is_infra_error(sys_name, tid, r):
                infra_errors.append(tid)
                continue

            if tid not in success_map:
                no_result.append(tid)
                continue

            costs_all.append(c)
            if success_map[tid] >= 1.0:
                costs_success.append(c)
            else:
                costs_fail.append(c)

        print(f"\n  --- {sys_name} ---")
        if infra_errors:
            print(f"    Infrastructure errors removed: {len(infra_errors)} tasks {infra_errors}")
        if no_result:
            print(f"    No evaluation result: {len(no_result)} tasks {no_result}")

        headers = ["Category", "Tasks", "Avg Cost", "Median Cost", "Stdev", "Total Cost"]
        rows = []
        for label, lst in [("All", costs_all),
                           ("Successful", costs_success),
                           ("Failed", costs_fail)]:
            if lst:
                avg = statistics.mean(lst)
                med = statistics.median(lst)
                sd = statistics.stdev(lst) if len(lst) > 1 else 0
                total = sum(lst)
            else:
                avg = med = sd = total = 0
            rows.append([
                label,
                str(len(lst)),
                f"${avg:.4f}",
                f"${med:.4f}",
                f"${sd:.4f}",
                f"${total:.4f}",
            ])
        print_table(headers, rows)

    # ── Cross-system comparison table (unfiltered) ──
    print_section("AVG COST PER TASK — CROSS-SYSTEM COMPARISON (no outlier removal)")
    headers = ["System", "All Avg", "All Med", "Succ Avg", "Succ Med", "Fail Avg", "Fail Med", "#Succ", "#Fail", "Total Cost"]
    rows = []
    for sys_name in SYSTEMS:
        p = get_pricing(sys_name)
        success_map = task_success[sys_name]
        costs_all = []
        costs_success = []
        costs_fail = []
        for r in all_data[sys_name]:
            tid = r["task_id"]
            if is_infra_error(sys_name, tid, r):
                continue
            if tid not in success_map:
                continue
            c = cost(r["input_tokens"], r["output_tokens"], p)
            costs_all.append(c)
            if success_map[tid] >= 1.0:
                costs_success.append(c)
            else:
                costs_fail.append(c)

        def safe_mean(lst): return statistics.mean(lst) if lst else 0
        def safe_median(lst): return statistics.median(lst) if lst else 0

        rows.append([
            sys_name,
            f"${safe_mean(costs_all):.4f}",
            f"${safe_median(costs_all):.4f}",
            f"${safe_mean(costs_success):.4f}",
            f"${safe_median(costs_success):.4f}",
            f"${safe_mean(costs_fail):.4f}",
            f"${safe_median(costs_fail):.4f}",
            str(len(costs_success)),
            str(len(costs_fail)),
            f"${sum(costs_all):.2f}",
        ])
    print_table(headers, rows)


if __name__ == "__main__":
    main()
