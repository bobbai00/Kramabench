#!/usr/bin/env python3
"""
LakeQA Phase A preparation: pick the stratified 24-task subset and download
its data files from the public LakeQA S3 bucket into the shared data lake dir.

Phase A = oracle-style adoption (each task is handed its own files locally,
mirroring KramaBench oracle mode) — measures the dataflow agent's
reasoning/pipeline half before any search-tool investment.

Subset design (see judgment_runs analysis): table-dominant tasks only
(wikipedia share <= 1/3 of files), no .pdf/.docx sources, k in {3,4,5}
(reasoning hops — 120/135 of mini), 8 per k, round-robin over d (breadth) so
file-count difficulty is spread. Deterministic given the task set.

Usage (combined venv per dataflow-agent pyproject `lakeqa` extra):
  ~/Desktop/bobflow/dataflow-agent/.venv/bin/python lakeqa/prepare.py            # select + size manifest
  ~/Desktop/bobflow/dataflow-agent/.venv/bin/python lakeqa/prepare.py --download # + fetch files
"""

import argparse
import concurrent.futures as cf
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

KB = Path(__file__).resolve().parent.parent
LAKEQA_REPO = Path(os.environ.get("LAKEQA_REPO", Path.home() / "Desktop/bobflow/lakeqa"))
DATA_DIR = KB / "data/lakeqa/input"
SUBSET_PATH = Path(__file__).resolve().parent / "subset_24.json"
BUCKET = "lakeqa-yc4103-datalake"
PER_K = {3: 8, 4: 8, 5: 8}


def load_mini_tasks():
    tasks = []
    for f in sorted(glob.glob(str(LAKEQA_REPO / "lakeqa_mini/*/*.json"))):
        t = json.load(open(f))
        g = Path(f).parent.name  # k-4-d-5
        t["_group"] = g
        t["_k"] = int(g.split("-")[1])
        t["_d"] = int(g.split("-")[3])
        t["_file"] = str(Path(f).relative_to(LAKEQA_REPO))
        tasks.append(t)
    return tasks


def eligible(t):
    ds = t.get("datasets_used", [])
    if not ds or t["_k"] not in PER_K:
        return False
    if any(x.lower().endswith((".pdf", ".docx")) for x in ds):
        return False
    wiki = sum(1 for x in ds if x.startswith("wikipedia/"))
    return wiki / len(ds) <= 1 / 3


def select_subset(tasks, task_mb, max_task_mb=None):
    """8 per k, round-robin over d within k (spreads breadth), deterministic.
    With max_task_mb, tasks whose total source bytes exceed the cap are skipped
    and backfilled from the same k stratum (keeps the run executable: the mini
    set has a handful of multi-GB data.gov dumps that would dominate download
    and execution time — Phase A is a reasoning smoke, not an I/O test)."""
    pool = [t for t in tasks if eligible(t)]
    if max_task_mb is not None:
        pool = [t for t in pool if task_mb(t) <= max_task_mb]
    chosen = []
    for k, quota in PER_K.items():
        by_d = defaultdict(list)
        for t in sorted((t for t in pool if t["_k"] == k), key=lambda t: t["question_id"]):
            by_d[t["_d"]].append(t)
        ds = sorted(by_d)
        i = 0
        while sum(1 for c in chosen if c["_k"] == k) < quota and any(by_d.values()):
            d = ds[i % len(ds)]
            if by_d[d]:
                chosen.append(by_d[d].pop(0))
            i += 1
    return chosen


def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch the files (default: select + size only)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-task-mb", type=float, default=500.0,
                    help="skip tasks whose sources exceed this, backfill same-k (default 500; 0 = no cap)")
    a = ap.parse_args()

    tasks = load_mini_tasks()
    s3 = s3_client()

    def head(key):
        try:
            return key, s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        except Exception:
            return key, -1

    # Size the whole eligible pool first so the cap can steer selection.
    pool_files = sorted({f for t in tasks if eligible(t) for f in t["datasets_used"]})
    sizes = {}
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for key, n in ex.map(head, pool_files):
            sizes[key] = n
    task_mb = lambda t: sum(max(0, sizes.get(f, 0)) for f in t["datasets_used"]) / 1e6

    subset = select_subset(tasks, task_mb, a.max_task_mb or None)
    files = sorted({f for t in subset for f in t["datasets_used"]})
    print(f"mini tasks {len(tasks)} -> eligible {sum(1 for t in tasks if eligible(t))} "
          f"-> subset {len(subset)} (cap {a.max_task_mb} MB/task)")
    print(f"groups: {sorted(t['_group'] + ':' + t['question_id'] for t in subset)}")
    print(f"distinct files: {len(files)}")
    missing = [k for k in files if sizes.get(k, -1) < 0]
    total = sum(sizes[k] for k in files if sizes.get(k, 0) > 0)
    print(f"total download: {total / 1e6:,.1f} MB; missing on S3: {len(missing)}")
    for k in missing:
        print(f"  MISSING {k}")
    per_task = [(t["question_id"], sum(sizes.get(f, 0) for f in t["datasets_used"]) / 1e6, t["_group"]) for t in subset]
    for qid, mb, g in sorted(per_task, key=lambda x: -x[1]):
        print(f"  {qid}  {g:9s} {mb:8.1f} MB")

    manifest = [{k: t[k] for k in ("question_id", "question", "answer", "datasets_used", "_group", "_k", "_d", "_file")}
                for t in subset]
    json.dump(manifest, open(SUBSET_PATH, "w"), indent=1)
    print(f"[manifest] {SUBSET_PATH.relative_to(KB)}")

    if not a.download:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(key):
        dst = DATA_DIR / key
        if dst.exists() and dst.stat().st_size == sizes.get(key, -2):
            return key, "cached"
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(dst))
        return key, "ok"

    done = 0
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for key, st in ex.map(fetch, [f for f in files if sizes.get(f, -1) >= 0]):
            done += 1
            if done % 50 == 0 or st != "ok":
                print(f"  [{done}/{len(files)}] {st} {key}")
    print(f"downloaded {done} files -> {DATA_DIR.relative_to(KB)}")


if __name__ == "__main__":
    main()
