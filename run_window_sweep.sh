#!/bin/bash
# Context-window compaction sweep: gpt-5.2 DELTA, compress vs sliding @ 3k/6k.
# Pairs run concurrently (--no-reeval → per-SUT scratch only, no shared-cache
# race); reeval runs sequentially at the end to sync results/aggregated_results.csv.
set -u
cd ~/Desktop/bobflow/Kramabench
mkdir -p logs
export PATH="$HOME/.bun/bin:$PATH"

C3=DataflowSystemGPT52DeltaWin3kCompress
S3=DataflowSystemGPT52DeltaWin3kSliding
C6=DataflowSystemGPT52DeltaWin6kCompress
S6=DataflowSystemGPT52DeltaWin6kSliding

echo "[sweep] $(date) — 3k pair (compress + sliding) concurrent"
python kb.py run --sut $C3 --parallel --no-reeval --watchdog-min 8 > logs/sweep-3k-compress.log 2>&1 &
P1=$!
python kb.py run --sut $S3 --parallel --no-reeval --watchdog-min 8 > logs/sweep-3k-sliding.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[sweep] $(date) — 3k pair done"

echo "[sweep] $(date) — 6k pair (compress + sliding) concurrent"
python kb.py run --sut $C6 --parallel --no-reeval --watchdog-min 8 > logs/sweep-6k-compress.log 2>&1 &
P3=$!
python kb.py run --sut $S6 --parallel --no-reeval --watchdog-min 8 > logs/sweep-6k-sliding.log 2>&1 &
P4=$!
wait $P3 $P4
echo "[sweep] $(date) — 6k pair done"

echo "[sweep] $(date) — reeval (sequential)"
for S in $C3 $S3 $C6 $S6; do
  echo "[sweep] reeval $S"
  python kb.py reeval --sut $S >> logs/sweep-reeval.log 2>&1
done
echo "[sweep] $(date) — SWEEP DONE"
