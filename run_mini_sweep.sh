#!/bin/bash
# gpt-5-mini cross-model replication of the compaction sweep: compress vs sliding
# @ 3k/6k. Pairs concurrent (--no-reeval), reeval sequential at the end.
set -u
cd ~/Desktop/bobflow/Kramabench
mkdir -p logs
export PATH="$HOME/.bun/bin:$PATH"

C3=DataflowSystemGPT5MiniDeltaWin3kCompress
S3=DataflowSystemGPT5MiniDeltaWin3kSliding
C6=DataflowSystemGPT5MiniDeltaWin6kCompress
S6=DataflowSystemGPT5MiniDeltaWin6kSliding

echo "[mini] $(date) — 3k pair concurrent"
python kb.py run --sut $C3 --parallel --no-reeval --watchdog-min 8 > logs/mini-3k-compress.log 2>&1 &
P1=$!
python kb.py run --sut $S3 --parallel --no-reeval --watchdog-min 8 > logs/mini-3k-sliding.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[mini] $(date) — 3k done"

echo "[mini] $(date) — 6k pair concurrent"
python kb.py run --sut $C6 --parallel --no-reeval --watchdog-min 8 > logs/mini-6k-compress.log 2>&1 &
P3=$!
python kb.py run --sut $S6 --parallel --no-reeval --watchdog-min 8 > logs/mini-6k-sliding.log 2>&1 &
P4=$!
wait $P3 $P4
echo "[mini] $(date) — 6k done"

echo "[mini] $(date) — reeval"
for S in $C3 $S3 $C6 $S6; do echo "[mini] reeval $S"; python kb.py reeval --sut $S >> logs/mini-reeval.log 2>&1; done
echo "[mini] $(date) — MINI SWEEP DONE"
