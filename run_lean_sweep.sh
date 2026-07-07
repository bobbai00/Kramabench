#!/bin/bash
# Lean-deck iteration: gpt-5.2 compress with capped deck stats-cols + rows, at
# 3k/6k. Compared against the recorded heavy-deck compress + sliding baselines.
set -u
cd ~/Desktop/bobflow/Kramabench
mkdir -p logs
export PATH="$HOME/.bun/bin:$PATH"

C3=DataflowSystemGPT52DeltaWin3kCompressLean
C6=DataflowSystemGPT52DeltaWin6kCompressLean

echo "[lean] $(date) — 3k + 6k compress-lean concurrent"
python kb.py run --sut $C3 --parallel --no-reeval --watchdog-min 8 > logs/lean-3k-compress.log 2>&1 &
P1=$!
python kb.py run --sut $C6 --parallel --no-reeval --watchdog-min 8 > logs/lean-6k-compress.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[lean] $(date) — runs done"

for S in $C3 $C6; do echo "[lean] reeval $S"; python kb.py reeval --sut $S >> logs/lean-reeval.log 2>&1; done
echo "[lean] $(date) — LEAN SWEEP DONE"
