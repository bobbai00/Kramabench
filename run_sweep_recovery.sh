#!/bin/bash
# Recover watchdog-killed / unrun tasks per SUT (isolated so a hang can't kill
# siblings; parallel to bound wall-time). Then reeval all 4 to sync scores.
set -u
cd ~/Desktop/bobflow/Kramabench
mkdir -p logs
export PATH="$HOME/.bun/bin:$PATH"

C3=DataflowSystemGPT52DeltaWin3kCompress
S3=DataflowSystemGPT52DeltaWin3kSliding
C6=DataflowSystemGPT52DeltaWin6kCompress
S6=DataflowSystemGPT52DeltaWin6kSliding

C3_IDS="astronomy-hard-10 astronomy-hard-11 astronomy-hard-12 astronomy-hard-8 astronomy-hard-9 environment-easy-6 environment-hard-11 environment-hard-12 environment-hard-13 environment-hard-14 environment-hard-15 environment-hard-16 environment-hard-17 environment-hard-18 environment-hard-19 environment-hard-20 environment-hard-7 environment-hard-8 environment-hard-9"
S3_IDS="astronomy-hard-10 astronomy-hard-11 astronomy-hard-12 astronomy-hard-8 astronomy-hard-9 environment-hard-10 environment-hard-11 environment-hard-12 environment-hard-13 environment-hard-14 environment-hard-15 environment-hard-16 environment-hard-17 environment-hard-18 environment-hard-19 environment-hard-20 environment-hard-9"
C6_IDS="astronomy-hard-8 astronomy-hard-9 astronomy-hard-10 astronomy-hard-11 astronomy-hard-12"
S6_IDS="astronomy-hard-9 astronomy-hard-10 astronomy-hard-11 astronomy-hard-12"

echo "[recover] $(date) — 3k pair missing tasks (isolated, parallel)"
python kb.py tasks --sut $C3 --ids "$C3_IDS" --isolate --parallel --no-reeval --watchdog-min 8 > logs/recover-3k-compress.log 2>&1 &
P1=$!
python kb.py tasks --sut $S3 --ids "$S3_IDS" --isolate --parallel --no-reeval --watchdog-min 8 > logs/recover-3k-sliding.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[recover] $(date) — 3k done"

echo "[recover] $(date) — 6k pair missing astronomy (isolated, parallel)"
python kb.py tasks --sut $C6 --ids "$C6_IDS" --isolate --parallel --no-reeval --watchdog-min 8 > logs/recover-6k-compress.log 2>&1 &
P3=$!
python kb.py tasks --sut $S6 --ids "$S6_IDS" --isolate --parallel --no-reeval --watchdog-min 8 > logs/recover-6k-sliding.log 2>&1 &
P4=$!
wait $P3 $P4
echo "[recover] $(date) — 6k done"

echo "[recover] $(date) — reeval (sequential)"
for S in $C3 $S3 $C6 $S6; do echo "[recover] reeval $S"; python kb.py reeval --sut $S >> logs/recover-reeval.log 2>&1; done
echo "[recover] $(date) — RECOVERY DONE"
