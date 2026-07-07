#!/bin/bash
set -u; cd ~/Desktop/bobflow/Kramabench; export PATH="$HOME/.bun/bin:$PATH"
C3=DataflowSystemGPT52DeltaWin3kCompressLean
C6=DataflowSystemGPT52DeltaWin6kCompressLean
C3_IDS="astronomy-hard-12 environment-hard-12 environment-hard-13 environment-hard-14 environment-hard-15 environment-hard-16 environment-hard-17 environment-hard-18 environment-hard-19 environment-hard-20"
C6_IDS="astronomy-hard-12 environment-hard-18 environment-hard-19 environment-hard-20"
echo "[lrec] $(date) start"
python kb.py tasks --sut $C3 --ids "$C3_IDS" --isolate --parallel --no-reeval --watchdog-min 8 > logs/lrec-3k.log 2>&1 &
P1=$!
python kb.py tasks --sut $C6 --ids "$C6_IDS" --isolate --parallel --no-reeval --watchdog-min 8 > logs/lrec-6k.log 2>&1 &
P2=$!
wait $P1 $P2
for S in $C3 $C6; do python kb.py reeval --sut $S >> logs/lrec-reeval.log 2>&1; done
echo "[lrec] $(date) DONE"
