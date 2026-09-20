#!/usr/bin/env bash
# 2x2 layout probe: DELTA vs opBlock, evidence off/on, 5 tasks each.
set -u
TASKS="archeology-easy-3 biomedical-easy-2 environment-hard-8 legal-easy-3 wildfire-hard-4"
ARMS="DataflowSystemTerraLayoutDeltaPlain20260919 \
      DataflowSystemTerraLayoutDeltaEvidence20260919 \
      DataflowSystemTerraLayoutOpBlockPlain20260919 \
      DataflowSystemTerraLayoutOpBlockEvidence20260919"
for arm in $ARMS; do
  echo "=========================================================="
  echo "ARM $arm  $(date +%H:%M:%S)"
  echo "=========================================================="
  .venv/bin/python kb.py tasks --sut "$arm" --ids "$TASKS" --isolate 2>&1 | tail -20
done
echo "ALL ARMS DONE $(date +%H:%M:%S)"
