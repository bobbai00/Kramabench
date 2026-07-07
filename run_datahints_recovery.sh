#!/usr/bin/env bash
# Recovery probe: run the data-quality/multi-header failures + the 2 loader-guard
# -blocked tasks under DataflowSystemGPT52LatestColumnStatsDataHints (guard now
# removed globally; data_hints ON). Isolated + parallel, then rescore + report.
set -u
cd /home/bob/Desktop/bobflow/Kramabench
set -a; . ./.env; set +a
PY=.venv/bin/python
SUT=DataflowSystemGPT52LatestColumnStatsDataHints
MAXPAR=6; TIMEOUT=480
TS=$(date +%Y%m%d_%H%M%S); LOGD="logs/datahints-recovery-$TS"; mkdir -p "$LOGD"
# 9 data-quality/multi-header candidates + 2 loader-guard-blocked (biomedical-hard-4, archeology-hard-1)
IDS="environment-hard-8 environment-hard-10 environment-hard-11 environment-hard-13 environment-hard-16 environment-easy-3 archeology-hard-2 archeology-hard-5 archeology-easy-11 biomedical-hard-4 archeology-hard-1"
echo "[dh] running $(wc -w <<<"$IDS") tasks under $SUT"
run_task(){ local tid="$1" wl="${1%%-*}"; echo "[dh] start $tid"; \
  timeout "$TIMEOUT" $PY -u evaluate.py --sut "$SUT" --workload "$wl" --task_id "$tid" \
    --no_pipeline_eval --verbose --use_truth_subset > "$LOGD/$tid.log" 2>&1; \
  echo "[dh] done  $tid (exit $?)"; }
for tid in $IDS; do while [ "$(jobs -r|wc -l)" -ge "$MAXPAR" ]; do sleep 3; done; run_task "$tid" & done
wait
echo "[dh] rescoring"
$PY kb.py reeval --sut "$SUT" >/dev/null 2>&1
echo "============ RECOVERY REPORT (data_hints on, guard removed) ============"
$PY - "$SUT" "$IDS" <<'PYEOF'
import json, sys, subprocess, os
sut=sys.argv[1]; ids=sys.argv[2].split()
r=subprocess.run([".venv/bin/python","scripts/list_failed_tasks.py","--sut",sut,"--json"],capture_output=True,text=True)
try: d=json.loads(r.stdout)
except Exception: d={}
failed={ (t.get("task_id") or t.get("id")):(t.get("score") or 0) for wl in d.values() for t in wl }
base=f"system_scratch/{sut}"
npass=0
for tid in ids:
    ap=f"{base}/{tid}/answer.json"; ans=json.load(open(ap)).get("answer") if os.path.exists(ap) else None
    if tid in failed:   # in failed list => score<1
        s=failed[tid]; tag="PARTIAL" if s>0 else "FAIL"
    else:               # not in failed list => passed (score>=1)
        tag="PASS"; npass+=1
    print(f"  {tid:22} {tag:8} answer={str(ans)[:30]}")
print(f"=> RECOVERED (now pass): {npass}/{len(ids)}")
PYEOF
echo "[dh] DONE"
