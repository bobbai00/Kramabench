"""Exhaustive audit: identical (answer, gold) pair scored differently. Which task,
which arm/rep, how often. Covers ALL answer types, all 30 dataflow runs + CA arms."""
import json, os, sys, re
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
from collections import defaultdict, Counter

ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
CA = {'CA_1kGuided':'CodeAgentSystemGpt5MiniProxyChars1kGuided','CA_5kGuided':'CodeAgentSystemGpt5MiniProxyChars5kGuided'}

atype = {}
for wl in ['archeology','astronomy','biomedical','environment','legal','wildfire']:
    for t in json.load(open(f'/home/bob/Desktop/bobflow/Kramabench/workload/{wl}.json')):
        if t.get('id'): atype[t['id']] = t.get('answer_type')

def norm(a):
    if a is None: return None
    s = re.sub(r'\s+', ' ', str(a).strip().lower())
    try: return repr(round(float(s.replace(',', '')), 9))
    except ValueError: return s

arms = []
for k, b in BASES.items():
    for i in range(6):
        a = f'{b}Replicate{i}'
        if os.path.isdir(f'{ROOT}/{a}'):
            s = kb.answer_scores(a)
            if len(s) >= 90: arms.append((k, i, a, s))
for k, b in CA.items():
    for suffix, i in [('', 0), ('Replicate1', 1), ('Replicate2', 2)]:
        a = b + suffix
        if os.path.isdir(f'{ROOT}/{a}'):
            s = kb.answer_scores(a)
            if len(s) >= 90: arms.append((k, i, a, s))

print(f"arms audited: {len(arms)}  (dataflow {sum(1 for x in arms if not x[0].startswith('CA'))}, code-agent {sum(1 for x in arms if x[0].startswith('CA'))})")

# group cells by (task, normalized answer)
groups = defaultdict(list)   # (task, nans) -> [(knob, rep, arm, score)]
for knob, rep, arm, sc in arms:
    for t, s in sc.items():
        try: ans = json.load(open(f'{ROOT}/{arm}/{t}/answer.json')).get('answer')
        except Exception: continue
        na = norm(ans)
        if na is None: continue
        groups[(t, na)].append((knob, rep, arm, s))

incons = {k: v for k, v in groups.items()
          if len({round(x[3], 6) for x in v}) > 1}
print(f"\n(task, identical-answer) groups: {len(groups)}   INCONSISTENTLY SCORED: {len(incons)}")

# per-task rollup
bytask = defaultdict(list)
for (t, na), v in incons.items():
    bytask[t].append((na, v))
print(f"tasks affected: {len(bytask)}")
print(f"answer types affected: {Counter(atype.get(t) for t in bytask)}")

# total cell-level impact
tot_cells = sum(len(v) for v in groups.values())
aff_cells = sum(len(v) for v in incons.values())
print(f"cells in inconsistent groups: {aff_cells} / {tot_cells} = {aff_cells/tot_cells:.1%}")

print("\n=== per task ===")
for t in sorted(bytask):
    print(f"\n{t}  [{atype.get(t)}]")
    for na, v in sorted(bytask[t], key=lambda x: -len(x[1])):
        hi = [x for x in v if x[3] >= 0.5]; lo = [x for x in v if x[3] < 0.5]
        print(f"  answer {na[:64]!r}: {len(v)} runs -> {len(hi)} scored high, {len(lo)} scored low ({len(lo)/len(v):.0%} discordant-low)")
        def fmt(rows): return ', '.join(sorted(f"{k}/r{r}" for k, r, _, _ in rows))
        print(f"     high: {fmt(hi)}")
        print(f"     low : {fmt(lo)}")

# is discordance arm-correlated (a real arm effect) or arm-random (judge noise)?
print("\n=== discordance by arm (are some arms systematically scored low?) ===")
cnt = Counter(); tot = Counter()
for (t, na), v in incons.items():
    for knob, rep, arm, s in v:
        tot[(knob, rep)] += 1
        if s < 0.5: cnt[(knob, rep)] += 1
rows = sorted(tot, key=lambda k: -tot[k])
for k in rows:
    print(f"  {k[0]:12s} rep{k[1]}  low {cnt[k]:2d}/{tot[k]:2d}")
