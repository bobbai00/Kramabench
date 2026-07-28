"""Divergence localization: same knob, pass vs fail rep, compare per-step op summaries."""
import json, os, sys, re, hashlib
sys.path.insert(0, '/home/bob/Desktop/bobflow/Kramabench')
import kb
from collections import Counter

SC = '/tmp/claude-1000/-home-bob-Desktop-bobflow-Kramabench/248950d7-0055-4e00-8db5-ee9cc795136d/scratchpad'
M = json.load(open(f'{SC}/score_matrix.json'))
classes = json.load(open(f'{SC}/classes.json'))
tasks = sorted(M)
BASES = {
 'anchor_D1k':'DataflowSystemGPT5MiniDelta1kSchemaOnly','C1_D5k':'DataflowSystemGPT5MiniDelta5kSchemaOnly',
 'C2_DS1k':'DataflowSystemGPT5MiniDeltaStats1kD2','C5_DS2k':'DataflowSystemGPT5MiniDeltaStats2kD2',
 'C4_DS5k':'DataflowSystemGPT5MiniDeltaStats5kD2','C3_L1kCode':'DataflowSystemGPT5MiniLatest1kCodeInSnap',
 'C6_LS1k':'DataflowSystemGPT5MiniLatestStats1kD2'}
ROOT = '/home/bob/Desktop/bobflow/Kramabench/system_scratch'

# cache scored arms per knob
arm_scores = {}
for k, b in BASES.items():
    for i in range(6):
        a = f'{b}Replicate{i}'
        if os.path.isdir(f'{ROOT}/{a}'):
            s = kb.answer_scores(a)
            if len(s) >= 90:
                arm_scores[(k, a)] = s

def step_sigs(arm, t):
    """ordered per-agent-step signature: sorted operatorIds + summary hashes in that step"""
    p = f'{ROOT}/{arm}/{t}/react_steps.json'
    try:
        doc = json.load(open(p))
    except Exception:
        return None
    sigs = []
    for s in doc.get('steps', []):
        if s.get('role') != 'agent':
            continue
        tc = s.get('toolCalls') or ''
        if not isinstance(tc, str):
            tc = str(tc)
        ops = re.findall(r"'operatorId': '([^']+)'", tc)
        sums = re.findall(r"'summary': [\"']((?:[^\"'\\\\]|\\\\.){0,120})", tc)
        if ops or sums:
            sig = hashlib.md5(('|'.join(ops) + '#' + '|'.join(sums)).encode()).hexdigest()[:8]
            sigs.append((tuple(ops), sig))
    return sigs

loc = Counter(); details = []
for t in tasks:
    if classes[t] != 'coin':
        continue
    done = False
    for k in BASES:
        if done: break
        karms = [(a, s) for (kk, a), s in arm_scores.items() if kk == k and t in s]
        ps = [a for a, s in karms if s[t] >= 0.5]
        fs = [a for a, s in karms if s[t] < 0.5]
        if not ps or not fs:
            continue
        s1, s2 = step_sigs(ps[0], t), step_sigs(fs[0], t)
        if not s1 or not s2:
            continue
        n = min(len(s1), len(s2)); div = n
        for j in range(n):
            if s1[j][1] != s2[j][1]:
                div = j; break
        frac = div / max(len(s1), len(s2), 1)
        bucket = 'step0' if div == 0 else ('early<1/3' if frac < 1/3 else 'mid' if frac < 2/3 else 'late>=2/3')
        loc[bucket] += 1
        details.append((t, k, div, len(s1), len(s2)))
        done = True

print("first-divergence position (1 same-config pass/fail pair per coin task):", dict(loc))
print(f"pairs analyzed: {len(details)}")
for t, k, div, l1, l2 in details:
    print(f"  {t:26s} {k:10s} diverge@step{div}  pass_steps={l1} fail_steps={l2}")
