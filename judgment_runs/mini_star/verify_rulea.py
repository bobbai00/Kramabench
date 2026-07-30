#!/usr/bin/env python3
"""Verify Rule A actually shaped the rendered context: sources should carry a
wide sample + stats; interior/terminal ops should be lean."""
import json, re, sys

sut = sys.argv[1] if len(sys.argv) > 1 else "DataflowSystemGPT5MiniA1RolePolicyReplicate1"
task = sys.argv[2] if len(sys.argv) > 2 else "legal-easy-11"
base = f"system_scratch/{sut}/{task}"

cfg = json.load(open(f"{base}/config.json")).get("agent_settings", {})
print("role_policy_config in config.json:", cfg.get("role_policy_config", "ABSENT"))

doc = json.load(open(f"{base}/react_steps.json"))
steps = [s for s in doc.get("steps", []) if s.get("inputMessages")]
if not steps:
    print("no inputMessages"); sys.exit(0)
ctx = "\n".join(str(m.get("content", "")) for m in steps[-1]["inputMessages"])

pat = re.compile(r"### Operator `([^`]+)` \(([^)]+)\)(.*?)(?=\n### |\Z)", re.S)
print(f"\n{'operator':<24}{'type':<18}{'stats':>7}{'profile':>9}{'tsv rows':>10}{'chars':>8}")
print("-" * 78)
for m in pat.finditer(ctx):
    op, typ, body = m.group(1), m.group(2), m.group(3)
    stats = "Column Schema and stats" in body
    prof = "Output Table profile" in body
    # count sample rows: lines starting with a digit index then a tab
    rows = len(re.findall(r"\n\s*\d+\t", body))
    print(f"{op:<24}{typ:<18}{str(stats):>7}{str(prof):>9}{rows:>10}{len(body):>8}")
