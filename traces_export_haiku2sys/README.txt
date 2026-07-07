Texera DataflowAgent — Haiku-4.5 traces for tasks SOLVED BY BOTH SYSTEMS.

Filter: kept only tasks whose best-of-2 primary score is > 0.9 in BOTH systems
(failed = score <= 0.9 in either system, excluded for both). 61 of 104 tasks kept.

Systems (both LATEST + lineage(flow_level=2) + data_level=2; differ only in recent-events):
  DataflowSystemHaiku45Annot2LineageThoughtReplay  = recent-events (thoughtReplay) ON
  DataflowSystemHaiku45Annot2Lineage               = recent-events OFF

Kept by workload: legal 19, environment 13, wildfire 16, archeology 4, astronomy 3, biomedical 6.
Each task dir: prompt.txt, react_steps.json, workflow.json, answer.json, ground_truth.json, stats.json, config.json.

Also included:
  system_prompt.md  - verbatim CODE-mode system prompt sent to the model
  tools.json        - the two core tools (createOrModifyOperator, deleteOperator) in OpenAI tools format
