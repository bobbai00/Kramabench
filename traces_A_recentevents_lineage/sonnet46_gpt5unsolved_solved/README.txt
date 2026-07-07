Sonnet-4.6 DataflowAgent — SOLVED tasks (4).

System: DataflowSystemSonnet46Annot2LineageThoughtReplay
  Sonnet-4.6 both-flags on the 29 tasks (of the 38 Haiku-failed) that NEITHER gpt-5-mini config solved. SOLVED = tasks Sonnet recovered that gpt-5-mini could not.

Passing = native KramaBench score > 0.9, unmodified evaluate suite, REAL gpt-4o-mini judge
(agent=claude-sonnet-4.6 via proxy->Anthropic).

Tasks (task_id  metric=score):
  archeology-easy-11       success=1.0
  astronomy-easy-4         llm_paraphrase=1.0
  astronomy-hard-9         success=1.0
  legal-hard-2             llm_paraphrase=1.0

Each task dir: prompt.txt, react_steps.json, workflow.json, answer.json,
ground_truth.json, response.txt, stats.json, config.json, evaluation.json
