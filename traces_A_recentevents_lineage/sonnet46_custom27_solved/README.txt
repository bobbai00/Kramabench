Sonnet-4.6 DataflowAgent — SOLVED tasks (23).

System: DataflowSystemSonnet46Annot2LineageThoughtReplay
  Sonnet-4.6 both-flags (latest, flow2/lineage + thoughtReplay ON K=5, max_steps=20) on the user's 27-task list. SOLVED = best-of-2 (one retry of the 7 initial failures).

Passing = native KramaBench score > 0.9, unmodified evaluate suite, REAL gpt-4o-mini judge
(agent=claude-sonnet-4.6 via proxy->Anthropic).

Tasks (task_id  metric=score):
  archeology-easy-4        success=1.0
  astronomy-easy-1         success=1.0
  astronomy-easy-5         success=1.0
  biomedical-easy-6        success=1.0
  biomedical-easy-9        success=1.0
  biomedical-hard-3        success=1.0
  environment-easy-2       f1=1.0
  environment-hard-11      success=1.0
  environment-hard-13      success=1.0
  environment-hard-14      success=1.0
  legal-easy-13            success=1.0
  legal-easy-25            llm_paraphrase=1.0
  legal-easy-27            success=1.0
  legal-hard-14            f1=1.0
  legal-hard-15            success=1.0
  legal-hard-16            success=1.0
  legal-hard-29            success=1.0
  wildfire-easy-8          success=1.0
  wildfire-hard-17         rae_score=1.0
  wildfire-hard-20         success=1.0
  wildfire-hard-5          success=1.0
  wildfire-hard-6          rae_score=1.0
  wildfire-hard-7          llm_paraphrase=1.0

Each task dir: prompt.txt, react_steps.json, workflow.json, answer.json,
ground_truth.json, response.txt, stats.json, config.json, evaluation.json
