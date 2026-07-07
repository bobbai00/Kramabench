gpt-5-mini DataflowAgent — SOLVED tasks (7/38 of the Haiku both-flags FAILED set).

System: DataflowSystemGPT5MiniAnnot2Lineage
  gpt-5-mini, context_mode=latest, flow_level=2 (lineage), data_level=2, thoughtReplay=OFF, char 1000/3000

These are the tasks this system got RIGHT among the 38 tasks that the Haiku-4.5
both-flags system (DataflowSystemHaiku45Annot2LineageThoughtReplay) failed.
Passing = native KramaBench score > 0.9, scored by the unmodified evaluate suite
with the REAL gpt-4o-mini judge (OPENAI_BASE_URL unset; agent=gpt-5-mini via proxy).

Tasks (task_id  metric=score):
  biomedical-hard-7        success=1.0
  environment-easy-3       success=1.0
  environment-hard-18      llm_paraphrase=1.0
  environment-hard-8       success=1.0
  legal-hard-18            success=1.0
  wildfire-easy-9          rae_score=0.9077
  wildfire-hard-12         llm_paraphrase=1.0

Each task dir: prompt.txt, react_steps.json, workflow.json, answer.json,
ground_truth.json, response.txt, stats.json, config.json, evaluation.json
