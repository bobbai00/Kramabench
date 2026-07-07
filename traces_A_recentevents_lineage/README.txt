Texera DataflowAgent (Haiku-4.5) — traces for the system with BOTH recent-events AND lineage ON.

System: DataflowSystemHaiku45Annot2LineageThoughtReplay
  context_mode=latest, flow_level=2 (lineage), data_level=2, thoughtReplay=ON (recent-events), char 1000/3000

Filter: kept only PASSING tasks — native KramaBench score > 0.9, scored by the unmodified
evaluate suite with the REAL gpt-4o-mini judge (evaluate_dataflow_system.sh --use_system_cache).
66 of 104 tasks kept: legal 25, environment 13, wildfire 15, archeology 4, astronomy 3, biomedical 6.

Each task dir: prompt.txt, react_steps.json, workflow.json, answer.json, ground_truth.json, stats.json, config.json
Also: system_prompt.md (verbatim CODE-mode system prompt), tools.json (the 2 core tools, OpenAI format)

----------------------------------------------------------------------
ADDED: gpt-5-mini solved-task folders (peer comparison)
----------------------------------------------------------------------
Two extra top-level folders hold the tasks that gpt-5-mini SOLVED among the
38 tasks the Haiku both-flags system above FAILED:

  gpt5mini_replayON_solved7/   DataflowSystemGPT5MiniAnnot2LineageThoughtReplay
                               (latest, flow2/lineage, data2, thoughtReplay=ON K=5)
                               7 solved: archeology-hard-7, astronomy-easy-6,
                               environment-easy-3, environment-hard-18,
                               environment-hard-8, legal-hard-18, wildfire-easy-9

  gpt5mini_replayOFF_solved7/  DataflowSystemGPT5MiniAnnot2Lineage
                               (latest, flow2/lineage, data2, thoughtReplay=OFF)
                               7 solved: biomedical-hard-7, environment-easy-3,
                               environment-hard-18, environment-hard-8,
                               legal-hard-18, wildfire-easy-9, wildfire-hard-12

Shared by both (5): environment-easy-3, environment-hard-18, environment-hard-8,
legal-hard-18, wildfire-easy-9.  ON-only: archeology-hard-7, astronomy-easy-6.
OFF-only: biomedical-hard-7, wildfire-hard-12.

Same passing bar and judge (native KramaBench score > 0.9, real gpt-4o-mini).
Each folder also has manifest.json (per-task metric+score) and README.txt.

----------------------------------------------------------------------
ADDED: Sonnet-4.6 solved-task folders
----------------------------------------------------------------------
System: DataflowSystemSonnet46Annot2LineageThoughtReplay
  claude-sonnet-4.6, context_mode=latest, flow_level=2 (lineage), data_level=2,
  thoughtReplay=ON (recent-events, K=5), max_steps=20, char 1000/3000.

  sonnet46_custom27_solved/        23 solved of the user's 27-task list
                                   (best-of-2: 3 of the initial 7 failures recovered
                                    on a 2nd attempt -> environment-hard-11, legal-hard-16,
                                    wildfire-hard-5).

  sonnet46_gpt5unsolved_solved/    4 solved of the 29 tasks (within the 38 Haiku-failed)
                                   that NEITHER gpt-5-mini config could solve:
                                   archeology-easy-11, astronomy-easy-4, astronomy-hard-9,
                                   legal-hard-2 — i.e. tasks Sonnet recovered beyond gpt-5-mini.

Same passing bar/judge as the rest of this bundle (native score > 0.9, real gpt-4o-mini).
Each folder has manifest.json (per-task metric+score) and README.txt.
