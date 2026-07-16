"""
System module initialization.
This module provides system-level functionality and configurations.
"""
from .baseline_example import ExampleBaselineSystem
from .dataflow_system import (
    # Core
    DataflowSystem,
    DataflowSystemHaiku45,
    DataflowSystemGPT5Mini,
    # Thesis converge stacks (DataFlow-vs-Script, both models)
    DataflowSystemGPT52LatestSchemaConverge,
    # gpt-5.4 converge pair — Latest vs Delta context A/B
    DataflowSystemGPT54LatestSchemaConverge,
    DataflowSystemGPT54LatestSchemaConvergeLevels,
    DataflowSystemGPT54LatestColumnStats,
    DataflowSystemGPT54LatestColumnStatsOnly,
    DataflowSystemGPT52LatestColumnStats,
    DataflowSystemGPT52LatestColumnStatsOnly,
    DataflowSystemGPT52LatestColumnStatsDataHints,
    # old-branch replication probe (rich example + old-style error rendering)
    DataflowSystemGPT52LatestColumnStatsOldStyle,
)
from .dataflow_system_old import (
    # OLD STACK (fe917396a) via the era-matched client — the true A/B arm
    DataflowSystemOldStackGPT52NoActionDetail,
)
from .dataflow_system import (
    DataflowSystemGPT52DeltaColumnStatsDataHints,
    DataflowSystemGPT52LatestStats2k,
    DataflowSystemGPT52LatestStats5k,
    DataflowSystemGPT52DeltaStats2k,
    DataflowSystemGPT52DeltaStats5k,
    DataflowSystemGPT52Latest3kSchemaOnly,
    DataflowSystemGPT52Delta3kSchemaOnly,
    DataflowSystemGPT52Delta1kSchemaOnly,
    DataflowSystemGPT52Delta5kSchemaOnlyRawProbe,
    DataflowSystemGPT52Latest3kSchemaOnlyRawProbe,
    DataflowSystemGPT52Latest5kSchemaOnly,
    DataflowSystemGPT52Delta5kSchemaOnly,
    DataflowSystemGPT52Latest7kSchemaOnly,
    DataflowSystemGPT52Delta7kSchemaOnly,
    DataflowSystemGPT52LatestStats3kD2,
    DataflowSystemGPT52LatestStats3kD2SmallTableControl,
    DataflowSystemGPT52LatestStats3kD2FrontierDecay,
    DataflowSystemGPT52DeltaStats3kD2,
    DataflowSystemGPT52DeltaStats1kD2,
    DataflowSystemGPT52DeltaStats3kD2RawProbe,
    DataflowSystemGPT52DeltaStats3kD2FoldControl,
    DataflowSystemGPT52DeltaStats3kD2FoldResolved,
    DataflowSystemGPT52LatestStats3kD2ProbeRetire,
    DataflowSystemGPT52LatestStats3kD2Explore,
    DataflowSystemGPT52DeltaStats3kD2Explore,
    DataflowSystemGPT52LatestStats3kD2ExploreList,
    DataflowSystemGPT52DeltaStats3kD2ExploreList,
    DataflowSystemGPT52DeltaStats5kD2FreshControl,
    DataflowSystemGPT52DeltaStats5kD2RenderPrefs,
    DataflowSystemGPT52LatestStats3kD2Lean3,
    DataflowSystemGPT52LatestStats3kD2Lean3Pull,
    DataflowSystemGPT52LatestStats5kD2,
    DataflowSystemGPT52DeltaStats5kD2,
    DataflowSystemGPT52LatestStats7kD2,
    DataflowSystemGPT52DeltaStats7kD2,
    DataflowSystemGPT52LatestStats10kD2,
    DataflowSystemGPT52DeltaStats10kD2,
    DataflowSystemGPT5MiniLatest3kSchemaOnly,
    DataflowSystemGPT5MiniDelta3kSchemaOnly,
    DataflowSystemGPT5MiniLatestStats3kD2,
    DataflowSystemGPT5MiniDeltaStats3kD2,
    DataflowSystemGPT52DeltaStats5kCompact,
    DataflowSystemGPT52DeltaStats5kCompactEC,
    # parallel-tool-calls ablation (parallel OFF, window OFF, else identical)
    DataflowSystemGPT52DeltaColumnStatsDataHintsNoParallel,
    DataflowSystemGPT54DeltaSchemaConverge,
    # context-window compaction sweep (gpt-5.2 DELTA): compress vs sliding @ 3k/6k
    DataflowSystemGPT52DeltaWin3kCompress,
    DataflowSystemGPT52DeltaWin3kSliding,
    DataflowSystemGPT52DeltaWin6kCompress,
    DataflowSystemGPT52DeltaWin6kSliding,
    # lean-deck iteration (capped deck stats cols + rows)
    DataflowSystemGPT52DeltaWin3kCompressLean,
    DataflowSystemGPT52DeltaWin6kCompressLean,
    # prompt-aware iteration (DELTA prompt describes the compaction deck)
    DataflowSystemGPT52DeltaWin3kCompressPromptAware,
    # gpt-5-mini cross-model replication of the compaction sweep
    DataflowSystemGPT5MiniDeltaWin3kCompress,
    DataflowSystemGPT5MiniDeltaWin3kSliding,
    DataflowSystemGPT5MiniDeltaWin6kCompress,
    DataflowSystemGPT5MiniDeltaWin6kSliding,
    # hybrid: latest core + selective reinjection (error-reflection [+ thought-replay])
    DataflowSystemGPT54LatestSchemaConvergeErrorReflect,
    DataflowSystemGPT54LatestSchemaConvergeReinject,
    # idea-1: latest core + full Thought/Action/Observation `# Agent Turns` section
    DataflowSystemGPT54LatestSchemaConvergeAgentTurns,
    DataflowSystemGPT5MiniLatestSchemaConverge,
    # Canonical level-based + the validated levers
    DataflowSystemGPT5MiniLatestSchemaConvergeLevels,      # flow_level/data_level config
    DataflowSystemGPT5MiniLatestSchemaConvergeTableStruct, # the accuracy win (data_level=2)
    DataflowSystemGPT5MiniLatestSchemaConvergeFewShot,     # W2 cost win
    DataflowSystemGPT5MiniLatestSchemaConvergeCap20,       # #31 step-cap (ACT lever)
    DataflowSystemGPT5MiniLatestSchemaConvergeThoughtReplay, # SELECT-side reasoning reinjection (thoughtReplay)
    # Haiku-4.5 2x2 study: thought_replay {off,on} x data-lineage flow {off,on}; data annotation L2 fixed
    DataflowSystemHaiku45Annot2,
    DataflowSystemHaiku45Annot2Lineage,
    DataflowSystemHaiku45Annot2LineageErrorReflect,
    DataflowSystemHaiku45Annot2ThoughtReplay,
    DataflowSystemHaiku45Annot2LineageThoughtReplay,
    DataflowSystemHaiku45DeltaLineageReplay,  # DELTA + lineage (thoughts via DELTA inline)
    # gpt-5-mini peers of the Haiku both-flags / lineage-only pair (replay K=5)
    DataflowSystemGPT5MiniAnnot2LineageThoughtReplay,
    DataflowSystemGPT5MiniAnnot2Lineage,
    # gpt-5-mini lineage + error-reflection (replay off) — churn-loop A/B
    DataflowSystemGPT5MiniAnnot2LineageErrorReflect,
    DataflowSystemGPT54Annot2LineageErrorReflect,
    # Sonnet-4.6 both-flags peer (lineage + replay ON, K=5, max_steps=20)
    DataflowSystemSonnet46Annot2LineageThoughtReplay,
    # gpt-5.4 latest + lineage + replay (K=5) — head-to-head vs code agent
    DataflowSystemGPT54Annot2LineageThoughtReplay,
    # gpt-5.4 latest + lineage, replay OFF (no-replay control)
    DataflowSystemGPT54Annot2Lineage,
    # gpt-5.4 Gate-0 headroom pair (latest-core vs full T/A/O delta trajectory, steps=12)
    DataflowSystemGPT54Gate0Latest,
    DataflowSystemGPT54Gate0Delta,
    # gpt-5.4 all-domains comparison pair (latest vs delta, steps=25, result char limit=3000)
    DataflowSystemGPT54AllLatest,
    DataflowSystemGPT54AllDelta,
    # Local-model sweep variants (local-react driver, qwen-xml dialect)
    DataflowSystemLocalLlm1,
    DataflowSystemLocalLlm2,
    DataflowSystemLocalLlm3,
    DataflowSystemLocalLlm4,
    DataflowSystemLocalLlm5,
    DataflowSystemLocalLlm6,
    DataflowSystemLocalLlm7,
    DataflowSystemLocalLlm8,
    DataflowSystemLocalLlm9,
    DataflowSystemLocalLlm10,
    DataflowSystemLocalLlm11,
    DataflowSystemLocalLlm12,
    DataflowSystemLocalLlm13,
    DataflowSystemLocalLlm14,
    DataflowSystemLocalLlm15,
    DataflowSystemLocalLlm30BV1,
    DataflowSystemLocalLlm30BV2,
    DataflowSystemLocalLlm30BV3,
    DataflowSystemLocalLlm30BV4,
    DataflowSystemLocalLlm30BV5,
    DataflowSystemLocalLlm30BV6,
    DataflowSystemLocalLlm30BV7,
    DataflowSystemLocalLlm30BV8,
    DataflowSystemLocalLlm30BV9,
    DataflowSystemLocalLlm30BV10,
    DataflowSystemLocalLlm30BBaseModel,
    # Local-model sweep under the previous ReAct text dialect (react-text)
    DataflowSystemLocalLlmReactText1,
    DataflowSystemLocalLlmReactText2,
    DataflowSystemLocalLlmReactText3,
    DataflowSystemLocalLlmReactText4,
    DataflowSystemLocalLlmReactText5,
)
from .code_agent_system import CodeAgentSystem, CodeAgentSystemHaiku, CodeAgentSystemSonnet, CodeAgentSystemGPT, CodeAgentSystemGptO3, CodeAgentSystemSonnet4, CodeAgentSystemHaiku45, CodeAgentSystemO4Mini, CodeAgentSystemGemini25Pro, CodeAgentSystemGpt52, CodeAgentSystemGpt52Chars2k, CodeAgentSystemGpt52Chars5k, CodeAgentSystemGpt52Chars2kGuided, CodeAgentSystemGpt52Chars3kGuided, CodeAgentSystemGpt52Chars3kGuidedExplore, CodeAgentSystemGpt52Chars5kGuided, CodeAgentSystemGpt52Chars7kGuided, CodeAgentSystemGpt52Chars10kGuided, CodeAgentSystemGpt52FineGrained, CodeAgentSystemGpt52FullInput, CodeAgentSystemGpt5MiniHigh, CodeAgentSystemGpt5MiniMedium, CodeAgentSystemGpt5MiniLow, CodeAgentSystemGpt5MiniProxy, CodeAgentSystemGpt5MiniProxyChars3kGuided, CodeAgentSystemGpt54Proxy
from .code_agent_session import CodeAgentSessionRunner, CodeAgentSessionSystem
from .dataflow_agent_session import DataflowAgentSessionRunner, DataflowAgentSessionSystem
