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
    DataflowSystemGPT54DeltaSchemaConverge,
    DataflowSystemGPT54DeltaSchemaConvergeCollapse,
    DataflowSystemGPT54DeltaSchemaConvergeCap4k,
    DataflowSystemGPT54DeltaSchemaConvergeCap8k,
    DataflowSystemGPT54DeltaSchemaConvergeTrunc4k,
    DataflowSystemGPT54DeltaSchemaConvergeTrunc8k,
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
from .code_agent_system import CodeAgentSystem, CodeAgentSystemHaiku, CodeAgentSystemSonnet, CodeAgentSystemGPT, CodeAgentSystemGptO3, CodeAgentSystemSonnet4, CodeAgentSystemHaiku45, CodeAgentSystemO4Mini, CodeAgentSystemGemini25Pro, CodeAgentSystemGpt52, CodeAgentSystemGpt52FineGrained, CodeAgentSystemGpt52FullInput, CodeAgentSystemGpt5MiniHigh, CodeAgentSystemGpt5MiniMedium, CodeAgentSystemGpt5MiniLow, CodeAgentSystemGpt5MiniProxy, CodeAgentSystemGpt54Proxy
from .code_agent_session import CodeAgentSessionRunner, CodeAgentSessionSystem
from .dataflow_agent_session import DataflowAgentSessionRunner, DataflowAgentSessionSystem