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
    DataflowSystemLocalLlm,
    # Thesis converge stacks (DataFlow-vs-Script, both models)
    DataflowSystemGPT52LatestSchemaConverge,
    DataflowSystemGPT5MiniLatestSchemaConverge,
    # Canonical level-based + the validated levers
    DataflowSystemGPT5MiniLatestSchemaConvergeLevels,      # flow_level/data_level config
    DataflowSystemGPT5MiniLatestSchemaConvergeTableStruct, # the accuracy win (data_level=2)
    DataflowSystemGPT5MiniLatestSchemaConvergeFewShot,     # W2 cost win
    DataflowSystemGPT5MiniLatestSchemaConvergeCap20,       # #31 step-cap (ACT lever)
)
from .code_agent_system import CodeAgentSystem, CodeAgentSystemHaiku, CodeAgentSystemSonnet, CodeAgentSystemGPT, CodeAgentSystemGptO3, CodeAgentSystemSonnet4, CodeAgentSystemHaiku45, CodeAgentSystemO4Mini, CodeAgentSystemGemini25Pro, CodeAgentSystemGpt52, CodeAgentSystemGpt52FineGrained, CodeAgentSystemGpt52FullInput, CodeAgentSystemGpt5MiniHigh, CodeAgentSystemGpt5MiniMedium, CodeAgentSystemGpt5MiniLow
from .code_agent_session import CodeAgentSessionRunner, CodeAgentSessionSystem
from .dataflow_agent_session import DataflowAgentSessionRunner, DataflowAgentSessionSystem