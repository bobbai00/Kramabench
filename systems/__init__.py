"""
System module initialization.
This module provides system-level functionality and configurations.
"""
from .baseline_example import ExampleBaselineSystem
from .evidence_only_system import *
from .compact_evidence_system import *
from .dataflow_system import (
    DataflowSystem,
)
from .native_python_system import (
    DataflowSystemLunaPythonPilotV1Reference20260915Rep1,
    DataflowSystemLunaPythonPilotBatchParent20260915Rep1,
    DataflowSystemLunaPythonPilotV2Control20260915Rep1,
    DataflowSystemLunaPythonPilotCollectorOnly20260915Rep1,
    DataflowSystemLunaPythonPilotDataOnly20260915Rep1,
    DataflowSystemLunaPythonPilotFlowOnly20260915Rep1,
    DataflowSystemLunaPythonPilotCombined20260915Rep1,
    DataflowSystemTerraPythonPilotV1Reference20260915Rep1,
    DataflowSystemTerraPythonPilotBatchParent20260915Rep1,
    DataflowSystemTerraPythonPilotV2Control20260915Rep1,
    DataflowSystemTerraPythonPilotCollectorOnly20260915Rep1,
    DataflowSystemTerraPythonPilotDataOnly20260915Rep1,
    DataflowSystemTerraPythonPilotFlowOnly20260915Rep1,
    DataflowSystemTerraPythonPilotCombined20260915Rep1,
)
from .native_campaign_system import (
    DataflowSystemLunaNativeCampaignBatchParent20260915Rep1,
    DataflowSystemLunaNativeCampaignDataOnly20260915Rep1,
    DataflowSystemLunaNativeCampaignFlowOnly20260915Rep1,
    DataflowSystemLunaNativeCampaignCombined20260915Rep1,
    DataflowSystemTerraNativeCampaignBatchParent20260915Rep1,
    DataflowSystemTerraNativeCampaignDataOnly20260915Rep1,
    DataflowSystemTerraNativeCampaignFlowOnly20260915Rep1,
    DataflowSystemTerraNativeCampaignCombined20260915Rep1,
    DataflowSystemLunaNativeCampaignObserveOnlyBatchParent20260916Rep1,
    DataflowSystemLunaNativeCampaignObserveOnlyDataOnly20260916Rep1,
    DataflowSystemLunaNativeCampaignObserveOnlyFlowOnly20260916Rep1,
    DataflowSystemLunaNativeCampaignObserveOnlyCombined20260916Rep1,
    DataflowSystemTerraNativeCampaignObserveOnlyBatchParent20260916Rep1,
    DataflowSystemTerraNativeCampaignObserveOnlyDataOnly20260916Rep1,
    DataflowSystemTerraNativeCampaignObserveOnlyFlowOnly20260916Rep1,
    DataflowSystemTerraNativeCampaignObserveOnlyCombined20260916Rep1,
)
from .code_agent_system import CodeAgentSystem
from .code_agent_session import CodeAgentSessionRunner, CodeAgentSessionSystem
from .dataflow_agent_session import DataflowAgentSessionRunner, DataflowAgentSessionSystem
from .claude_code_system import (
    ClaudeCodeSystem,
    ClaudeCodeSystemHaiku45Stateless,
    ClaudeCodeSystemHaiku45Persistent,
    ClaudeCodeSystemHaiku45PersistentChars2k,
)

# Model-grouped char-budget x prompt matrix (medium reasoning on every GPT arm)

# gpt-5.2 @ medium reasoning (the -medium litellm alias). Defined but never
# exported before, so kb.py's getattr(systems, name) could not resolve them.

# gpt-5.2 @ medium reasoning: C4 (existing cell) and C5 (new, = luna/terra C5)

# Scoped-stats matrix (Idea 1): same channels, different placement
from .dataflow_system import (
    DataflowSystemHaikuScopedControl,
    DataflowSystemHaikuScopedSplit,
    DataflowSystemHaikuScopedLean,
    DataflowSystemHaikuScopedSrcStats,
)

# Message-framing pair (Idea 2): block vs native tool-calling transcript
from .dataflow_system import (
    DataflowSystemHaikuLayoutBlock,
    DataflowSystemHaikuLayoutBlockSplit,
    DataflowSystemHaikuLayoutNative,
)

# Luna rows-axis midpoint on the stats ray (delta 2k + stats, no code).
from .dataflow_system import DataflowSystemLunaDeltaStats2kRep1
from .dataflow_system import DataflowSystemLunaDeltaStats2kCacheRep1
from .dataflow_system import (
    DataflowSystemLunaLatest1kRep1,
    DataflowSystemLunaLatest1kOpSplitRep1,
)
from .dataflow_system import (
    DataflowSystemLunaDelta2kStatsRep2,
    DataflowSystemLunaLatest2kStatsCodeRep1,
    DataflowSystemLunaNativeAllRep1,
    DataflowSystemLunaNativeRuleRep1,
    DataflowSystemLunaNativeRuleInspectRep1,
    DataflowSystemLunaNativeRuleSplitRep1,
    DataflowSystemLunaNativeAllSplitRep1,
    DataflowSystemLunaLatest2kStatsCodeSplitRep1,
    DataflowSystemLunaNativeRuleDeltaRep1,
    DataflowSystemLunaNativeAllDeltaRep1,
    DataflowSystemLunaNativeToolsRuleRep1,
    DataflowSystemLunaNativeToolsAllRep1,
    DataflowSystemLunaNativeToolsRuleDeltaRep1,
    DataflowSystemLunaNativeToolsRuleDeltaRep2,
    DataflowSystemLunaNativeToolsScopedDeltaRep1,
    DataflowSystemLunaNativeToolsScopedDeltaRep2,
    DataflowSystemLunaNativeToolsScopedDeltaRep3,
    DataflowSystemLunaNativeAnchorRep1,
    DataflowSystemLunaNativeAnchorRep2,
    DataflowSystemLunaNativeAnchorRep3,
    DataflowSystemLunaNativePilotBaseline20260914Rep1,
    DataflowSystemLunaNativePilotBatch20260914Rep1,
    DataflowSystemLunaNativePilotEvidence20260914Rep1,
    DataflowSystemLunaNativeToolsScopedFixRep1,
    DataflowSystemLunaNativeToolsScopedFix2Rep1,
    DataflowSystemLunaLatest2kStatsNoCodeRep1,
)


# Canonical Anchor/C1..C5 grid (model x config x replicate). Generated in
# dataflow_system.py; re-exported here so the harness's getattr lookup finds
# each class by name.
from .dataflow_system import GRID_SYSTEM_NAMES as _GRID_NAMES  # noqa: E402
import systems.dataflow_system as _dfs  # noqa: E402

for _n in _GRID_NAMES:
    globals()[_n] = getattr(_dfs, _n)
__all__ = list(__all__) + list(_GRID_NAMES) if "__all__" in dir() else None


# Guided code-agent matrix (3 models x {1k,2k,5k} x Rep0-2), generated in
# code_agent_system.py; re-exported so the harness getattr lookup finds them.
from .code_agent_system import CODE_AGENT_MATRIX_NAMES as _CA_NAMES  # noqa: E402
import systems.code_agent_system as _cas  # noqa: E402

for _n in _CA_NAMES:
    globals()[_n] = getattr(_cas, _n)


# Sonnet-5 Claude Code arms (1k/5k x Rep0-2), generated in claude_code_system.py.
from .claude_code_system import CLAUDE_CODE_SONNET_NAMES as _CC_SONNET  # noqa: E402
import systems.claude_code_system as _ccs  # noqa: E402

for _n in _CC_SONNET:
    globals()[_n] = getattr(_ccs, _n)


# 2x2 layout probe (DELTA vs opBlock) x (evidence off/on), 2026-09-19.
from .layout_probe_system import LAYOUT_PROBE_ARMS as _LAYOUT_ARMS  # noqa: E402

for _cls in _LAYOUT_ARMS:
    globals()[_cls.__name__] = _cls
