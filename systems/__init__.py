"""
System module initialization.
This module provides system-level functionality and configurations.
"""
from .baseline_example import ExampleBaselineSystem
from .dataflow_system import (
    DataflowSystem,
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
