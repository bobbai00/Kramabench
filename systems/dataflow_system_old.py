"""
Old-stack SUT: drives the OLD (fe917396a) dataflow-agent stack through the
ERA-MATCHED client (dataflow_agent_old.py = harness commit 7173a6c, 2026-03-16,
the version that actually ran against that branch).

Prereqs (see session notes): the old stack must be running —
  - old JVM services from ~/Desktop/bobflow/dataflow-agent-copilot-fe91739
    (main DB on :5433, texera/texera user seeded)
  - old agent-service (bun) on :3001
  - litellm on :4000

Settings replicate the old benchmark configuration:
  no_action_detail=True (tool history rewritten into the DAG summary),
  carry_metadata=True   (per-column Column Stats in execution metadata),
  parallel_tool_calls=True,
  serialization "table" (era default), char limits 1000/3000, max_steps=25
  (matched to the current-branch comparison SUTs).
"""

from dataflow_agent_old import DataflowAgent as OldDataflowAgent

from .dataflow_system import DataflowSystem


class DataflowSystemOldStackGPT52NoActionDetail(DataflowSystem):
    """gpt-5.2 on the OLD stack via the era client — NoActionDetail + carry
    metadata + parallel tool calls, char limits 1000/3000, 25 steps."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(
            model_type="gpt-5.2",
            context_mode="latest",  # recorded in config dumps only; era client ignores it
            max_steps=25,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            parallel_tool_calls=True,
            name="DataflowSystemOldStackGPT52NoActionDetail",
            verbose=verbose,
            *args,
            **kwargs,
        )

    def _setup_agent(self) -> None:
        """Construct the ERA client against the old stack instead of today's
        client. The era client exposes the same surface today's serve_query
        uses (setup/run/reset/cleanup/agent_id/agent_service_endpoint and a
        MessageResult with response/messages/usage/stats/stopped/error)."""
        if self.verbose:
            print(f"[OldStackDataflowSystem] Setting up ERA agent: {self.model_type}")
        self.agent = OldDataflowAgent(
            model_type="gpt-5.2",
            max_steps=25,
            max_operator_result_char_limit=1000,
            max_operator_result_cell_char_limit=3000,
            operator_result_serialization_mode="table",  # era default
            tool_timeout_seconds=240,
            execution_timeout_minutes=4,
            agent_mode="code",
            parallel_tool_calls=True,
            no_action_detail=True,
            carry_metadata=True,
            username="texera",
            password="texera",
            verbosity_level=2 if self.verbose else 1,
        )
        self.agent.setup()
