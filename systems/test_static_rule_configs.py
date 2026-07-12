import unittest

from dataflow_agent import AgentSettings
from systems.dataflow_system import (
    DataflowSystemGPT52DeltaStats3kD2,
    DataflowSystemGPT52DeltaStats3kD2FoldControl,
    DataflowSystemGPT52DeltaStats3kD2FoldResolved,
    DataflowSystemGPT52LatestStats3kD2,
    DataflowSystemGPT52LatestStats3kD2Lean3,
    DataflowSystemGPT52LatestStats3kD2Lean3Pull,
    DataflowSystemGPT52LatestStats3kD2ProbeRetire,
    DataflowSystemGPT52LatestStats3kD2SmallTableControl,
)

MATCHED_FIELDS = [
    "model_type",
    "context_mode",
    "max_steps",
    "flow_level",
    "data_level",
    "column_stats",
    "attempt_reflection",
    "max_operator_result_char_limit",
    "max_operator_result_cell_char_limit",
]


class StaticRuleConfigTest(unittest.TestCase):
    def test_client_only_serializes_explicit_configs(self):
        payload = AgentSettings().to_api_dict()
        self.assertNotIn("foldResolvedRevisionsConfig", payload)
        self.assertNotIn("probeRetirementConfig", payload)

        fold = {"graceEvents": 1}
        self.assertEqual(fold, AgentSettings(fold_resolved_revisions_config=fold).to_api_dict()["foldResolvedRevisionsConfig"])

        probe = {"minStepsSinceEdit": 2, "minValueLength": 4}
        self.assertEqual(probe, AgentSettings(probe_retirement_config=probe).to_api_dict()["probeRetirementConfig"])

    def test_rank3_arms_match_delta_baseline_except_for_the_rule(self):
        baseline = DataflowSystemGPT52DeltaStats3kD2()
        control = DataflowSystemGPT52DeltaStats3kD2FoldControl()
        treatment = DataflowSystemGPT52DeltaStats3kD2FoldResolved()

        for field in MATCHED_FIELDS:
            self.assertEqual(getattr(baseline, field), getattr(control, field), field)
            self.assertEqual(getattr(baseline, field), getattr(treatment, field), field)
        self.assertEqual("delta", control.context_mode)

        for arm in (baseline, control):
            self.assertIsNone(arm.fold_resolved_revisions_config)
            self.assertIsNone(arm.probe_retirement_config)
            self.assertIsNone(arm.frontier_decay_config)
        self.assertEqual({"graceEvents": 1}, treatment.fold_resolved_revisions_config)
        self.assertIsNone(treatment.probe_retirement_config)
        self.assertIsNone(treatment.frontier_decay_config)

    def test_rank4_arm_matches_latest_control_except_for_the_rule(self):
        control = DataflowSystemGPT52LatestStats3kD2SmallTableControl()
        treatment = DataflowSystemGPT52LatestStats3kD2ProbeRetire()
        baseline = DataflowSystemGPT52LatestStats3kD2()

        for field in MATCHED_FIELDS:
            self.assertEqual(getattr(baseline, field), getattr(control, field), field)
            self.assertEqual(getattr(baseline, field), getattr(treatment, field), field)
        self.assertEqual("latest", treatment.context_mode)

        self.assertIsNone(control.probe_retirement_config)
        self.assertIsNone(control.fold_resolved_revisions_config)
        self.assertIsNone(control.frontier_decay_config)
        self.assertEqual({"minStepsSinceEdit": 2, "minValueLength": 4}, treatment.probe_retirement_config)
        self.assertIsNone(treatment.fold_resolved_revisions_config)
        self.assertIsNone(treatment.frontier_decay_config)

    def test_e1_arms_match_latest_control_except_rows_and_pull(self):
        control = DataflowSystemGPT52LatestStats3kD2SmallTableControl()
        lean = DataflowSystemGPT52LatestStats3kD2Lean3()
        pull = DataflowSystemGPT52LatestStats3kD2Lean3Pull()

        for field in MATCHED_FIELDS:
            self.assertEqual(getattr(control, field), getattr(lean, field), field)
            self.assertEqual(getattr(control, field), getattr(pull, field), field)

        self.assertEqual(0, control.max_result_rows)
        self.assertEqual(3, lean.max_result_rows)
        self.assertEqual(3, pull.max_result_rows)
        self.assertFalse(control.enable_inspect_tool)
        self.assertFalse(lean.enable_inspect_tool)
        self.assertTrue(pull.enable_inspect_tool)
        for arm in (lean, pull):
            self.assertIsNone(arm.frontier_decay_config)
            self.assertIsNone(arm.fold_resolved_revisions_config)
            self.assertIsNone(arm.probe_retirement_config)

    def test_inspect_tool_serialization(self):
        self.assertNotIn("enableInspectTool", AgentSettings().to_api_dict())
        self.assertTrue(AgentSettings(enable_inspect_tool=True).to_api_dict()["enableInspectTool"])


if __name__ == "__main__":
    unittest.main()
