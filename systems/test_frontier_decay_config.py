import unittest

from dataflow_agent import AgentSettings
from systems.dataflow_system import (
    DataflowSystemGPT52LatestStats3kD2,
    DataflowSystemGPT52LatestStats3kD2FrontierDecay,
    DataflowSystemGPT52LatestStats3kD2SmallTableControl,
)


class FrontierDecayConfigTest(unittest.TestCase):
    def test_client_only_serializes_an_explicit_overlay(self):
        self.assertNotIn("frontierDecayConfig", AgentSettings().to_api_dict())

        config = {
            "sampleRows": 3,
            "minStepsSinceEdit": 1,
            "minConsumerStepsSinceEdit": 1,
            "minConsumerStepsSinceHealthy": 1,
        }
        payload = AgentSettings(frontier_decay_config=config).to_api_dict()
        self.assertEqual(config, payload["frontierDecayConfig"])

    def test_treatment_matches_latest_3k_baseline_except_for_overlay(self):
        baseline = DataflowSystemGPT52LatestStats3kD2()
        current_control = DataflowSystemGPT52LatestStats3kD2SmallTableControl()
        treatment = DataflowSystemGPT52LatestStats3kD2FrontierDecay()

        matched_fields = [
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
        for field in matched_fields:
            self.assertEqual(getattr(baseline, field), getattr(current_control, field), field)
            self.assertEqual(getattr(baseline, field), getattr(treatment, field), field)

        self.assertIsNone(baseline.frontier_decay_config)
        self.assertIsNone(current_control.frontier_decay_config)
        self.assertEqual(
            treatment.frontier_decay_config,
            {
                "sampleRows": 3,
                "minStepsSinceEdit": 1,
                "minConsumerStepsSinceEdit": 1,
                "minConsumerStepsSinceHealthy": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
