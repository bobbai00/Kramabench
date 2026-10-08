# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.

"""CodeAgentWrapper keeps the agent's answer whatever its memory holds."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smolagents.memory import ActionStep, TaskStep
from smolagents.monitoring import Timing, TokenUsage
from smolagents.utils import AgentMaxStepsError

from code_agent import CodeAgentWrapper


def wrapper_with(steps, output):
    """A wrapper around a finished agent run: these steps, this final output."""

    class Memory:
        def __init__(self):
            self.steps = steps

        def get_full_steps(self):
            raise RecursionError("maximum recursion depth exceeded")

    wrapper = CodeAgentWrapper(model_type="gpt-5.6-luna")
    wrapper._agent = types.SimpleNamespace(memory=Memory(), run=lambda prompt: output)
    wrapper._model = types.SimpleNamespace(
        usage_acc={"input": 0, "output": 0, "cached": 60, "reasoning": 5}
    )
    return wrapper


def action(number, **fields):
    return ActionStep(
        step_number=number,
        timing=Timing(start_time=0.0, end_time=1.0),
        token_usage=TokenUsage(input_tokens=100, output_tokens=10),
        **fields
    )


class CodeAgentWrapperTests(unittest.TestCase):
    def test_answer_survives_a_memory_that_cannot_be_exported(self):
        result = wrapper_with(
            [TaskStep(task="question"), action(1), action(2, is_final_answer=True)],
            0.4321,
        ).run("question")
        self.assertEqual(
            (result.response, result.state, result.error), ("0.4321", "success", None)
        )
        self.assertEqual(
            (result.input_tokens, result.output_tokens, result.total_tokens),
            (200, 20, 220),
        )
        self.assertEqual(
            (result.cached_tokens, result.reasoning_tokens, result.num_steps),
            (60, 5, 3),
        )

    def test_zero_is_an_answer_and_the_step_limit_is_reported(self):
        self.assertEqual(
            wrapper_with(
                [TaskStep(task="question"), action(1, is_final_answer=True)], 0
            )
            .run("q")
            .response,
            "0",
        )
        quiet = types.SimpleNamespace(log_error=lambda message: None)
        limit = action(26, error=AgentMaxStepsError("Reached max steps.", quiet))
        result = wrapper_with(
            [TaskStep(task="question"), action(25), limit], "It is probably 7."
        ).run("question")
        self.assertEqual(
            (result.response, result.state), ("It is probably 7.", "max_steps_error")
        )


if __name__ == "__main__":
    unittest.main()
