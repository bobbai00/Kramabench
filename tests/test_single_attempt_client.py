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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dataflow_agent import DataflowAgent, MessageResult, _send_message_once, send_message


class SingleAttemptClientTest(unittest.TestCase):
    def empty(self):
        return MessageResult(response="", messages=[], usage={}, stats={"steps": 0}, stopped=False)

    def test_explicit_zero_overrides_environment_retries(self):
        with (
            patch.dict("os.environ", {"KB_EMPTY_TURN_RETRIES": "2"}),
            patch("dataflow_agent._send_message_once", return_value=self.empty()) as send,
            patch("dataflow_agent.time.sleep") as sleep,
        ):
            result = send_message("agent", "query", empty_turn_retries=0)
            self.assertEqual(result.stats["steps"], 0)
            send.assert_called_once()
            sleep.assert_not_called()

    def test_legacy_environment_policy_remains_available(self):
        with (
            patch.dict("os.environ", {"KB_EMPTY_TURN_RETRIES": "1"}),
            patch("dataflow_agent._send_message_once", return_value=self.empty()) as send,
            patch("dataflow_agent.time.sleep"),
        ):
            send_message("agent", "query")
            self.assertEqual(send.call_count, 2)

    def test_client_run_transmits_single_attempt_policy(self):
        agent = DataflowAgent(verbosity_level=0)
        agent._agent_info = SimpleNamespace(id="agent")
        with patch("dataflow_agent.send_message", return_value=self.empty()) as send:
            agent.run("query", empty_turn_retries=0)
            self.assertEqual(send.call_args.kwargs["empty_turn_retries"], 0)

    def test_bad_retry_counts_fail_before_dispatch(self):
        with patch("dataflow_agent._send_message_once") as send:
            for count in (-1, True, 1.5, "0"):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    send_message("agent", "query", empty_turn_retries=count)
            send.assert_not_called()

    def test_websocket_complete_and_silent_close_are_distinct(self):
        for frames, completed in ((['{"type":"complete"}'], True), ([""], False)):
            socket = SimpleNamespace(send=lambda *args: None, recv=iter(frames).__next__, close=lambda: None)
            with patch("dataflow_agent.websocket.create_connection", return_value=socket):
                result = _send_message_once("agent", "query")
                self.assertIs(result.completed, completed)

    def test_progress_callback_survives_a_later_transport_failure(self):
        import json

        step = {"id": "s1", "role": "agent", "usage": {"inputTokens": 10}}
        frames = iter([json.dumps({"type": "step", "step": step})])
        socket = SimpleNamespace(send=lambda *args: None, recv=frames.__next__, close=lambda: None)
        events = []
        with patch("dataflow_agent.websocket.create_connection", return_value=socket):
            with self.assertRaises(StopIteration):
                send_message("agent", "query", empty_turn_retries=0, on_event=events.append)
        self.assertEqual(events, [{"type": "step", "step": step}])

    def test_client_forwards_event_callback(self):
        agent = DataflowAgent(verbosity_level=0)
        agent._agent_info = SimpleNamespace(id="agent")
        callback = lambda event: None
        with patch("dataflow_agent.send_message", return_value=self.empty()) as send:
            agent.run("query", empty_turn_retries=0, on_event=callback)
            self.assertIs(send.call_args.kwargs["on_event"], callback)

    def test_final_step_rebroadcast_is_not_billed_twice(self):
        import json

        step = {
            "id": "s1",
            "role": "agent",
            "isEnd": False,
            "content": "17",
            "usage": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120, "cachedInputTokens": 80},
        }
        frames = iter(
            [
                json.dumps({"type": "step", "step": step}),
                json.dumps({"type": "step", "step": {**step, "isEnd": True}}),
                '{"type":"complete"}',
            ]
        )
        socket = SimpleNamespace(send=lambda *args: None, recv=frames.__next__, close=lambda: None)
        with patch("dataflow_agent.websocket.create_connection", return_value=socket):
            result = _send_message_once("agent", "query")
        self.assertEqual(result.stats["steps"], 1)
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertEqual(result.usage["cached_input_tokens"], 80)
        self.assertEqual(result.response, "17")


if __name__ == "__main__":
    unittest.main()
