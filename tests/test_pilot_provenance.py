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

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from utils.pilot_provenance import inspect_service_listener


class ListenerProvenanceTest(unittest.TestCase):
    def fixture(self, directory, *, cmd=None):
        root = Path(directory)
        repo = root / "arbitrary-worktree"
        source = repo / "agent-service" / "src" / "server.ts"
        source.parent.mkdir(parents=True)
        source.write_text("// controlled test source\n")
        proc = root / "proc"
        process = proc / "123"
        process.mkdir(parents=True)
        (process / "cwd").symlink_to(source.parent.parent, target_is_directory=True)
        (process / "cmdline").write_bytes(cmd or b"bun\x00run\x00--watch\x00src/server.ts\x00")
        (process / "stat").write_text("123 (bun worker) S " + "0 " * 18 + "4567 0 0\n")
        return repo, proc, process

    def command(self, repo, *, listeners='LISTEN users:(("bun",pid=123,fd=17))', dirty="", after_git=None):
        def run(args, **kwargs):
            self.assertNotIn("shell", kwargs)
            if args[0] == "ss":
                return SimpleNamespace(returncode=0, stdout=listeners)
            if "--show-toplevel" in args:
                value = str(repo)
            elif "rev-parse" in args:
                value = "a" * 40
            elif "status" in args:
                value = dirty
                if after_git:
                    after_git()
            else:
                raise AssertionError(args)
            return SimpleNamespace(returncode=0, stdout=value, stderr="")

        return run

    def test_resolves_the_actual_listener_not_a_port_to_worktree_guess(self):
        with TemporaryDirectory() as directory:
            repo, proc, _ = self.fixture(directory)
            result = inspect_service_listener("http://localhost:3011", proc_root=proc, run=self.command(repo))
            self.assertEqual(result["status"], "resolved")
            self.assertEqual(result["source_root"], str(repo))
            self.assertEqual(result["git_sha"], "a" * 40)
            self.assertFalse(result["source_dirty"])
            self.assertEqual(result["pid"], 123)
            self.assertEqual(result["process_start_ticks"], 4567)
            self.assertEqual(len(result["entry_sha256"]), 64)
            self.assertFalse(result["loaded_revision_verified"])  # a source lookup is not a boot attestation

    def test_missing_remote_ambiguous_and_non_agent_listeners_stay_unknown(self):
        with TemporaryDirectory() as directory:
            repo, proc, process = self.fixture(directory)
            for endpoint, listeners in (
                ("https://example.com:3011", ""),
                ("http://localhost:3011", ""),
                ("http://localhost:3011", "pid=123 pid=456"),
            ):
                with self.subTest(endpoint=endpoint, listeners=listeners):
                    result = inspect_service_listener(
                        endpoint, proc_root=proc, run=self.command(repo, listeners=listeners)
                    )
                    self.assertEqual(result["status"], "unavailable")
                    self.assertIsNone(result["git_sha"])
            (process / "cmdline").write_bytes(b"python\x00http.server\x00")
            result = inspect_service_listener("http://localhost:3011", proc_root=proc, run=self.command(repo))
            self.assertEqual(result["status"], "unavailable")

    def test_dirty_sources_are_not_labeled_clean_and_pid_reuse_is_rejected(self):
        with TemporaryDirectory() as directory:
            repo, proc, process = self.fixture(directory)
            result = inspect_service_listener(
                "http://localhost:3011",
                proc_root=proc,
                run=self.command(repo, dirty=" M agent-service/src/server.ts\n"),
            )
            self.assertTrue(result["source_dirty"])
            result = inspect_service_listener(
                "http://localhost:3011",
                proc_root=proc,
                run=self.command(
                    repo, after_git=lambda: (process / "stat").write_text("123 (bun) S " + "0 " * 18 + "9999 0 0\n")
                ),
            )
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], "process_changed")

    def test_does_not_record_arguments_environment_or_raw_command_errors(self):
        with TemporaryDirectory() as directory:
            repo, proc, _ = self.fixture(directory, cmd=b"bun\x00src/server.ts\x00--private\x00secret-argument\x00")
            result = inspect_service_listener("http://localhost:3011", proc_root=proc, run=self.command(repo))
            self.assertEqual(result["status"], "resolved")
            self.assertNotIn("secret", json.dumps(result))

            def fail(*args, **kwargs):
                raise RuntimeError("secret-command-body")

            result = inspect_service_listener("http://localhost:3011", proc_root=proc, run=fail)
            self.assertEqual(result["status"], "unavailable")
            self.assertNotIn("secret", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
