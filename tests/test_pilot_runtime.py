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

import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from utils.pilot_runtime import archive_jars, inspect_engine_process, verify_engine_runtime


class PilotRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.worktree = self.root / "candidate"
        self.amber = self.worktree / "amber"
        self.amber.mkdir(parents=True)
        self.archive = self.root / "amber-fixture.zip"
        self.lib = self.amber / "target/amber-fixture/lib"
        self.lib.mkdir(parents=True)
        self.names = ["org.apache.texera.amber-fixture.jar", "dependency.jar"]
        with zipfile.ZipFile(self.archive, "w") as target:
            for name in self.names:
                content = ("jar fixture " + name).encode()
                target.writestr("amber-fixture/lib/" + name, content)
                (self.lib / name).write_bytes(content)
        self.proc = self.root / "proc"
        self.process = self.proc / "123"
        self.process.mkdir(parents=True)
        (self.proc / "sys/kernel/random").mkdir(parents=True)
        (self.proc / "sys/kernel/random/boot_id").write_text("fixture-boot")
        (self.process / "cwd").symlink_to(self.amber)
        java = self.root / "java"
        java.write_bytes(b"fixture java executable")
        (self.process / "exe").symlink_to(java)
        self.python = self.worktree / ".venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"fixture python executable")
        self.args = [
            str(java),
            "-cp",
            ":".join(str(self.lib / name) for name in self.names),
            "org.apache.texera.web.ComputingUnitMaster",
        ]
        self.environment = {"UDF_PYTHON_PATH": str(self.python), "SECRET": "never-copy-private-token"}
        self.write_process()
        self.source = {
            "source_root": str(self.worktree),
            "git_sha": "a" * 40,
            "source_sha256": "source",
            "proto_sha256": "proto",
        }
        self.python_info = {"path": str(self.python), "prefix": "fixture", "version": "3.12"}
        self.build = {
            "version": 1,
            "status": "built",
            "command": ["sbt", "-no-colors", "WorkflowExecutionService/dist"],
            "exit_code": 0,
            "source_before": self.source,
            "source_after": self.source,
            "completed_clock": {"boot_id": "fixture-boot", "ticks": 100},
            "archive": {
                "resolved_path": str(self.archive),
                "bytes": self.archive.stat().st_size,
                "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            },
            "jars": archive_jars(self.archive),
            "python": self.python_info,
        }
        self.engine_source = patch("utils.pilot_runtime.engine_source", return_value=self.source).start()
        self.python_runtime = patch("utils.pilot_runtime.python_runtime", return_value=self.python_info).start()
        self.command = patch(
            "utils.pilot_runtime._command", return_value='LISTEN *:8085 users:(("java",pid=123,fd=1))'
        ).start()
        self.addCleanup(patch.stopall)

    def write_process(self, ticks=200, state="S"):
        (self.process / "cmdline").write_bytes(b"\0".join(value.encode() for value in self.args) + b"\0")
        (self.process / "environ").write_bytes(
            b"\0".join((key + "=" + value).encode() for key, value in self.environment.items())
        )
        fields = [state] + ["0"] * 18 + [str(ticks)] + ["0"] * 10
        (self.process / "stat").write_text("123 (java (fixture)) " + " ".join(fields))

    def inspect(self):
        return inspect_engine_process("http://127.0.0.1:8085", proc_root=self.proc)

    def verify(self, expected=None):
        return verify_engine_runtime(self.build, "http://127.0.0.1:8085", expected=expected, proc_root=self.proc)

    def test_process_record_does_not_export_arguments_or_environment_secrets(self):
        record = self.inspect()
        self.assertEqual(record["pid"], 123)
        self.assertEqual(record["cwd"], str(self.amber))
        self.assertEqual(record["start_ticks"], 200)
        self.assertNotIn("never-copy-private-token", json.dumps(record))
        self.assertNotIn("SECRET", json.dumps(record))

    def test_matching_new_process_jar_bytes_worker_source_and_interpreter_bind(self):
        record = self.verify()
        self.assertTrue(record["build_bound"])
        self.assertFalse(record["live_correctness_qualified"])
        self.assertEqual(self.verify(expected=record), record)

    def test_old_process_cannot_be_qualified_by_a_new_build_on_disk(self):
        self.write_process(ticks=50)
        with self.assertRaisesRegex(ValueError, "predates"):
            self.verify()

    def test_other_worktree_cannot_be_adopted_by_port(self):
        other = self.root / "other-amber"
        other.mkdir()
        (self.process / "cwd").unlink()
        (self.process / "cwd").symlink_to(other)
        with self.assertRaisesRegex(ValueError, "working_directory"):
            self.verify()

    def test_jar_drift_and_unpacked_wrong_build_are_rejected(self):
        (self.lib / self.names[0]).write_bytes(b"different compiled classes")
        with self.assertRaisesRegex(ValueError, "classpath"):
            self.verify()

    def test_adding_a_classpath_directory_or_shadowing_jar_is_rejected(self):
        self.args[2] += ":" + str(self.root)
        self.write_process()
        with self.assertRaisesRegex(ValueError, "classpath"):
            self.inspect()

    def test_python_override_or_named_environment_cannot_be_assumed_default(self):
        self.environment["UDF_PYTHON_PATH"] = str(self.root / "different-python")
        self.write_process()
        with self.assertRaisesRegex(ValueError, "python"):
            self.verify()

    def test_injected_java_options_and_config_overrides_are_refused(self):
        for options in ("-javaagent:foreign.jar", "-Dconfig.file=other.conf", "-DUDF_PYTHON_PATH=other"):
            with self.subTest(options=options):
                self.environment["JAVA_TOOL_OPTIONS"] = options
                self.write_process()
                with self.assertRaisesRegex(ValueError, "injected"):
                    self.inspect()
        self.environment.pop("JAVA_TOOL_OPTIONS")
        self.args.insert(1, "-Dconfig.file=other.conf")
        self.write_process()
        with self.assertRaisesRegex(ValueError, "override"):
            self.inspect()

    def test_process_restart_or_environment_drift_invalidates_bound_runtime(self):
        bound = self.verify()
        self.environment["STORAGE_S3_ENDPOINT"] = "http://different.invalid"
        self.write_process()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.verify(expected=bound)
        self.environment.pop("STORAGE_S3_ENDPOINT")
        self.write_process(ticks=201)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.verify(expected=bound)

    def test_protos_or_interpreter_drift_are_not_hidden_by_same_git_sha(self):
        bound = self.verify()
        self.engine_source.return_value = {**self.source, "proto_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "source"):
            self.verify(expected=bound)
        self.engine_source.return_value = self.source
        self.python_runtime.return_value = {**self.python_info, "prefix": "different-venv"}
        with self.assertRaisesRegex(ValueError, "python"):
            self.verify(expected=bound)

    def test_docs_only_head_change_preserves_actual_built_source_identity(self):
        bound = self.verify()
        self.engine_source.return_value = {**self.source, "git_sha": "b" * 40}
        self.assertEqual(self.verify(expected=bound), bound)
        self.assertEqual(bound["source"]["git_sha"], "a" * 40)

    def test_a_passed_flag_without_exact_build_evidence_is_not_qualification(self):
        self.build = {"status": "built"}
        with self.assertRaisesRegex(ValueError, "build"):
            self.verify()

    def test_failed_build_or_changed_archive_is_refused(self):
        self.build["exit_code"] = 1
        with self.assertRaisesRegex(ValueError, "build"):
            self.verify()
        self.build["exit_code"] = 0
        self.archive.write_bytes(b"new archive")
        with self.assertRaisesRegex(ValueError, "archive"):
            self.verify()

    def test_nonlocal_endpoint_and_missing_process_are_not_resolved(self):
        with self.assertRaises(ValueError):
            inspect_engine_process("https://remote.invalid:8085", proc_root=self.proc)
        self.command.return_value = ""
        with self.assertRaisesRegex(ValueError, "listener"):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
