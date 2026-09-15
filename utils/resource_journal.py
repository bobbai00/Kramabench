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

"""Private allocation intents/receipts, never a resource-deletion authority.

Each receipt is persisted before the next allocation. A hard crash between a
server commit and its receipt leaves a dangling intent; do not retry or adopt
a resource by name. References (such as a supplied CU ID) are not ownership.
Cleanup still requires fresh service/resource/terminal-execution verification.
"""

import copy
import json
import os
import re
import time
import uuid
from pathlib import Path


def _identifier(kind, value):
    if kind == "agent":
        return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value))
    return type(value) is int and 0 < value <= 2147483647


def _name(value):
    return isinstance(value, str) and 0 < len(value) <= 160 and all(ord(character) >= 32 for character in value)


def _transition(state, event):
    if not isinstance(event, dict) or event.get("kind") not in ("computing_unit", "workflow", "agent"):
        raise ValueError("invalid resource kind")
    kind, action = event["kind"], event.get("event")
    if action == "reference":
        if set(event) != {"event", "kind", "id"} or kind != "computing_unit" or not _identifier(kind, event["id"]):
            raise ValueError("invalid external computing-unit reference")
        previous = state["references"].get(kind)
        owned = state["owned"].get(kind)
        if (previous is not None and previous != event["id"]) or (owned and owned["id"] != event["id"]):
            raise ValueError("computing-unit identity changed")
        state["references"][kind] = event["id"]
    elif action == "create_intent":
        if set(event) != {"event", "kind", "name"} or not _name(event["name"]):
            raise ValueError("invalid resource creation intent")
        if kind in state["owned"] or kind in state["pending"]:
            raise ValueError("allocation already attempted; do not retry")
        state["pending"][kind] = event["name"]
    elif action == "created":
        if (
            set(event) != {"event", "kind", "id", "name"}
            or not _name(event["name"])
            or not _identifier(kind, event["id"])
        ):
            raise ValueError("invalid creation receipt")
        if kind not in state["pending"] or state["pending"][kind] != event["name"] or kind in state["owned"]:
            raise ValueError("creation receipt has no matching intent")
        if kind in state["references"] and state["references"][kind] != event["id"]:
            raise ValueError("creation receipt does not match the computing-unit reference")
        state["owned"][kind] = {"id": event["id"], "name": event["name"]}
        del state["pending"][kind]
    else:
        raise ValueError("invalid allocation event")


def _summary(state, run_id, closed):
    return {
        "version": 1,
        "run_id": run_id,
        "closed": closed,
        "owned": copy.deepcopy(state["owned"]),
        "references": dict(state["references"]),
        "pending": sorted(state["pending"]),
        "allocations_complete": closed and not state["pending"],
    }


class ResourceJournal:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.run_id = uuid.uuid4().hex
        self._state = {"owned": {}, "references": {}, "pending": {}}
        self._sequence = 0
        self._failed = False
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            self._append({"event": "journal_started"})
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            os.close(self._fd)
            self._fd = None
            raise

    def _append(self, event):
        if self._fd is None or self._failed:
            raise RuntimeError("resource journal is closed or failed")
        data = (
            json.dumps(
                {**event, "version": 1, "run_id": self.run_id, "sequence": self._sequence, "timestamp": time.time()},
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        try:
            written = 0
            while written < len(data):
                count = os.write(self._fd, data[written:])
                if count <= 0:
                    raise OSError("incomplete resource journal write")
                written += count
            os.fsync(self._fd)
        except BaseException:
            self._failed = True
            raise
        self._sequence += 1

    def record(self, event):
        candidate = copy.deepcopy(self._state)
        _transition(candidate, event)
        self._append(event)
        self._state = candidate

    def close(self):
        if self._fd is None:
            return
        try:
            if not self._failed:
                self._append({"event": "journal_closed"})
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_resource_journal(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("invalid resource journal size or type")
    state = {"owned": {}, "references": {}, "pending": {}}
    run_id, closed = None, False
    with path.open(encoding="utf-8") as stream:
        for sequence, line in enumerate(stream):
            try:
                record = json.loads(line)
            except (ValueError, RecursionError) as error:
                raise ValueError("malformed resource journal") from error
            if (
                closed
                or not isinstance(record, dict)
                or type(record.get("version")) is not int
                or record["version"] != 1
                or type(record.get("sequence")) is not int
                or record["sequence"] != sequence
                or not isinstance(record.get("run_id"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", record["run_id"])
            ):
                raise ValueError("resource journal identity/order mismatch")
            if sequence == 0:
                if record.get("event") != "journal_started":
                    raise ValueError("resource journal has no startup record")
                run_id = record["run_id"]
            elif record["run_id"] != run_id:
                raise ValueError("resource journal run identity changed")
            event = {
                key: value for key, value in record.items() if key not in {"version", "run_id", "sequence", "timestamp"}
            }
            if sequence == 0 or event.get("event") == "journal_closed":
                if set(event) != {"event"}:
                    raise ValueError("invalid resource journal boundary")
                closed = sequence != 0
            else:
                _transition(state, event)
    if run_id is None:
        raise ValueError("empty resource journal")
    return _summary(state, run_id, closed)
