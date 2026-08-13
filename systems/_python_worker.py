# -*- coding: utf-8 -*-
"""
Persistent Python worker for ClaudeCodeSystem's `run_python` tool.

Reads one JSON request per line on stdin, executes the snippet in a module
namespace that survives across requests, and writes one JSON response per
line on stdout.

The snippet's own stdout/stderr are captured into a buffer, so the protocol
stream on real stdout is never polluted by user prints. The real stdout fd is
duplicated before anything is redirected.

Request:  {"code": "<python source>"}
Response: {"output": "<captured stdout+stderr>", "error": "<traceback or empty>"}
"""

import contextlib
import io
import json
import os
import sys
import traceback

# Duplicate the real stdout before any redirection, so protocol writes always
# reach the parent even while the snippet has stdout redirected.
_protocol_fd = os.dup(sys.stdout.fileno())
_protocol = os.fdopen(_protocol_fd, "w", buffering=1)


def main() -> None:
    namespace: dict = {"__name__": "__kramabench_worker__"}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        buffer = io.StringIO()
        error = ""
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(compile(request["code"], "<run_python>", "exec"), namespace)
        except BaseException:
            # Includes SystemExit: a snippet calling sys.exit() must not kill the
            # worker and silently wipe the namespace.
            error = traceback.format_exc()

        _protocol.write(json.dumps({"output": buffer.getvalue(), "error": error}) + "\n")
        _protocol.flush()


if __name__ == "__main__":
    main()
