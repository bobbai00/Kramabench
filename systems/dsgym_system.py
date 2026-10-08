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

"""
DSGymSystem - KramaBench System wrapper for DSGym's ReAct agent.

DSGym (https://github.com/fannie1208/DSGym) is a separate checkout with its own
interpreter (it requires Python 3.12). Each query runs `dsgym_runner.py` there:
upstream's `create_custom_task` builds the prompt and upstream's `ReActDSAgent`
solves it, writing `<python>` blocks that an executor manager runs in a
persistent Jupyter kernel. The manager is DSGym's docker setup or any server
with the same HTTP contract; start it first and pass its URL.
"""

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Optional

from benchmark.benchmark_api import System
from code_agent import DEFAULT_API_BASE, DEFAULT_API_KEY
from systems.cost_utils import compute_cost
from systems.data_source_utils import expand_data_sources
from systems.dsgym_runner import usage_totals
from utils.answer_parser import parse_explicit_answer

HARNESS_ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("dsgym_runner.py")
API_KEY_VARIABLE = "DSGYM_API_KEY"
PATH_NOTE = (
    'All paths are relative. Some paths may contain wildcards (e.g., "folder/*" or "file-*.csv"). '
    "Use glob patterns to match and read those files."
)


class DSGymSystem(System):
    """KramaBench System using DSGym's ReActDSAgent."""

    def __init__(
        self,
        model_type: str = "gpt-5.6-luna",
        max_turns: int = 15,
        max_output_length: int = 5000,
        manager_url: str = None,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        dsgym_home: str = None,
        dsgym_python: str = None,
        verbose: bool = False,
        name: str = "DSGymSystem",
        *args,
        **kwargs,
    ):
        super().__init__(name, verbose=verbose, *args, **kwargs)
        self.model_type = model_type
        # Upstream defaults: 15 turns, and 5,000 characters of output per
        # execution (the `max_stdout_length` its cleaner declares; the runner
        # is what applies it).
        self.max_turns = max_turns
        self.max_output_length = max_output_length
        self.manager_url = manager_url or os.environ.get(
            "DSGYM_MANAGER_URL", "http://localhost:5000"
        )
        self.api_base = api_base
        self.api_key = api_key
        self.dsgym_home = Path(
            dsgym_home or os.environ.get("DSGYM_HOME") or HARNESS_ROOT.parent / "DSGym"
        ).resolve()
        self.dsgym_python = str(
            dsgym_python
            or os.environ.get("DSGYM_PYTHON")
            or self.dsgym_home / ".venv/bin/python"
        )
        self.output_dir = f"./system_scratch/{name}"
        self.format_hints: Dict[str, str] = {}  # Map task_id -> format_hint string
        os.makedirs(self.output_dir, exist_ok=True)

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """Process the dataset directory."""
        self.dataset_directory = dataset_directory
        self.dataset = {}
        for dirpath, _, filenames in os.walk(dataset_directory):
            for fname in filenames:
                rel_path = os.path.relpath(
                    os.path.join(dirpath, fname), dataset_directory
                )
                self.dataset[rel_path] = None

        # Format hints of the domain: format_hint/<domain>.json
        domain = Path(dataset_directory).parent.name
        hint_path = HARNESS_ROOT / "format_hint" / f"{domain}.json"
        if hint_path.exists():
            for hint in json.loads(hint_path.read_text()):
                self.format_hints[hint["id"]] = hint.get("format_hint", "")

        if self.verbose:
            print(
                f"[{self.name}] Found {len(self.dataset)} files in {dataset_directory}, "
                f"{len(self.format_hints)} format hints"
            )

    def serve_query(
        self,
        query: str,
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None,
    ) -> Dict:
        """Serve a query using DSGym's agent."""
        if self.dataset_directory is None:
            raise RuntimeError("Call process_dataset() first.")

        # Same file list the other systems receive
        if subset_files:
            file_paths = expand_data_sources(
                data_sources=subset_files,
                dataset_directory=self.dataset_directory,
                all_files=list(self.dataset.keys()),
                verbose=self.verbose,
            )
        else:
            file_paths = [os.path.relpath(self.dataset_directory) + "/**/*"]

        # The task in DSGym's custom-task template: the question with its
        # answer format, the note on paths as context, the files as locations.
        format_hint = self.format_hints.get(query_id, "")
        query_dir = Path(self.output_dir).resolve() / query_id
        query_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "dsgym_home": str(self.dsgym_home),
            "model": self.model_type,
            "api_base": self.api_base,
            "api_key_env": API_KEY_VARIABLE,
            "manager_url": self.manager_url,
            "max_turns": self.max_turns,
            "max_output_length": self.max_output_length,
            "output_dir": str(query_dir),
            "task": {
                "query": (
                    f"{query}\nAnswer format: {format_hint}" if format_hint else query
                ),
                "data_files": file_paths,
                "context": PATH_NOTE,
            },
        }
        request_path = query_dir / "request.json"
        request_path.write_text(
            json.dumps(request, indent=2, ensure_ascii=False) + "\n"
        )

        if self.verbose:
            print(f"[{self.name}] Query: {query_id}, Files: {len(file_paths)}")

        # Run the agent in DSGym's interpreter. It gets the model key and nothing else.
        started = time.time()
        error = None
        with open(query_dir / "runner.log", "w") as log:
            process = subprocess.Popen(
                [self.dsgym_python, str(RUNNER), str(request_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=query_dir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    API_KEY_VARIABLE: self.api_key,
                },
            )
            try:
                process.wait()
            except (Exception, SystemExit) as stop:
                # A wall-clock limit ends the wait: the task fails with the usage
                # recorded so far instead of aborting the workload.
                process.kill()
                process.wait()
                error = f"{type(stop).__name__}: {stop}"
        wall_seconds = time.time() - started

        result_path = query_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            error = (
                error
                or f"DSGym runner exited with code {process.returncode}; see runner.log"
            )
            result = {
                "answer": "",
                "turns": 0,
                "error": error,
                "conversation": [],
                **usage_totals(query_dir / "model-request-trace.jsonl"),
            }
        cost_usd = (
            compute_cost(
                self.model_type,
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cached_tokens=result["cached_tokens"],
                cache_creation_tokens=result["cache_creation_tokens"],
            )
            or 0.0
        )

        # Save results
        prompt_path = query_dir / "prompt.json"
        if prompt_path.exists():
            system_message, user_message = json.loads(prompt_path.read_text())
            (query_dir / "prompt.txt").write_text(user_message["content"])
            (query_dir / "system-info.json").write_text(
                json.dumps(
                    {"systemPrompt": system_message["content"]},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        (query_dir / "response.txt").write_text(result["answer"] or "(empty)")
        with open(query_dir / "reasoning_trace.json", "w") as f:
            json.dump(result["conversation"], f, indent=2, ensure_ascii=False)

        # Save stats.json
        stats = {
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["input_tokens"] + result["output_tokens"],
            "reasoning_tokens": result["reasoning_tokens"],
            "cached_tokens": result["cached_tokens"],
            "cache_creation_tokens": result["cache_creation_tokens"],
            "cost_usd": cost_usd,
            "num_steps": result["turns"],
            "elapsed_seconds": round(result.get("elapsed_seconds", wall_seconds), 2),
            "wall_seconds": round(wall_seconds, 2),
            "model_calls": result["model_calls"],
            "failed_model_calls": result["failed_model_calls"],
            "model_seconds": round(result["model_seconds"], 2),
            "executions": result.get("executions", 0),
            "execution_seconds": round(result.get("execution_seconds", 0.0), 2),
            "truncated_outputs": result.get("truncated_outputs", 0),
            "error": error or result["error"],
        }
        with open(query_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # The <answer> tag's content is the answer
        answer = parse_explicit_answer(result["answer"])
        with open(query_dir / "answer.json", "w") as f:
            json.dump({"answer": answer}, f, indent=2)

        if self.verbose:
            print(
                f"[{self.name}] Answer: {answer}, Turns: {result['turns']}, Time: {wall_seconds:.1f}s"
            )

        return {
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": "",
            "token_usage": stats["total_tokens"],
            "token_usage_input": stats["input_tokens"],
            "token_usage_output": stats["output_tokens"],
            "token_usage_reasoning": stats["reasoning_tokens"],
            "token_usage_cached": stats["cached_tokens"],
            "cost_usd": cost_usd,
        }


# Pre-configured variants


class DSGymSystemLuna(DSGymSystem):
    """gpt-5.6-luna (medium reasoning at the proxy), 25 turns, 5,000 characters of output per execution."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        kw = {"max_turns": 25, "max_output_length": 5000}
        kw.update(kwargs)
        super().__init__(
            model_type="gpt-5.6-luna",
            name="DSGymSystemLuna",
            verbose=verbose,
            *args,
            **kw,
        )
