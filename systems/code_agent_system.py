# -*- coding: utf-8 -*-
"""
CodeAgentSystem - KramaBench System wrapper for smolagents CodeAgent.
"""

import os
import json
from typing import Dict, List, Optional

from benchmark.benchmark_api import System
from code_agent import CodeAgentWrapper, CodeAgentResult, DEFAULT_API_BASE, DEFAULT_API_KEY
from systems.data_source_utils import expand_data_sources
from utils.answer_parser import parse_answer


# Default max steps (can be overridden by CODE_AGENT_MAX_STEPS env var)
DEFAULT_MAX_STEPS = int(os.environ.get("CODE_AGENT_MAX_STEPS", 50))


class CodeAgentSystem(System):
    """KramaBench System using smolagents CodeAgent."""

    def __init__(
        self,
        model_type: str = "claude-haiku-4.5",
        max_steps: int = DEFAULT_MAX_STEPS,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        verbose: bool = False,
        name: str = "CodeAgentSystem",
        use_fine_grained_prompt: bool = None,
        use_custom_prompt: bool = None,
        use_pitfalls_prompt: bool = None,
        no_action_detail: bool = False,
        max_print_outputs_length: int = None,
        *args, **kwargs
    ):
        super().__init__(name, verbose=verbose, *args, **kwargs)
        self.model_type = model_type
        self.max_steps = max_steps
        self.api_base = api_base
        self.api_key = api_key
        self.use_fine_grained_prompt = use_fine_grained_prompt
        self.use_custom_prompt = use_custom_prompt
        self.use_pitfalls_prompt = use_pitfalls_prompt
        self.no_action_detail = no_action_detail
        # Per-instance stdout-preview cap (the code agent's analog of the dataflow
        # agent's max_operator_result_char_limit). None ⇒ fall back to the
        # CODE_AGENT_MAX_PRINT_OUTPUTS_LENGTH env var / smolagents default.
        self.max_print_outputs_length = max_print_outputs_length
        self.agent: Optional[CodeAgentWrapper] = None
        self.output_dir = f"./system_scratch/{name}"
        self.format_hints: Dict[str, str] = {}  # Map task_id -> format_hint string
        os.makedirs(self.output_dir, exist_ok=True)

    def process_dataset(self, dataset_directory: str | os.PathLike) -> None:
        """Process the dataset directory."""
        self.dataset_directory = dataset_directory
        self.dataset = {}
        for dirpath, _, filenames in os.walk(dataset_directory):
            for fname in filenames:
                rel_path = os.path.relpath(os.path.join(dirpath, fname), dataset_directory)
                self.dataset[rel_path] = None

        if self.verbose:
            print(f"[{self.name}] Found {len(self.dataset)} files in {dataset_directory}")

        # Load format hints
        self._load_format_hints(dataset_directory)

        # Setup agent
        self.agent = CodeAgentWrapper(
            model_type=self.model_type,
            max_steps=self.max_steps,
            api_base=self.api_base,
            api_key=self.api_key,
            verbosity_level=2 if self.verbose else 1,
            use_fine_grained_prompt=self.use_fine_grained_prompt,
            use_custom_prompt=self.use_custom_prompt,
            use_pitfalls_prompt=self.use_pitfalls_prompt,
            no_action_detail=self.no_action_detail,
            max_print_outputs_length=self.max_print_outputs_length,
        )
        self.agent.setup()

    def _load_format_hints(self, dataset_directory: str) -> None:
        """Load format hints for the domain."""
        try:
            parts = str(dataset_directory).rstrip('/').split('/')
            if 'data' in parts:
                data_idx = parts.index('data')
                domain = parts[data_idx + 1]
                project_root = '/'.join(parts[:data_idx])
                hint_path = os.path.join(project_root, 'format_hint', f'{domain}.json')
                if os.path.exists(hint_path):
                    with open(hint_path, 'r') as f:
                        hints = json.load(f)
                        for hint in hints:
                            self.format_hints[hint['id']] = hint.get('format_hint', '')
                    if self.verbose:
                        print(f"[{self.name}] Loaded {len(hints)} format hints from {hint_path}")
        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] Could not load format hints: {e}")

    def _expand_data_sources(self, data_sources: List[str]) -> List[str]:
        """
        Expand wildcard patterns in data_sources to actual file paths.

        Args:
            data_sources: List of file patterns (may contain wildcards)

        Returns:
            List of actual file paths (relative to current working directory)
        """
        return expand_data_sources(
            data_sources=data_sources,
            dataset_directory=self.dataset_directory,
            all_files=list(self.dataset.keys()),
            verbose=self.verbose
        )

    def serve_query(
        self,
        query: str,
        query_id: str = "default-0",
        subset_files: Optional[List[str]] = None
    ) -> Dict:
        """Serve a query using the CodeAgent."""
        if not self.agent:
            raise RuntimeError("Call process_dataset() first.")

        # Expand wildcards and build file paths
        if subset_files:
            file_paths = self._expand_data_sources(subset_files)
        else:
            # Use a recursive wildcard instead of listing every file
            file_paths = [os.path.relpath(self.dataset_directory) + "/**/*"]

        if self.verbose:
            print(f"[{self.name}] Query: {query_id}, Files: {len(file_paths)}")

        # Build prompt
        format_hint = self.format_hints.get(query_id, "")
        prompt = f"""You are a data scientist. Answer the following question based on the data files.

Data files available (use these paths to read the data):
{json.dumps(file_paths, indent=2)}

Note: All paths are relative. Some paths may contain wildcards (e.g., "folder/*" or "file-*.csv"). Use glob patterns to match and read those files.

Question: {query}

Answer format: {format_hint}

Your last line MUST BE: **Final Answer: <value>**"""

        # Save outputs
        query_dir = os.path.join(self.output_dir, query_id)
        os.makedirs(query_dir, exist_ok=True)
        with open(os.path.join(query_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        # Reset and run
        self.agent.reset()
        result: CodeAgentResult = self.agent.run(prompt)

        # Save results
        with open(os.path.join(query_dir, "response.txt"), "w") as f:
            f.write(result.response or "(empty)")
        with open(os.path.join(query_dir, "reasoning_trace.json"), "w") as f:
            json.dump(result.reasoning_trace, f, indent=2, default=str)

        # Save stats.json
        stats = {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "cached_tokens": result.cached_tokens,
            "cache_creation_tokens": result.cache_creation_tokens,
            "cost_usd": result.cost_usd,
            "num_steps": result.num_steps,
            "elapsed_seconds": round(result.elapsed_seconds, 2),
        }
        with open(os.path.join(query_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        # Parse answer
        answer = parse_answer(result.response)
        with open(os.path.join(query_dir, "answer.json"), "w") as f:
            json.dump({"answer": answer}, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] Answer: {answer}, Steps: {len(result.reasoning_trace)}, Time: {result.elapsed_seconds:.1f}s")

        return {
            "explanation": {"id": "main-task", "answer": answer},
            "pipeline_code": "",
            "token_usage": result.total_tokens,
            "token_usage_input": result.input_tokens,
            "token_usage_output": result.output_tokens,
            "token_usage_reasoning": result.reasoning_tokens,
            "token_usage_cached": result.cached_tokens,
            "cost_usd": result.cost_usd,
        }

    def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.agent:
            self.agent.cleanup()
            self.agent = None


# Pre-configured variants


































# Model-routing proxy (claude->Anthropic, gpt/o->OpenAI). Code-agent peers that
# hit the SAME upstream as the DataflowSystem so the comparison is apples-to-apples.
PROXY_API_BASE = "http://localhost:8099/v1"






















def _mk_replicate(base_cls, n):
    class _Rep(base_cls):
        def __init__(self, verbose: bool = False, *args, **kwargs):
            super().__init__(verbose=verbose, *args, **kwargs)
            self.name = f"{base_cls.__name__}Replicate{n}"
            self.output_dir = f"./system_scratch/{self.name}"
            os.makedirs(self.output_dir, exist_ok=True)
    _Rep.__name__ = f"{base_cls.__name__}Replicate{n}"
    _Rep.__qualname__ = _Rep.__name__
    return _Rep




# --- CA-guided replicate study (gpt-5-mini via litellm :4000): guided code agent
# at 1k and 5k stdout-preview caps, 5 single-shot samples each (base = rep0,
# + Replicate1-4). Code-agent runs never touch the Texera engine, so these pools
# can run concurrently with dataflow pools.




















# ===========================================================================
#  MODEL-GROUPED CHAR-BUDGET x PROMPT MATRIX (medium reasoning)
#
#  The code-agent peer of the dataflow char-budget sweep. Per model: the
#  stdout-preview cap at 1k / 2k / 5k, with CUSTOM_INSTRUCTIONS off (plain) and
#  on (Guided). Only those two axes move, so any pair is a one-variable read —
#  `...Chars2k` vs `...Chars2kGuided` isolates the prompt, `...Chars1k` vs
#  `...Chars5k` isolates the budget.
#
#  REASONING IS MEDIUM ON EVERY GPT ARM, and the names say so. The bare
#  `gpt-5.2` / `gpt-5-mini` aliases send no reasoning_effort, which probes
#  byte-identical to "none" (0 reasoning tokens) — the pre-existing
#  CodeAgentSystemGpt52Chars* arms are those NO-REASONING baselines and are
#  deliberately left alone. `Med` in a name means the `-medium` litellm alias
#  (see bin/single-node/litellm-config.yaml). gpt-5.6-luna is already pinned
#  medium at the proxy; claude-haiku-4.5 has no reasoning-effort knob at all.
# ===========================================================================
_CHAR_BUDGETS = (("1k", 1000), ("2k", 2000), ("5k", 5000))

#: display tag -> litellm model alias (all medium-reasoning where the knob exists)
_MATRIX_MODELS = (
    ("MiniMed",  "gpt-5-mini-medium"),  # gpt-5-mini @ medium
    ("Gpt52Med", "gpt-5.2-medium"),     # gpt-5.2    @ medium
    ("Luna",     "gpt-5.6-luna"),       # pinned medium at the proxy
    ("Sonnet",   "claude-sonnet-5"),    # 1k/5k only, see _MODEL_BUDGETS
)

#: Replicates per cell. The matrix is guided-only (CUSTOM_INSTRUCTIONS ON):
#: the unguided arms were a prompt A/B that is finished, and keeping them
#: doubles the grid for a comparison nobody runs any more.
_MATRIX_REPS = (0, 1, 2)


def _mk_matrix_arm(model_tag, model_alias, chars_tag, chars, guided, rep=None):
    name = (f"CodeAgentSystem{model_tag}Chars{chars_tag}"
            f"{'Guided' if guided else ''}{'' if rep is None else f'Rep{rep}'}")
    # Never shadow an existing arm: rebinding a name that already has recorded
    # results would silently change what that SUT means.
    if name in globals():
        raise RuntimeError(f"refusing to overwrite existing SUT {name}")

    def __init__(self, verbose: bool = False, *args, **kwargs):
        # max_steps=25 matches every dataflow arm in the factorial; the
        # CodeAgentSystem default is 50, which would make a code-agent vs
        # dataflow read differ on the step budget as well as the substrate.
        kw = {"max_print_outputs_length": chars, "max_steps": 25}
        if guided:
            kw["use_custom_prompt"] = True
        kw.update(kwargs)
        super(cls, self).__init__(model_type=model_alias, name=name, verbose=verbose, *args, **kw)

    cls = type(name, (CodeAgentSystem,), {
        "__init__": __init__,
        "__doc__": (f"{model_alias} code agent, {chars_tag} stdout-preview cap, "
                    f"CUSTOM_INSTRUCTIONS {'ON' if guided else 'OFF'}."),
    })
    return name, cls


#: Per-model budget override. Models absent here take the full _CHAR_BUDGETS
#: sweep; sonnet-5 is registered at the two ends only (1k / 5k).
_MODEL_BUDGETS = {"Sonnet": (("1k", 1000), ("5k", 5000))}

CODE_AGENT_MATRIX_NAMES = []
for _tag, _alias in _MATRIX_MODELS:
    for _ctag, _chars in _MODEL_BUDGETS.get(_tag, _CHAR_BUDGETS):
        for _rep in _MATRIX_REPS:
            _n, _c = _mk_matrix_arm(_tag, _alias, _ctag, _chars, True, _rep)
            globals()[_n] = _c
            CODE_AGENT_MATRIX_NAMES.append(_n)
CODE_AGENT_MATRIX_NAMES = sorted(CODE_AGENT_MATRIX_NAMES)
