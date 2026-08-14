# -*- coding: utf-8 -*-
"""Accurate per-run LLM cost from the token-class breakdown — shared by ALL SUTs
(DataflowSystem and CodeAgentSystem) so cost is computed identically.

Token-class semantics (OpenAI / AI-SDK / litellm convention):
  - input_tokens          : TOTAL prompt tokens (uncached + cached). For OpenAI and
                            the Vercel AI SDK, the prompt count already INCLUDES the
                            cached subset, so uncached = input - cached - cache_write.
  - cached_tokens         : prompt tokens served from cache (billed at the cheaper
                            cache-read rate, ~0.1x input).
  - cache_creation_tokens : Anthropic cache-WRITE tokens (billed ~1.25x input). 0 for
                            OpenAI (no separate cache-write charge).
  - output_tokens         : completion tokens. For OpenAI/Anthropic this ALREADY
                            INCLUDES reasoning/thinking tokens, so reasoning is not
                            added again — it is reported separately for visibility only.

Prices come from litellm.model_cost (maintained); a small fallback table covers
models litellm lacks. Returns a USD float, or None if no price is known (callers
still emit the raw token counts so cost can be recomputed later).
"""
from typing import Optional, Tuple

# Per-token USD fallbacks for models litellm.model_cost may not carry.
# (input, output, cache_read, cache_creation)
_FALLBACK = {
    "claude-haiku-4.5": (1.0e-6, 5.0e-6, 1.0e-7, 1.25e-6),
    "claude-haiku-4-5": (1.0e-6, 5.0e-6, 1.0e-7, 1.25e-6),
    "claude-haiku-4-5-20251001": (1.0e-6, 5.0e-6, 1.0e-7, 1.25e-6),
    "claude-sonnet-4.6": (3.0e-6, 15.0e-6, 3.0e-7, 3.75e-6),
    # OpenAI models. litellm is not installed in this venv (and would not know
    # these ids anyway), so without these entries every gpt arm recorded
    # cost_usd = 0 — which is why the gpt-5.2 scratch dirs carry no cost at all.
    # input/output are the repo's own published rates: compare_tokens.py
    # MODEL_PRICING and result-analysis/.../analyze_result.py COST_CONFIG both
    # give gpt-5.2 $1.75/$14.00 and gpt-5-mini $0.25/$2.00 per M tokens.
    # The cache legs are this module's standard ratios (read = 10% of input,
    # creation = 1.25x input), the same defaults _prices() applies when a
    # litellm entry omits them — only input/output are externally sourced.
    "gpt-5.2": (1.75e-6, 14.0e-6, 0.175e-6, 2.1875e-6),
    "gpt-5-mini": (0.25e-6, 2.0e-6, 0.025e-6, 0.3125e-6),
    # luna/terra publish an explicit cached rate in
    # judgment_runs/mini_star/LUNA_TERRA_FINAL_TABLE.md ("$0.20 / $0.02 / $1.20
    # per M tok" = input / cached / output), so no ratio guessing here.
    "gpt-5.6-luna": (0.20e-6, 1.20e-6, 0.02e-6, 0.25e-6),
    "gpt-5.6-terra": (2.00e-6, 12.0e-6, 0.20e-6, 2.50e-6),
}

_REASONING_EFFORT_SUFFIXES = ("-high", "-medium", "-low")


def _normalize(model: str) -> str:
    m = (model or "").strip()
    # smolagents-style reasoning-effort suffixes are not part of the price key
    for suf in _REASONING_EFFORT_SUFFIXES:
        if m.endswith(suf):
            m = m[: -len(suf)]
            break
    # strip an "openai/" / "anthropic/" provider prefix if present
    if "/" in m:
        m = m.split("/", 1)[1]
    return m


def _prices(model: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (input, output, cache_read, cache_creation) per-token USD, or None."""
    m = _normalize(model)
    try:
        import litellm
        c = litellm.model_cost.get(m) or litellm.model_cost.get(model)
        if c and c.get("input_cost_per_token") is not None:
            in_c = c.get("input_cost_per_token") or 0.0
            out_c = c.get("output_cost_per_token") or 0.0
            cread = c.get("cache_read_input_token_cost")
            cwrite = c.get("cache_creation_input_token_cost")
            return (in_c, out_c,
                    cread if cread is not None else in_c * 0.1,
                    cwrite if cwrite is not None else in_c * 1.25)
    except Exception:
        pass
    f = _FALLBACK.get(m) or _FALLBACK.get(model)
    return f


def has_price(model: str) -> bool:
    return _prices(model) is not None


def compute_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    input_includes_cached: bool = True,
) -> Optional[float]:
    p = _prices(model)
    if not p:
        return None
    in_c, out_c, cread_c, cwrite_c = p
    inp = int(input_tokens or 0)
    out = int(output_tokens or 0)
    cached = int(cached_tokens or 0)
    cwrite = int(cache_creation_tokens or 0)
    if input_includes_cached:
        uncached = max(0, inp - cached - cwrite)
    else:
        uncached = max(0, inp)
    cost = uncached * in_c + cached * cread_c + cwrite * cwrite_c + out * out_c
    return round(cost, 6)
