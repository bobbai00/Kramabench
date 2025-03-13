from systems.baseline_example import ExampleBaselineSystem
from systems.mixture_agent_example import ExamplemixtureAgentSystem
from systems.reflection_example import ExampleReflectionSystem


configs = {
    "baseline-gpt-4o-mini": {
        "class": ExampleBaselineSystem,
        "kwargs": {"model": "gpt-4o-mini"},
    },
    "baseline-gpt-4o": {
        "class": ExampleBaselineSystem,
        "kwargs": {"model": "gpt-4o"},
    },
    "baseline-gemma-2b-it": {
        "class": ExampleBaselineSystem,
        "kwargs": {
            "model": "google/gemma-2b-it",
        },
    },
    "baseline-llama-2-13b-chat-hf": {
        "class": ExampleBaselineSystem,
        "kwargs": {
            "model": "meta-llama/Llama-2-13b-chat-hf",
        },
    },
    "baseline-deepseek-r1-distill-qwen-14b": {
        "class": ExampleBaselineSystem,
        "kwargs": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        },
    },
    "reflection-gpt-4o-mini": {
        "class": ExampleReflectionSystem,
        "kwargs": {
            "executor": "gpt-4o-mini",
            "reflector": "gpt-4o-mini",
        },
    },
    "reflection-gpt-4o": {
        "class": ExampleReflectionSystem,
        "kwargs": {
            "executor": "gpt-4o",
            "reflector": "gpt-4o",
        },
    },
    "mixture-agent-gpt-4o-mini": {
        "class": ExamplemixtureAgentSystem,
        "kwargs": {
            "suggesters": ["gpt-4o-mini", "gpt-4o-mini"],
            "merger": "gpt-4o-mini",
        },
    },
    "mixture-agent-gpt-4o": {
        "class": ExamplemixtureAgentSystem,
        "kwargs": {
            "suggesters": ["gpt-4o-mini", "gpt-4o-mini"],
            "merger": "gpt-4o",
        },
    },
}


def system_selector(sut: str = None, verbose=False):
    # keep this for system level arguments
    extra_kwargs = {"verbose": verbose}
    return configs[sut]["class"](**configs[sut]["kwargs"], **extra_kwargs)
