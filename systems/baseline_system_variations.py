import os

from systems.baseline_system import BaselineLLMSystem

class BaselineLLMSystemGPT4oFewShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="gpt-4o", name="BaselineLLMSystemGPT4oFewShot", variance="few_shot", verbose=verbose, *args, **kwargs)
