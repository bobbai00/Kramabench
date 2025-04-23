import os

from systems.baseline_system import BaselineLLMSystem

class BaselineLLMSystemGPT4oFewShot(BaselineLLMSystem):
    def __init__(self, output_dir=os.path.join(os.getcwd(), 'test_outputs/BaselineLLMSystemGPT4oFewShot'), verbose=False, *args, **kwargs):
        super().__init__(model="gpt-4o", name="BaselineLLMSystemGPT4oFewShot", variance="few_shot", verbose=False, *args, **kwargs)