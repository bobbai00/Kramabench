from .baseline_system import BaselineLLMSystem, BaselineLLMSystemOllama

class BaselineLLMSystemGPT4oFewShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="gpt-4o",
            name="BaselineLLMSystemGPT4oFewShot",
            variance="few_shot",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class BaselineLLMSystemGPT4oOneShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="gpt-4o",
            name="BaselineLLMSystemGPT4oOneShot",
            variance="one_shot",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class BaselineLLMSystemGPT4oNaive(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="gpt-4o",
            name="BaselineLLMSystemGPT4oOneShot",
            variance="one_shot",
            verbose=verbose,
            supply_data_snippet=False,
            *args, **kwargs
        )

class BaselineLLMSystemDeepseekR1FewShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="deepseek-ai/DeepSeek-R1",
                          name="BaselineLLMSystemDeepseekR1FewShot", 
                          variance="few_shot", 
                          verbose=verbose, 
                          supply_data_snippet=True,
                          *args, **kwargs)

class BaselineLLMSystemDeepseekR1OneShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="deepseek-ai/DeepSeek-R1",
                          name="BaselineLLMSystemDeepseekR1OneShot", 
                          variance="one_shot", 
                          verbose=verbose, 
                          supply_data_snippet=True,
                          *args, **kwargs)

class BaselineLLMSystemDeepseekR1Naive(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="deepseek-ai/DeepSeek-R1",
                          name="BaselineLLMSystemDeepseekR1Naive", 
                          variance="one_shot", 
                          verbose=verbose, 
                          supply_data_snippet=False,
                          *args, **kwargs)


class BaselineLLMSystemQwen2_5CoderFewShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="Qwen/Qwen2.5-Coder-32B-Instruct", 
                         name="BaselineLLMSystemQwen2_5CoderFewShot", 
                         variance="few_shot", 
                         verbose=verbose, 
                         supply_data_snippet=True,
                         *args, **kwargs)

class BaselineLLMSystemQwen2_5CoderOneShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="Qwen/Qwen2.5-Coder-32B-Instruct", 
                         name="BaselineLLMSystemQwen2_5CoderOneShot", 
                         variance="one_shot", 
                         verbose=verbose, 
                         supply_data_snippet=True,
                         *args, **kwargs)

class BaselineLLMSystemQwen2_5CoderNaive(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="Qwen/Qwen2.5-Coder-32B-Instruct", 
                         name="BaselineLLMSystemQwen2_5CoderNaive", 
                         variance="one_shot", 
                         verbose=verbose, 
                         supply_data_snippet=False,
                         *args, **kwargs)

class BaselineLLMSystemLlama3_3InstructFewShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                         name="BaselineLLMSystemLlama3_3InstructFewShot", 
                         variance="few_shot", 
                         supply_data_snippet=True,
                         verbose=verbose, *args, **kwargs)

class BaselineLLMSystemLlama3_3InstructOneShot(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                         name="BaselineLLMSystemLlama3_3InstructOneShot", 
                         variance="one_shot", 
                         supply_data_snippet=True,
                         verbose=verbose, *args, **kwargs)

class BaselineLLMSystemLlama3_3InstructNaive(BaselineLLMSystem):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
                         name="BaselineLLMSystemLlama3_3InstructNaive", 
                         variance="one_shot", 
                         supply_data_snippet=False,
                         verbose=verbose, *args, **kwargs)

class BaselineLLMSystemDeepseekCoderFewShot(BaselineLLMSystemOllama):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="deepseek-coder:33b", 
                         name="BaselineLLMSystemDeepseekCoderFewShot", 
                         variance="few_shot", 
                         supply_data_snippet=True,
                         verbose=verbose, 
                         *args, **kwargs)

class BaselineLLMSystemDeepseekCoderOneShot(BaselineLLMSystemOllama):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="deepseek-coder:33b", 
                         name="BaselineLLMSystemDeepseekCoderOneShot", 
                         variance="one_shot", 
                         supply_data_snippet=True,
                         verbose=verbose, 
                         *args, **kwargs)

class BaselineLLMSystemDeepseekCoderNaive(BaselineLLMSystemOllama):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="deepseek-coder:33b", 
                         name="BaselineLLMSystemDeepseekCoderNaive", 
                         variance="one_shot", 
                         supply_data_snippet=False,
                         verbose=verbose, 
                         *args, **kwargs)


class BaselineLLMSystemGemma3FewShot(BaselineLLMSystemOllama):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="gemma3:27b", 
                         name="BaselineLLMSystemGemma3FewShot", 
                         variance="few_shot", 
                         supply_data_snippet=True, 
                         verbose=verbose, 
                         *args, 
                         **kwargs)

class BaselineLLMSystemGemma3OneShot(BaselineLLMSystemOllama):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="gemma3:27b", 
                         name="BaselineLLMSystemGemma3OneShot", 
                         variance="one_shot", 
                         verbose=verbose, 
                         supply_data_snippet=True, 
                         *args, 
                         **kwargs)

class BaselineLLMSystemGemma3Naive(BaselineLLMSystemOllama):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(model="gemma3:27b", 
                         name="BaselineLLMSystemGemma3Naive", 
                         variance="one_shot", 
                         verbose=verbose, 
                         supply_data_snippet=False, 
                         *args, 
                         **kwargs)
