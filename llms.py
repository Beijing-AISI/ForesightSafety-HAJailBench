from llm_base import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from oai import OpenAiChatLLM, OpenAiChatLLMConfig, OpenAiLLM, OpenAiLLMConfig


def create_llm(config: BaseLLMConfig) -> BaseLLM:
    if isinstance(config, OpenAiLLMConfig):
        return OpenAiLLM(config)
    if isinstance(config, OpenAiChatLLMConfig):
        return OpenAiChatLLM(config)
    raise ValueError(f"Cannot Create LLM from Config: {config}")
