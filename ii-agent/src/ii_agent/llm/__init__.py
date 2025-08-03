from src.ii_agent.llm.base import LLMClient
from src.ii_agent.llm.anthropic import AnthropicDirectClient
from src.ii_agent.llm.gemini import GeminiDirectClient
from src.ii_agent.llm.ollama import OllamaDirectClient
from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
from src.ii_agent.llm.model_registry import ChutesModelRegistry, ModelCapability


def get_client(client_name: str, **kwargs) -> LLMClient:
    """Get a client for a given client name."""
    if client_name == "anthropic-direct":
        return AnthropicDirectClient(**kwargs)
    elif client_name == "openai-direct":
        from src.ii_agent.llm.openai import OpenAIDirectClient
        return OpenAIDirectClient(**kwargs)
    elif client_name == "gemini-direct":
        return GeminiDirectClient(**kwargs)
    elif client_name == "ollama":
        return OllamaDirectClient(**kwargs)
    elif client_name == "chutes":
        from src.ii_agent.llm.chutes_openai import ChutesOpenAIClient
        return ChutesOpenAIClient(**kwargs)
    else:
        raise ValueError(f"Unknown client name: {client_name}")


__all__ = [
    "LLMClient",
    "OpenAIDirectClient",
    "AnthropicDirectClient",
    "GeminiDirectClient",
    "OllamaDirectClient",
    "get_client",
    "ChutesOpenAIClient",
    "ChutesModelRegistry",
    "ModelCapability",
]