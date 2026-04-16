"""LLM adapter layer — provider-agnostic interface for all LLM interactions."""

from pare.llm.base import (
    ContentBlock,
    LLMAdapter,
    LLMResponse,
    Message,
    ModelProfile,
    StopReason,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
    get_profile,
)
from pare.llm.output_parser import ParseError, parse_json_response
from pare.llm.retry import RetryPolicy

__all__ = [
    "ContentBlock",
    "LLMAdapter",
    "LLMResponse",
    "Message",
    "ModelProfile",
    "ParseError",
    "RetryPolicy",
    "StopReason",
    "StreamChunk",
    "TokenUsage",
    "ToolCallRequest",
    "create_llm",
    "get_profile",
    "parse_json_response",
]


# ---------------------------------------------------------------------------
# Provider presets — maps provider name to (adapter_class, default_kwargs)
# ---------------------------------------------------------------------------

_PROVIDER_PRESETS: dict[str, dict] = {
    "minimax": {
        "base_url": "https://api.minimaxi.com/v1",
        "default_model": "MiniMax-M2.7",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-chat",
    },
    "glm": {
        # Zhipu AI BigModel — OpenAI-compatible at /api/paas/v4
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-5",
    },
}


def create_llm(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    profile: ModelProfile | None = None,
    retry_policy: RetryPolicy | None = None,
) -> LLMAdapter:
    """Factory function to create an LLM adapter by provider name.

    All providers use the OpenAI-compatible adapter. MiniMax and OpenRouter
    have preset base_url values. For custom endpoints (local vLLM, etc.),
    use provider="openai" with a custom base_url.

    Usage:
        llm = create_llm("minimax", model="MiniMax-M2.5", api_key="...")
        llm = create_llm("openrouter", model="deepseek/deepseek-chat", api_key="...")
        llm = create_llm("openai", model="my-model", base_url="http://localhost:8000/v1")
    """
    provider = provider.lower()

    from pare.llm.openai_adapter import OpenAIAdapter

    preset = _PROVIDER_PRESETS.get(provider, {})
    resolved_base_url = base_url or preset.get("base_url")
    resolved_model = model or preset.get("default_model", "gpt-4o")

    return OpenAIAdapter(
        model=resolved_model,
        profile=profile,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=resolved_base_url,
        retry_policy=retry_policy,
    )
