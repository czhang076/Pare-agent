"""Core data types and abstract base class for LLM adapters.

Every module in pare talks to LLMs through these types. The adapter ABC
defines the contract that each provider (Anthropic, OpenAI, OpenRouter)
must implement. The rest of the framework never imports a provider SDK
directly — only pare.llm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StopReason(Enum):
    """Why the LLM stopped generating."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"


class ContentBlockType(Enum):
    """Discriminator for content blocks inside a Message."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


# ---------------------------------------------------------------------------
# Data classes — provider-agnostic message format
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCallRequest:
    """A single tool invocation requested by the LLM.

    Attributes:
        id: Opaque identifier used to correlate the result back to the call.
        name: The tool name (must match a registered tool).
        arguments: Parsed parameter dict (already JSON-decoded).
    """

    id: str
    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single LLM call."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """One block inside a Message's content list.

    A message with mixed content (e.g. text + tool call) is represented as
    a list of ContentBlocks rather than a flat string.
    """

    type: ContentBlockType
    text: str = ""
    tool_call: ToolCallRequest | None = None
    tool_call_id: str = ""


@dataclass(slots=True)
class Message:
    """A single message in the conversation.

    Roles:
        system   — system prompt (only Anthropic treats this specially)
        user     — human turn
        assistant — model turn
        tool_result — result of a tool call (sent back as a user turn)
    """

    role: str
    content: str | list[ContentBlock]

    def text_content(self) -> str:
        """Extract plain text from content, regardless of shape."""
        if isinstance(self.content, str):
            return self.content
        return "".join(block.text for block in self.content if block.text)

    def tool_calls(self) -> list[ToolCallRequest]:
        """Extract all tool call requests from content blocks."""
        if isinstance(self.content, str):
            return []
        return [
            block.tool_call
            for block in self.content
            if block.type == ContentBlockType.TOOL_USE and block.tool_call is not None
        ]


# ---------------------------------------------------------------------------
# LLM response types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Complete (non-streaming) response from an LLM call."""

    content: str
    tool_calls: list[ToolCallRequest]
    stop_reason: StopReason
    usage: TokenUsage


@dataclass(slots=True)
class StreamChunk:
    """One chunk from a streaming LLM response.

    Chunk types:
        text_delta       — incremental text output
        tool_call_start  — beginning of a tool call (name + id)
        tool_call_delta  — incremental tool call argument JSON
        tool_call_end    — tool call argument stream complete
        usage            — final token usage (last chunk)
    """

    type: str
    content: str = ""
    tool_call: ToolCallRequest | None = None
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Model capability profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Declares what a specific model can do.

    The adapter layer reads this to decide how to format requests and parse
    responses. For models that don't support native tool_use, the adapter
    falls back to describing tools in the system prompt and parsing text-
    based tool calls from the assistant's response.
    """

    supports_native_tool_use: bool = True
    supports_structured_json: bool = True
    supports_system_message: bool = True
    supports_cache_control: bool = False
    tool_call_format: str = "native"  # "native" | "text"
    max_context_tokens: int = 128_000
    max_output_tokens: int = 4_096


# ---------------------------------------------------------------------------
# Tool schema (what the LLM sees)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Provider-agnostic tool definition sent to the LLM."""

    name: str
    description: str
    parameters: dict  # JSON Schema


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------


class LLMAdapter(ABC):
    """Provider-agnostic interface for LLM interactions.

    Each provider implements three methods:
        chat        — full request/response (blocks until complete)
        chat_stream — streaming response (yields chunks)
        count_tokens — estimate token count for a message list

    The adapter is responsible for:
    - Translating Message/ToolSchema to provider-specific format
    - Handling provider-specific quirks (tool result role, schema keys, etc.)
    - Applying retry logic (via RetryPolicy)
    - Respecting ModelProfile capabilities (falling back to text tool calls
      when native tool_use is unavailable)
    """

    def __init__(
        self,
        model: str,
        profile: ModelProfile | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.profile = profile or ModelProfile()
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a conversation and get a complete response."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send a conversation and stream the response chunk by chunk."""

    @abstractmethod
    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a message list.

        This is used by the context manager to decide when compaction is
        needed. Implementations should prefer speed over precision — a
        character-based heuristic is acceptable for the threshold check.
        """


# ---------------------------------------------------------------------------
# Default model profiles for known models
# ---------------------------------------------------------------------------

DEFAULT_PROFILES: dict[str, ModelProfile] = {
    # Anthropic
    "claude-sonnet-4-20250514": ModelProfile(
        supports_native_tool_use=True,
        supports_structured_json=True,
        supports_system_message=True,
        supports_cache_control=True,
        max_context_tokens=200_000,
        max_output_tokens=8_192,
    ),
    "claude-haiku-3-5-20241022": ModelProfile(
        supports_native_tool_use=True,
        supports_structured_json=True,
        supports_system_message=True,
        supports_cache_control=True,
        max_context_tokens=200_000,
        max_output_tokens=8_192,
    ),
    # OpenAI
    "gpt-4o": ModelProfile(
        supports_native_tool_use=True,
        supports_structured_json=True,
        supports_system_message=True,
        supports_cache_control=False,
        max_context_tokens=128_000,
        max_output_tokens=4_096,
    ),
    # MiniMax (via api.minimax.io, OpenAI-compatible)
    "MiniMax-M2.5": ModelProfile(
        supports_native_tool_use=True,
        supports_structured_json=True,
        supports_system_message=True,
        supports_cache_control=False,
        max_context_tokens=204_800,
        max_output_tokens=4_096,
    ),
    "MiniMax-M2.5-highspeed": ModelProfile(
        supports_native_tool_use=True,
        supports_structured_json=True,
        supports_system_message=True,
        supports_cache_control=False,
        max_context_tokens=204_800,
        max_output_tokens=4_096,
    ),
    "MiniMax-M2.7": ModelProfile(
        supports_native_tool_use=True,
        supports_structured_json=True,
        supports_system_message=True,
        supports_cache_control=False,
        max_context_tokens=204_800,
        max_output_tokens=4_096,
    ),
    # DeepSeek (via OpenRouter)
    "deepseek/deepseek-chat": ModelProfile(
        supports_native_tool_use=False,
        supports_structured_json=False,
        supports_system_message=True,
        supports_cache_control=False,
        tool_call_format="text",
        max_context_tokens=64_000,
        max_output_tokens=4_096,
    ),
    # Qwen (via OpenRouter)
    "qwen/qwen-2.5-coder-32b-instruct": ModelProfile(
        supports_native_tool_use=False,
        supports_structured_json=False,
        supports_system_message=True,
        supports_cache_control=False,
        tool_call_format="text",
        max_context_tokens=32_000,
        max_output_tokens=4_096,
    ),
}


def get_profile(model: str) -> ModelProfile:
    """Look up the default profile for a model, falling back to defaults."""
    return DEFAULT_PROFILES.get(model, ModelProfile())
