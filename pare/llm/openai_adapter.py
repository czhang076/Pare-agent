"""OpenAI-compatible adapter — supports OpenAI, MiniMax, DeepSeek, Qwen, and any
provider that exposes an OpenAI-compatible chat completions endpoint.

This single adapter covers:
- OpenAI native (api.openai.com)
- MiniMax official API (api.minimax.io/v1) — OpenAI-compatible with quirks
- OpenRouter (openrouter.ai/api/v1) — proxy to DeepSeek, Qwen, etc.
- Any local vLLM/SGLang/Ollama endpoint

The adapter respects ModelProfile to handle behavioral differences:
- When supports_native_tool_use=True: uses OpenAI's native tools parameter
- When supports_native_tool_use=False: injects tool descriptions into system
  prompt and parses text-based tool calls from assistant responses

MiniMax-specific handling:
- temperature clamped to (0.0, 1.0]
- unsupported params (presence_penalty, frequency_penalty) not sent
- tool_calls in assistant message must be preserved in full for multi-turn
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

import openai

from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    LLMAdapter,
    LLMResponse,
    Message,
    ModelProfile,
    StopReason,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
    ToolSchema,
    get_profile,
)
from pare.llm.retry import RetryPolicy, with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text-based tool call parsing (for models without native tool_use)
# ---------------------------------------------------------------------------

# Matches <tool_call>{ ... }</tool_call> or ```tool_call\n{ ... }\n```
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

_TOOL_CALL_FENCE_RE = re.compile(
    r"```(?:tool_call|json)?\s*\n(\{.*?\})\n\s*```",
    re.DOTALL,
)


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Some models (e.g. MiniMax-M2.7, DeepSeek-R1) wrap chain-of-thought
    reasoning in <think> tags.  We strip these so the user only sees the
    final answer, and tool-call parsing isn't confused by thinking content.
    """
    return _THINK_RE.sub("", text).strip()


def _parse_text_tool_calls(text: str) -> list[ToolCallRequest]:
    """Extract tool calls from assistant text when native tool_use is unavailable.

    Supports two formats commonly used in system-prompt-injected tool calling:
    1. XML tags: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. Code fences: ```tool_call\n{"name": "...", "arguments": {...}}\n```

    Returns empty list if no tool calls found.
    """
    calls: list[ToolCallRequest] = []
    call_id = 0

    for pattern in (_TOOL_CALL_XML_RE, _TOOL_CALL_FENCE_RE):
        for match in pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                name = data.get("name", "")
                arguments = data.get("arguments", data.get("parameters", {}))
                if name:
                    calls.append(
                        ToolCallRequest(
                            id=f"text_tc_{call_id}",
                            name=name,
                            arguments=arguments if isinstance(arguments, dict) else {},
                        )
                    )
                    call_id += 1
            except (json.JSONDecodeError, AttributeError):
                logger.warning("Failed to parse text tool call: %s", match.group(0)[:200])

    return calls


def _build_tool_system_prompt(tools: list[ToolSchema]) -> str:
    """Generate a system prompt section describing available tools.

    Used when the model doesn't support native tool_use. The prompt instructs
    the model to emit tool calls in a parseable XML format.
    """
    lines = [
        "## Available Tools",
        "",
        "You have access to the following tools. To call a tool, wrap a JSON "
        "object in <tool_call> tags like this:",
        "",
        '<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>',
        "",
        "You may call multiple tools in one response. After each tool call, "
        "you will receive the result and can continue reasoning.",
        "",
    ]

    for tool in tools:
        lines.append(f"### {tool.name}")
        lines.append(tool.description)
        lines.append(f"Parameters: {json.dumps(tool.parameters, indent=2)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def _stop_reason_from_openai(reason: str | None) -> StopReason:
    match reason:
        case "stop":
            return StopReason.END_TURN
        case "tool_calls":
            return StopReason.TOOL_USE
        case "length":
            return StopReason.MAX_TOKENS
        case _:
            return StopReason.END_TURN


def _build_openai_tools(tools: list[ToolSchema]) -> list[dict]:
    """Convert ToolSchema list to OpenAI's function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _build_openai_messages(
    messages: list[Message],
    tools: list[ToolSchema] | None,
    profile: ModelProfile,
) -> list[dict]:
    """Convert our Message list to OpenAI message format.

    Key differences from Anthropic:
    - System message goes in the messages array (role="system")
    - Tool results use role="tool" with tool_call_id
    - Tool calls live in message.tool_calls, not in content blocks

    When profile.supports_native_tool_use is False, tool descriptions are
    injected into the system prompt and tool_result messages are converted
    to user messages with the result text.
    """
    api_messages: list[dict] = []
    text_mode = not profile.supports_native_tool_use

    for msg in messages:
        if msg.role == "system":
            content = msg.text_content()
            # Inject tool descriptions into system prompt for text-mode models
            if text_mode and tools:
                content = content + "\n\n" + _build_tool_system_prompt(tools)
            api_messages.append({"role": "system", "content": content})
            continue

        if msg.role == "tool_result":
            if text_mode:
                # In text mode, tool results go as user messages
                if isinstance(msg.content, list):
                    parts = []
                    for block in msg.content:
                        if block.type == ContentBlockType.TOOL_RESULT:
                            parts.append(
                                f"Tool result for {block.tool_call_id}:\n{block.text}"
                            )
                    api_messages.append({"role": "user", "content": "\n\n".join(parts)})
                else:
                    api_messages.append({"role": "user", "content": msg.text_content()})
            else:
                # Native mode: each tool result is a separate message
                if isinstance(msg.content, list):
                    for block in msg.content:
                        if block.type == ContentBlockType.TOOL_RESULT:
                            api_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.tool_call_id,
                                    "content": block.text,
                                }
                            )
                else:
                    api_messages.append({"role": "user", "content": msg.text_content()})
            continue

        if msg.role == "assistant":
            if isinstance(msg.content, str):
                api_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Build assistant message with optional tool_calls
                text_parts = []
                tool_calls_list = []
                for block in msg.content:
                    if block.type == ContentBlockType.TEXT:
                        text_parts.append(block.text)
                    elif block.type == ContentBlockType.TOOL_USE and block.tool_call:
                        tool_calls_list.append(
                            {
                                "id": block.tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": block.tool_call.name,
                                    "arguments": json.dumps(block.tool_call.arguments),
                                },
                            }
                        )

                assistant_msg: dict = {"role": "assistant"}
                content = "\n".join(text_parts) if text_parts else None
                if content:
                    assistant_msg["content"] = content
                if tool_calls_list and not text_mode:
                    assistant_msg["tool_calls"] = tool_calls_list
                    if not content:
                        assistant_msg["content"] = ""
                elif not content:
                    assistant_msg["content"] = ""
                api_messages.append(assistant_msg)
            continue

        # user messages
        api_messages.append({"role": "user", "content": msg.text_content()})

    return api_messages


def _parse_openai_response(
    response: openai.types.chat.ChatCompletion,
    profile: ModelProfile,
) -> LLMResponse:
    """Convert an OpenAI ChatCompletion to our LLMResponse."""
    choice = response.choices[0]
    message = choice.message

    raw_text = message.content or ""
    text = _strip_think_tags(raw_text)
    tool_calls: list[ToolCallRequest] = []

    if profile.supports_native_tool_use and message.tool_calls:
        # Native tool_use mode
        for tc in message.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool call arguments: %s",
                    tc.function.arguments[:200],
                )
                arguments = {}

            tool_calls.append(
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )
    elif not profile.supports_native_tool_use and text:
        # Text-based tool call extraction
        tool_calls = _parse_text_tool_calls(text)

    usage = TokenUsage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
    )

    stop_reason = _stop_reason_from_openai(choice.finish_reason)
    # If we found text-based tool calls, override stop_reason
    if tool_calls and not (profile.supports_native_tool_use and message.tool_calls):
        stop_reason = StopReason.TOOL_USE

    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenAIAdapter(LLMAdapter):
    """LLM adapter for OpenAI-compatible APIs.

    Supports:
    - OpenAI native: base_url=None (uses default)
    - MiniMax: base_url="https://api.minimax.io/v1"
    - OpenRouter: base_url="https://openrouter.ai/api/v1"
    - Local vLLM: base_url="http://localhost:8000/v1"

    Usage:
        # MiniMax
        adapter = OpenAIAdapter(
            model="MiniMax-M2.5",
            base_url="https://api.minimax.io/v1",
            api_key="your-minimax-key",
        )

        # OpenAI
        adapter = OpenAIAdapter(model="gpt-4o")

        # DeepSeek via OpenRouter
        adapter = OpenAIAdapter(
            model="deepseek/deepseek-chat",
            base_url="https://openrouter.ai/api/v1",
            profile=ModelProfile(supports_native_tool_use=False, tool_call_format="text"),
        )
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        profile: ModelProfile | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: str | None = None,
        base_url: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        resolved_profile = profile or get_profile(model)
        super().__init__(model, resolved_profile, temperature, max_tokens)

        self._base_url = base_url
        client_kwargs: dict = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._retry_policy = retry_policy or RetryPolicy()

    def _clamp_temperature(self, temperature: float) -> float:
        """Clamp temperature for providers that require specific ranges.

        MiniMax requires temperature in (0.0, 1.0]. We clamp 0.0 to a
        small epsilon to avoid API errors.
        """
        if self._base_url and "minimax" in self._base_url:
            if temperature <= 0.0:
                return 0.01
            if temperature > 1.0:
                return 1.0
        return temperature

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temp = self._clamp_temperature(
            temperature if temperature is not None else self.temperature
        )
        api_messages = _build_openai_messages(messages, tools, self.profile)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temp,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        # Only pass tools param in native mode
        if tools and self.profile.supports_native_tool_use:
            kwargs["tools"] = _build_openai_tools(tools)

        response = await with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            policy=self._retry_policy,
        )

        return _parse_openai_response(response, self.profile)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        temp = self._clamp_temperature(
            temperature if temperature is not None else self.temperature
        )
        api_messages = _build_openai_messages(messages, tools, self.profile)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temp,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools and self.profile.supports_native_tool_use:
            kwargs["tools"] = _build_openai_tools(tools)

        # Track tool call state across stream deltas
        current_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
        accumulated_text = ""

        stream = await with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            policy=self._retry_policy,
        )

        async for chunk in stream:
            # Usage chunk (final)
            if chunk.usage:
                usage = TokenUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )
                yield StreamChunk(type="usage", usage=usage)
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Text content
            if delta.content:
                accumulated_text += delta.content
                yield StreamChunk(type="text_delta", content=delta.content)

            # Native tool calls (streamed as deltas with index)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in current_tool_calls:
                        # New tool call
                        current_tool_calls[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "arguments": "",
                        }
                        if tc_delta.id and tc_delta.function and tc_delta.function.name:
                            yield StreamChunk(
                                type="tool_call_start",
                                tool_call=ToolCallRequest(
                                    id=tc_delta.id,
                                    name=tc_delta.function.name,
                                    arguments={},
                                ),
                            )

                    # Accumulate arguments
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tool_calls[idx]["arguments"] += tc_delta.function.arguments
                        yield StreamChunk(
                            type="tool_call_delta",
                            content=tc_delta.function.arguments,
                        )

            # Check for finish
            if chunk.choices[0].finish_reason:
                # Emit completed tool calls
                for idx in sorted(current_tool_calls.keys()):
                    tc_data = current_tool_calls[idx]
                    try:
                        arguments = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        arguments = {}

                    yield StreamChunk(
                        type="tool_call_end",
                        tool_call=ToolCallRequest(
                            id=tc_data["id"],
                            name=tc_data["name"],
                            arguments=arguments,
                        ),
                    )

                # For text-mode models, parse tool calls from accumulated text
                if not self.profile.supports_native_tool_use and accumulated_text:
                    text_calls = _parse_text_tool_calls(accumulated_text)
                    for tc in text_calls:
                        yield StreamChunk(type="tool_call_start", tool_call=tc)
                        yield StreamChunk(type="tool_call_end", tool_call=tc)

    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count.

        Uses tiktoken for OpenAI models, falls back to character heuristic.
        """
        from pare.llm.token_counter import (
            estimate_tokens_heuristic,
            estimate_tokens_tiktoken,
        )

        # Only use tiktoken for actual OpenAI models
        if not self._base_url or "openai.com" in (self._base_url or ""):
            return estimate_tokens_tiktoken(messages, self.model)

        return estimate_tokens_heuristic(messages)
