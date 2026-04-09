"""Anthropic adapter — Claude models via the official SDK.

Translates Pare's provider-agnostic Message/ToolSchema format to
Anthropic's API format and back. Handles Anthropic-specific features:
- System message as a top-level parameter (not in the messages array)
- tool_result as a user message with tool_result content blocks
- cache_control hints on system prompt and early messages
- Streaming via the Messages stream API

This adapter is ~200 lines because we handle every translation explicitly.
Each provider gets its own adapter to avoid the "universal translator"
anti-pattern that litellm suffers from.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

import anthropic

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
# Translation helpers
# ---------------------------------------------------------------------------


def _stop_reason_from_anthropic(reason: str) -> StopReason:
    """Map Anthropic's stop_reason string to our enum."""
    match reason:
        case "end_turn":
            return StopReason.END_TURN
        case "tool_use":
            return StopReason.TOOL_USE
        case "max_tokens":
            return StopReason.MAX_TOKENS
        case _:
            logger.warning("Unknown Anthropic stop_reason: %s, defaulting to END_TURN", reason)
            return StopReason.END_TURN


def _build_anthropic_tools(tools: list[ToolSchema]) -> list[dict]:
    """Convert our ToolSchema list to Anthropic's tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _build_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict]]:
    """Split messages into system prompt + message array.

    Anthropic requires the system prompt as a separate parameter, not in the
    messages list. We extract the first system message and convert the rest.

    Returns (system_prompt, messages_list).
    """
    system_prompt = ""
    api_messages: list[dict] = []

    for msg in messages:
        if msg.role == "system":
            # Anthropic: system is a top-level param, not a message
            system_prompt = msg.text_content()
            continue

        if msg.role == "tool_result":
            # Anthropic: tool results are sent as user messages with
            # content blocks of type "tool_result"
            if isinstance(msg.content, str):
                # Shouldn't happen in well-formed conversations, but handle it
                api_messages.append({"role": "user", "content": msg.content})
                continue

            content_blocks = []
            for block in msg.content:
                if block.type == ContentBlockType.TOOL_RESULT:
                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_call_id,
                            "content": block.text,
                        }
                    )
            api_messages.append({"role": "user", "content": content_blocks})
            continue

        if msg.role == "assistant":
            if isinstance(msg.content, str):
                api_messages.append({"role": "assistant", "content": msg.content})
            else:
                content_blocks = []
                for block in msg.content:
                    if block.type == ContentBlockType.TEXT:
                        content_blocks.append({"type": "text", "text": block.text})
                    elif block.type == ContentBlockType.TOOL_USE and block.tool_call:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.tool_call.id,
                                "name": block.tool_call.name,
                                "input": block.tool_call.arguments,
                            }
                        )
                api_messages.append({"role": "assistant", "content": content_blocks})
            continue

        # user messages
        api_messages.append({"role": "user", "content": msg.text_content()})

    return system_prompt, api_messages


def _parse_anthropic_response(response: anthropic.types.Message) -> LLMResponse:
    """Convert an Anthropic Message response to our LLMResponse."""
    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input) if block.input else {},
                )
            )

    usage = TokenUsage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cache_create_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    )

    return LLMResponse(
        content="\n".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=_stop_reason_from_anthropic(response.stop_reason or "end_turn"),
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AnthropicAdapter(LLMAdapter):
    """LLM adapter for Anthropic Claude models.

    Usage:
        adapter = AnthropicAdapter(model="claude-sonnet-4-20250514")
        response = await adapter.chat(messages, tools)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        profile: ModelProfile | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: str | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        resolved_profile = profile or get_profile(model)
        super().__init__(model, resolved_profile, temperature, max_tokens)

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._retry_policy = retry_policy or RetryPolicy()

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        system_prompt, api_messages = _build_anthropic_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _build_anthropic_tools(tools)

        response = await with_retry(
            lambda: self._client.messages.create(**kwargs),
            policy=self._retry_policy,
        )

        return _parse_anthropic_response(response)

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, api_messages = _build_anthropic_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if tools:
            kwargs["tools"] = _build_anthropic_tools(tools)

        # Accumulate tool call arguments across deltas
        current_tool_id: str = ""
        current_tool_name: str = ""
        current_tool_json: str = ""

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_json = ""
                        yield StreamChunk(
                            type="tool_call_start",
                            tool_call=ToolCallRequest(
                                id=block.id,
                                name=block.name,
                                arguments={},
                            ),
                        )
                    elif block.type == "text":
                        # Text block start — no content yet
                        pass

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamChunk(type="text_delta", content=delta.text)
                    elif delta.type == "input_json_delta":
                        current_tool_json += delta.partial_json
                        yield StreamChunk(
                            type="tool_call_delta",
                            content=delta.partial_json,
                        )

                elif event.type == "content_block_stop":
                    if current_tool_id:
                        # Parse accumulated JSON for the completed tool call
                        import json

                        try:
                            arguments = json.loads(current_tool_json) if current_tool_json else {}
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to parse tool call JSON: %s", current_tool_json[:200]
                            )
                            arguments = {}

                        yield StreamChunk(
                            type="tool_call_end",
                            tool_call=ToolCallRequest(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=arguments,
                            ),
                        )
                        current_tool_id = ""
                        current_tool_name = ""
                        current_tool_json = ""

                elif event.type == "message_stop":
                    pass

            # Final message with usage
            final = await stream.get_final_message()
            usage = TokenUsage(
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                cache_read_tokens=getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                cache_create_tokens=getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
            )
            yield StreamChunk(type="usage", usage=usage)

    def count_tokens(self, messages: list[Message]) -> int:
        """Estimate token count using character heuristic.

        Anthropic's token counting API is a separate API call that adds
        latency and cost. For the compaction threshold check, a rough
        estimate is sufficient. We use ~3.5 chars per token as the
        heuristic for English text with code.

        For exact counts (e.g., before a critical compaction decision),
        the caller should use the Anthropic count_tokens API directly.
        """
        total_chars = 0
        for msg in messages:
            total_chars += len(msg.text_content())
        return int(total_chars / 3.5)
