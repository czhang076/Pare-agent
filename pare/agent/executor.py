"""ReAct executor — the core agent loop.

This module implements the bounded ReAct (Reason + Act) loop that drives
the agent. Each iteration:
  1. Send conversation to LLM (with tool schemas)
  2. LLM responds with text and/or tool calls
  3. Guardrails check each tool call
  4. Execute approved tool calls
  5. Feed results back as tool_result messages
  6. Repeat until: LLM stops calling tools, budget exhausted, or error

The executor is deliberately simple — it does NOT know about planning,
orient, or replan. Those are layered on top by the orchestrator. This
separation keeps the core loop testable and composable.

Message flow:
    [system prompt] + [user task] → LLM → [text + tool_calls]
    → execute tools → [tool_results] → LLM → ...
    → LLM returns end_turn (no tool calls) → done
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    LLMAdapter,
    LLMResponse,
    Message,
    StopReason,
    StreamChunk,
    ToolCallRequest,
)
from pare.tools.base import ToolContext, ToolRegistry, ToolResult
from pare.agent.guardrails import Guardrails
from pare.telemetry import EventLog

import re

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionResult:
    """Final result of a ReAct execution loop.

    Attributes:
        success: Whether the agent completed without hitting guardrails.
        output: The final text output from the LLM.
        messages: The full conversation history (for context management).
        tool_call_count: Total tool calls made.
        stop_reason: Why the loop ended.
    """

    success: bool
    output: str
    messages: list[Message]
    tool_call_count: int = 0
    stop_reason: str = "end_turn"  # end_turn | budget_exhausted | error | max_tokens


@dataclass(slots=True)
class ToolCallEvent:
    """Emitted for each tool call during execution (for UI rendering)."""

    tool_name: str
    arguments: dict
    result: ToolResult | None = None
    blocked_reason: str = ""
    duration: float = 0.0


# Callback type for streaming events to the UI
OnToolCall = Callable[[ToolCallEvent], None]
OnTextDelta = Callable[[str], None]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ReActExecutor:
    """Bounded ReAct loop that connects LLM to tools.

    Usage:
        executor = ReActExecutor(llm, registry, guardrails)
        result = await executor.run(
            system_prompt="You are a coding agent...",
            user_message="Fix the bug in main.py",
            context=ToolContext(cwd=Path(".")),
        )
    """

    def __init__(
        self,
        llm: LLMAdapter,
        registry: ToolRegistry,
        guardrails: Guardrails | None = None,
        event_log: EventLog | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.guardrails = guardrails or Guardrails()
        self.event_log = event_log
        self.max_iterations = max_iterations

    async def run(
        self,
        system_prompt: str,
        user_message: str,
        context: ToolContext,
        *,
        messages: list[Message] | None = None,
        on_tool_call: OnToolCall | None = None,
        on_text_delta: OnTextDelta | None = None,
    ) -> ExecutionResult:
        """Run the ReAct loop until the LLM stops or guardrails trigger.

        Args:
            system_prompt: System instructions for the LLM.
            user_message: The user's task description.
            context: Tool execution context (cwd, env, permissions).
            messages: Optional pre-existing conversation history to continue.
            on_tool_call: Callback for each tool call event (for UI).
            on_text_delta: Callback for streaming text (for UI).

        Returns:
            ExecutionResult with the final state.
        """
        # Build initial message list
        if messages is not None:
            conversation = list(messages)
        else:
            conversation = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_message),
            ]

        tool_schemas = self.registry.get_all_schemas()
        total_calls = 0
        iterations = 0
        final_text = ""

        while True:
            # Check per-step iteration limit (from planner budget)
            if self.max_iterations is not None and iterations >= self.max_iterations:
                return ExecutionResult(
                    success=False,
                    output=final_text,
                    messages=conversation,
                    tool_call_count=total_calls,
                    stop_reason="budget_exhausted",
                )

            # Check total budget before calling LLM
            if self.guardrails.budget_remaining <= 0:
                return ExecutionResult(
                    success=False,
                    output=final_text,
                    messages=conversation,
                    tool_call_count=total_calls,
                    stop_reason="budget_exhausted",
                )

            # Call LLM
            self._log("llm_request", message_count=len(conversation))
            start_time = time.time()

            try:
                if on_text_delta:
                    # Streaming mode: collect response while streaming text
                    response = await self._stream_response(
                        conversation, tool_schemas, on_text_delta
                    )
                else:
                    response = await self.llm.chat(conversation, tool_schemas)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                self._log("llm_error", error=str(e), error_type=type(e).__name__)
                return ExecutionResult(
                    success=False,
                    output=final_text,
                    messages=conversation,
                    tool_call_count=total_calls,
                    stop_reason="error",
                )

            iterations += 1
            duration = time.time() - start_time
            self._log(
                "llm_response",
                duration=round(duration, 2),
                stop_reason=response.stop_reason.value,
                tool_call_count=len(response.tool_calls),
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": response.usage.cache_read_tokens,
                    "cache_create_tokens": response.usage.cache_create_tokens,
                },
            )

            final_text = response.content

            # Build assistant message with content blocks
            assistant_blocks = self._build_assistant_blocks(response)
            conversation.append(Message(role="assistant", content=assistant_blocks))

            # If no tool calls, we're done
            if not response.tool_calls:
                stop = "max_tokens" if response.stop_reason == StopReason.MAX_TOKENS else "end_turn"
                return ExecutionResult(
                    success=True,
                    output=response.content,
                    messages=conversation,
                    tool_call_count=total_calls,
                    stop_reason=stop,
                )

            # Execute tool calls
            result_blocks: list[ContentBlock] = []

            for tc in response.tool_calls:
                event = ToolCallEvent(tool_name=tc.name, arguments=tc.arguments)
                start_time = time.time()

                # Check guardrails
                block_msg = self.guardrails.check(tc.name, tc.arguments)
                if block_msg:
                    logger.info("Guardrail blocked %s: %s", tc.name, block_msg)
                    self._log("guardrail", tool=tc.name, message=block_msg)
                    event.blocked_reason = block_msg
                    event.duration = time.time() - start_time
                    if on_tool_call:
                        on_tool_call(event)

                    result_blocks.append(
                        ContentBlock(
                            type=ContentBlockType.TOOL_RESULT,
                            tool_call_id=tc.id,
                            text=f"[BLOCKED] {block_msg}",
                        )
                    )
                    continue

                # Check if tool exists
                if tc.name not in self.registry:
                    error_msg = f"Unknown tool: '{tc.name}'. Available: {self.registry.tool_names}"
                    result_blocks.append(
                        ContentBlock(
                            type=ContentBlockType.TOOL_RESULT,
                            tool_call_id=tc.id,
                            text=error_msg,
                        )
                    )
                    continue

                # Execute
                self.guardrails.record_call(tc.name, tc.arguments)
                total_calls += 1

                self._log("tool_call", tool=tc.name, params=_summarize_params(tc.arguments))

                tool = self.registry.get(tc.name)
                result = await self.registry._execute_one(tool, tc.arguments, context)

                self.guardrails.record_result(tc.name, tc.arguments, result.success)

                event.result = result
                event.duration = time.time() - start_time
                if on_tool_call:
                    on_tool_call(event)

                self._log(
                    "tool_result",
                    tool=tc.name,
                    success=result.success,
                    output_lines=result.output.count("\n") + 1 if result.output else 0,
                    error=result.error or None,
                    duration=round(event.duration, 2),
                )

                # Build result text
                if result.success:
                    result_text = result.output or "(no output)"
                else:
                    result_text = f"ERROR: {result.error}\n{result.output}" if result.output else f"ERROR: {result.error}"

                # Truncate long outputs
                result_obj = ToolResult(success=result.success, output=result_text)
                result_text = result_obj.truncate(max_lines=200).output

                result_blocks.append(
                    ContentBlock(
                        type=ContentBlockType.TOOL_RESULT,
                        tool_call_id=tc.id,
                        text=result_text,
                    )
                )

            # Add tool results to conversation
            conversation.append(Message(role="tool_result", content=result_blocks))

    async def _stream_response(
        self,
        messages: list[Message],
        tool_schemas: list,
        on_text_delta: OnTextDelta,
    ) -> LLMResponse:
        """Call LLM with streaming, forwarding text deltas to the callback.

        Collects the full response for return while streaming text to the UI.
        Buffers and hides <think>...</think> blocks from the UI stream.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        usage = None
        # State for <think> tag filtering
        in_think = False
        think_buffer = ""

        async for chunk in self.llm.chat_stream(messages, tool_schemas):
            if chunk.type == "text_delta":
                text_parts.append(chunk.content)

                # Filter <think> blocks from the UI stream
                if in_think:
                    think_buffer += chunk.content
                    if "</think>" in think_buffer:
                        # End of think block — emit any text after the tag
                        after = think_buffer.split("</think>", 1)[1]
                        in_think = False
                        think_buffer = ""
                        if after.strip():
                            on_text_delta(after)
                elif "<think>" in chunk.content:
                    # Start of think block — emit text before the tag
                    before = chunk.content.split("<think>", 1)[0]
                    remainder = chunk.content.split("<think>", 1)[1]
                    if before.strip():
                        on_text_delta(before)
                    in_think = True
                    think_buffer = remainder
                    if "</think>" in think_buffer:
                        after = think_buffer.split("</think>", 1)[1]
                        in_think = False
                        think_buffer = ""
                        if after.strip():
                            on_text_delta(after)
                else:
                    on_text_delta(chunk.content)
            elif chunk.type == "tool_call_end" and chunk.tool_call:
                tool_calls.append(chunk.tool_call)
            elif chunk.type == "usage" and chunk.usage:
                usage = chunk.usage

        from pare.llm.base import TokenUsage

        # Strip <think> tags from the full content for the conversation record
        full_text = "".join(text_parts)
        clean_text = _THINK_RE.sub("", full_text).strip()

        return LLMResponse(
            content=clean_text,
            tool_calls=tool_calls,
            stop_reason=StopReason.TOOL_USE if tool_calls else StopReason.END_TURN,
            usage=usage or TokenUsage(input_tokens=0, output_tokens=0),
        )

    @staticmethod
    def _build_assistant_blocks(response: LLMResponse) -> list[ContentBlock]:
        """Convert LLMResponse to assistant message content blocks."""
        blocks: list[ContentBlock] = []

        if response.content:
            blocks.append(ContentBlock(type=ContentBlockType.TEXT, text=response.content))

        for tc in response.tool_calls:
            blocks.append(
                ContentBlock(type=ContentBlockType.TOOL_USE, tool_call=tc)
            )

        return blocks if blocks else [ContentBlock(type=ContentBlockType.TEXT, text="")]

    def _log(self, event_type: str, **data) -> None:
        """Log to telemetry if available."""
        if self.event_log:
            self.event_log.log(event_type, **data)


def _summarize_params(params: dict) -> dict:
    """Create a concise summary of tool params for logging.

    Truncates long string values to avoid bloating the event log.
    """
    summary = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 200:
            summary[k] = v[:200] + "..."
        else:
            summary[k] = v
    return summary
