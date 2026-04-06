"""Tests for forge/agent/executor.py.

Uses a mock LLM adapter that returns predetermined responses, allowing
us to test the ReAct loop logic without real API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import pytest

from forge.agent.executor import ReActExecutor, ExecutionResult
from forge.agent.guardrails import GuardrailConfig, Guardrails
from forge.llm.base import (
    LLMAdapter,
    LLMResponse,
    Message,
    ModelProfile,
    StopReason,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
    ToolSchema,
)
from forge.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Mock LLM adapter
# ---------------------------------------------------------------------------


class MockLLM(LLMAdapter):
    """LLM that returns a predefined sequence of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="mock", profile=ModelProfile())
        self._responses = list(responses)
        self._call_count = 0

    async def chat(
        self, messages, tools=None, *, temperature=None, max_tokens=None
    ) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return LLMResponse(
                content="I'm done.",
                tool_calls=[],
                stop_reason=StopReason.END_TURN,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
        response = self._responses[self._call_count]
        self._call_count += 1
        return response

    async def chat_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    def count_tokens(self, messages) -> int:
        return 100


# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    """Tool that echoes its input."""

    name = "echo"
    description = "Echo input"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output=f"echo: {params.get('text', '')}")


class FailTool(Tool):
    """Tool that always fails."""

    name = "fail"
    description = "Always fails"
    parameters = {"type": "object", "properties": {}}
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        return ToolResult(success=False, output="", error="intentional failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_USAGE = TokenUsage(input_tokens=100, output_tokens=50)


def _make_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return registry


def _tool_call_response(name: str, args: dict, text: str = "") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[ToolCallRequest(id=f"tc_{name}", name=name, arguments=args)],
        stop_reason=StopReason.TOOL_USE,
        usage=_USAGE,
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
        usage=_USAGE,
    )


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReActExecutor:
    @pytest.mark.asyncio
    async def test_single_tool_call_then_done(self, ctx):
        """LLM calls echo once, then produces final text."""
        llm = MockLLM([
            _tool_call_response("echo", {"text": "hello"}),
            _text_response("All done."),
        ])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run("system", "task", ctx)

        assert result.success is True
        assert result.output == "All done."
        assert result.tool_call_count == 1
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_no_tool_calls(self, ctx):
        """LLM responds immediately without tool calls."""
        llm = MockLLM([_text_response("Nothing to do.")])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run("system", "task", ctx)

        assert result.success is True
        assert result.output == "Nothing to do."
        assert result.tool_call_count == 0

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, ctx):
        """LLM calls tools multiple times before stopping."""
        llm = MockLLM([
            _tool_call_response("echo", {"text": "first"}),
            _tool_call_response("echo", {"text": "second"}),
            _text_response("Done after two calls."),
        ])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run("system", "task", ctx)

        assert result.success is True
        assert result.tool_call_count == 2

    @pytest.mark.asyncio
    async def test_budget_exhaustion(self, ctx):
        """Stops when total budget is exhausted."""
        config = GuardrailConfig(max_tool_calls=2)
        guard = Guardrails(config)

        llm = MockLLM([
            _tool_call_response("echo", {"text": "1"}),
            _tool_call_response("echo", {"text": "2"}),
            _tool_call_response("echo", {"text": "3"}),  # Should not execute
        ])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry, guard)

        result = await executor.run("system", "task", ctx)

        assert result.success is False
        assert result.stop_reason == "budget_exhausted"
        assert result.tool_call_count == 2

    @pytest.mark.asyncio
    async def test_unknown_tool_handled(self, ctx):
        """Unknown tool name produces an error result, not a crash."""
        llm = MockLLM([
            _tool_call_response("nonexistent_tool", {}),
            _text_response("OK, that tool doesn't exist."),
        ])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run("system", "task", ctx)

        assert result.success is True
        assert result.tool_call_count == 0  # Unknown tool not counted

    @pytest.mark.asyncio
    async def test_read_before_write_enforced(self, ctx):
        """Guardrail blocks file_edit without prior file_read."""
        from forge.tools.file_edit import FileEditTool
        from forge.tools.file_read import FileReadTool

        # Create a file to edit
        (ctx.cwd / "main.py").write_text("x = 1\n")

        llm = MockLLM([
            # LLM tries to edit without reading first
            _tool_call_response(
                "file_edit",
                {"file_path": "main.py", "old_str": "x = 1", "new_str": "x = 2"},
            ),
            # After being blocked, reads first
            _tool_call_response("file_read", {"file_path": "main.py"}),
            # Then edits
            _tool_call_response(
                "file_edit",
                {"file_path": "main.py", "old_str": "x = 1", "new_str": "x = 2"},
            ),
            _text_response("Fixed."),
        ])
        registry = _make_registry(FileReadTool(), FileEditTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run("system", "task", ctx)

        assert result.success is True
        # First edit blocked, then read + edit = 2 actual calls
        assert result.tool_call_count == 2
        assert (ctx.cwd / "main.py").read_text() == "x = 2\n"

    @pytest.mark.asyncio
    async def test_on_tool_call_callback(self, ctx):
        """The on_tool_call callback is invoked for each tool call."""
        events = []

        llm = MockLLM([
            _tool_call_response("echo", {"text": "hi"}),
            _text_response("Done."),
        ])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run(
            "system", "task", ctx,
            on_tool_call=lambda e: events.append(e),
        )

        assert len(events) == 1
        assert events[0].tool_name == "echo"
        assert events[0].result is not None
        assert events[0].result.success is True

    @pytest.mark.asyncio
    async def test_conversation_history_preserved(self, ctx):
        """Messages accumulate correctly across turns."""
        llm = MockLLM([
            _tool_call_response("echo", {"text": "test"}),
            _text_response("Done."),
        ])
        registry = _make_registry(EchoTool())
        executor = ReActExecutor(llm, registry)

        result = await executor.run("You are a bot.", "Do something.", ctx)

        # system + user + assistant(tool_call) + tool_result + assistant(text)
        assert len(result.messages) == 5
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"
        assert result.messages[3].role == "tool_result"
        assert result.messages[4].role == "assistant"
