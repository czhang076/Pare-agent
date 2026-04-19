"""Tests for DeclareDoneTool and the executor's early-exit behaviour.

Two layers:
1. The tool itself — arg validation + metadata shape.
2. Executor integration — calling declare_done in a tool turn ends the
   ReAct loop cleanly and surfaces the status on ExecutionResult.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from pare.agent.executor import ReActExecutor
from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    LLMResponse,
    StopReason,
    TokenUsage,
    ToolCallRequest,
)
from pare.tools.base import ToolContext, create_default_registry
from pare.tools.declare_done import DeclareDoneTool


# ---------------------------------------------------------------------------
# Tool unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


@pytest.mark.asyncio
async def test_declare_done_fixed(ctx: ToolContext) -> None:
    tool = DeclareDoneTool()
    result = await tool.execute(
        {"status": "fixed", "summary": "Fixed the off-by-one in point.py"}, ctx
    )
    assert result.success is True
    assert result.metadata["status"] == "fixed"
    assert result.metadata["summary"].startswith("Fixed")


@pytest.mark.asyncio
async def test_declare_done_cannot_fix(ctx: ToolContext) -> None:
    result = await DeclareDoneTool().execute(
        {"status": "cannot_fix", "summary": "Bug is in an installed wheel, not in-repo."}, ctx
    )
    assert result.success is True
    assert result.metadata["status"] == "cannot_fix"


@pytest.mark.asyncio
async def test_declare_done_rejects_invalid_status(ctx: ToolContext) -> None:
    result = await DeclareDoneTool().execute(
        {"status": "kind-of-done", "summary": "meh"}, ctx
    )
    assert result.success is False
    assert "kind-of-done" in result.error


@pytest.mark.asyncio
async def test_declare_done_requires_summary(ctx: ToolContext) -> None:
    result = await DeclareDoneTool().execute({"status": "fixed", "summary": "   "}, ctx)
    assert result.success is False
    assert "summary" in result.error.lower()


def test_declare_done_registered_in_default_registry() -> None:
    registry = create_default_registry()
    assert "declare_done" in registry


# ---------------------------------------------------------------------------
# Executor integration — declare_done terminates the ReAct loop
# ---------------------------------------------------------------------------


def _fake_llm_response(
    *,
    text: str = "",
    tool_calls: list[ToolCallRequest] | None = None,
    stop_reason: StopReason = StopReason.TOOL_USE,
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=0, output_tokens=0),
    )


@pytest.mark.asyncio
async def test_executor_exits_cleanly_on_declare_done_fixed(tmp_path: Path) -> None:
    """declare_done('fixed') ends the loop; result surfaces declared_status + summary."""
    registry = create_default_registry()

    declare_call = ToolCallRequest(
        id="tc-1",
        name="declare_done",
        arguments={"status": "fixed", "summary": "Applied the patch."},
    )

    llm = AsyncMock()
    llm.chat.side_effect = [
        _fake_llm_response(text="Done.", tool_calls=[declare_call]),
    ]

    executor = ReActExecutor(llm=llm, registry=registry)
    ctx = ToolContext(cwd=tmp_path, headless=True)
    result = await executor.run(system_prompt="sys", user_message="task", context=ctx)

    assert result.stop_reason == "declared_done"
    assert result.declared_status == "fixed"
    assert result.declared_summary == "Applied the patch."
    assert result.success is True
    assert result.llm_claimed_success is True
    # The LLM should have been called exactly once — we exit after the
    # first turn that contains declare_done; no further rounds.
    assert llm.chat.await_count == 1


@pytest.mark.asyncio
async def test_executor_marks_cannot_fix_as_unsuccessful(tmp_path: Path) -> None:
    registry = create_default_registry()

    declare_call = ToolCallRequest(
        id="tc-1",
        name="declare_done",
        arguments={"status": "cannot_fix", "summary": "The bug is in a binary wheel."},
    )

    llm = AsyncMock()
    llm.chat.side_effect = [
        _fake_llm_response(tool_calls=[declare_call]),
    ]

    executor = ReActExecutor(llm=llm, registry=registry)
    ctx = ToolContext(cwd=tmp_path, headless=True)
    result = await executor.run(system_prompt="sys", user_message="task", context=ctx)

    assert result.declared_status == "cannot_fix"
    # "cannot_fix" must NOT be reported as success downstream, otherwise
    # Module B will double-count give-ups as wins.
    assert result.success is False
    assert result.llm_claimed_success is False


@pytest.mark.asyncio
async def test_silent_stop_leaves_declared_status_empty(tmp_path: Path) -> None:
    """Back-compat: LLM that ends without calling declare_done still works."""
    registry = create_default_registry()

    llm = AsyncMock()
    llm.chat.side_effect = [
        _fake_llm_response(text="I'm done.", tool_calls=[], stop_reason=StopReason.END_TURN),
    ]

    executor = ReActExecutor(llm=llm, registry=registry)
    ctx = ToolContext(cwd=tmp_path, headless=True)
    result = await executor.run(system_prompt="sys", user_message="task", context=ctx)

    assert result.stop_reason == "end_turn"
    # No declaration → empty string, downstream can filter on it.
    assert result.declared_status == ""
    assert result.declared_summary == ""
