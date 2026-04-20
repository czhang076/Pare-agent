"""Tests for :class:`DeclareDoneTool` argument validation and metadata shape.

R5 note: the former "executor integration — declare_done terminates the
ReAct loop" block was deleted along with ``pare.agent.executor``. The
equivalent exit-path assertions live in ``tests/test_agent/test_loop.py``
(``test_declared_done_exits`` and siblings), which drive the flat ReAct
loop end-to-end with a scripted LLM fake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pare.tools.base import ToolContext, create_default_registry
from pare.tools.declare_done import DeclareDoneTool


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
        {"status": "cannot_fix", "summary": "Bug is in an installed wheel, not in-repo."},
        ctx,
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
