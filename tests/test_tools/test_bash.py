"""Tests for BashTool."""

from pathlib import Path

import pytest

from forge.tools.base import ToolContext
from forge.tools.bash import BashTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


class TestBashTool:
    @pytest.mark.asyncio
    async def test_simple_command(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute({"command": "echo hello"}, ctx)
        assert result.success is True
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_return_code(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute({"command": "exit 1"}, ctx)
        assert result.success is False
        assert result.metadata["return_code"] == 1

    @pytest.mark.asyncio
    async def test_stderr_captured(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute({"command": "echo err >&2"}, ctx)
        assert "err" in result.output

    @pytest.mark.asyncio
    async def test_timeout(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute(
            {"command": "sleep 60", "timeout": 1},
            ctx,
        )
        assert result.success is False
        assert "timed out" in result.error.lower()
        assert result.metadata.get("timed_out") is True

    @pytest.mark.asyncio
    async def test_empty_command(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute({"command": ""}, ctx)
        assert result.success is False
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cwd_respected(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute({"command": "pwd"}, ctx)
        assert result.success is True
        # The output should contain the tmp_path
        # (normalize for Windows: compare basenames)
        assert ctx.cwd.name in result.output

    @pytest.mark.asyncio
    async def test_multiline_output(self, ctx: ToolContext):
        tool = BashTool()
        result = await tool.execute(
            {"command": "echo line1 && echo line2 && echo line3"},
            ctx,
        )
        assert result.success is True
        assert "line1" in result.output
        assert "line3" in result.output

    @pytest.mark.asyncio
    async def test_permission_level(self):
        tool = BashTool()
        from forge.tools.base import PermissionLevel
        assert tool.permission_level == PermissionLevel.ALWAYS_CONFIRM
