"""Tests for BashTool."""

import sys
from pathlib import Path

import pytest

from pare.tools.base import ToolContext
from pare.tools.bash import BashTool


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
        cmd = f'"{sys.executable}" -c "import time; time.sleep(60)"'
        result = await tool.execute(
            {"command": cmd, "timeout": 1},
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
        cmd = f'"{sys.executable}" -c "import os; print(os.getcwd())"'
        result = await tool.execute({"command": cmd}, ctx)
        assert result.success is True
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
        from pare.tools.base import PermissionLevel
        assert tool.permission_level == PermissionLevel.ALWAYS_CONFIRM
