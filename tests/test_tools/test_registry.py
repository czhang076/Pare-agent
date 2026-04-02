"""Tests for ToolRegistry, ToolResult, and permission logic."""

import pytest

from forge.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    create_default_registry,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeReadTool(Tool):
    name = "fake_read"
    description = "A fake read tool"
    parameters = {"type": "object", "properties": {}}
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output="read result")


class FakeWriteTool(Tool):
    name = "fake_write"
    description = "A fake write tool"
    parameters = {"type": "object", "properties": {}}
    mutation_type = MutationType.WRITE
    permission_level = PermissionLevel.CONFIRM_ONCE

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output="write result")


class FakeErrorTool(Tool):
    name = "fake_error"
    description = "Always raises"
    parameters = {"type": "object", "properties": {}}
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        raise RuntimeError("boom")


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


# ---------------------------------------------------------------------------
# ToolResult tests
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_truncate_short(self):
        result = ToolResult(success=True, output="line1\nline2\nline3")
        truncated = result.truncate(max_lines=10)
        assert truncated.output == "line1\nline2\nline3"

    def test_truncate_long(self):
        lines = [f"line {i}" for i in range(300)]
        result = ToolResult(success=True, output="\n".join(lines))
        truncated = result.truncate(max_lines=200)
        assert "[truncated — 100 more lines]" in truncated.output
        assert truncated.output.count("\n") < 210


# ---------------------------------------------------------------------------
# Permission tests
# ---------------------------------------------------------------------------


class TestPermissions:
    def test_auto_never_needs_confirmation(self, ctx: ToolContext):
        tool = FakeReadTool()
        # Even in non-headless mode
        ctx.headless = False
        assert tool.needs_confirmation(ctx) is False

    def test_confirm_once_first_time(self, ctx: ToolContext):
        tool = FakeWriteTool()
        ctx.headless = False
        assert tool.needs_confirmation(ctx) is True

    def test_confirm_once_after_marking(self, ctx: ToolContext):
        tool = FakeWriteTool()
        ctx.headless = False
        tool.mark_confirmed(ctx)
        assert tool.needs_confirmation(ctx) is False

    def test_always_confirm(self, ctx: ToolContext):
        from forge.tools.bash import BashTool
        tool = BashTool()
        ctx.headless = False
        tool.mark_confirmed(ctx)
        # ALWAYS_CONFIRM still asks even after marking
        assert tool.needs_confirmation(ctx) is True

    def test_headless_skips_all_confirmation(self, ctx: ToolContext):
        from forge.tools.bash import BashTool
        tool = BashTool()
        ctx.headless = True
        assert tool.needs_confirmation(ctx) is False


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = FakeReadTool()
        registry.register(tool)
        assert registry.get("fake_read") is tool

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        registry.register(FakeReadTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(FakeReadTool())

    def test_get_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool"):
            registry.get("nonexistent")

    def test_get_all_schemas(self):
        registry = ToolRegistry()
        registry.register(FakeReadTool())
        registry.register(FakeWriteTool())
        schemas = registry.get_all_schemas()
        assert len(schemas) == 2
        names = {s.name for s in schemas}
        assert names == {"fake_read", "fake_write"}

    def test_get_schemas_by_mutation(self):
        registry = ToolRegistry()
        registry.register(FakeReadTool())
        registry.register(FakeWriteTool())
        reads = registry.get_schemas_by_mutation(MutationType.READ)
        assert len(reads) == 1
        assert reads[0].name == "fake_read"

    def test_contains_and_len(self):
        registry = ToolRegistry()
        registry.register(FakeReadTool())
        assert "fake_read" in registry
        assert "nope" not in registry
        assert len(registry) == 1

    def test_tool_names(self):
        registry = ToolRegistry()
        registry.register(FakeReadTool())
        registry.register(FakeWriteTool())
        assert set(registry.tool_names) == {"fake_read", "fake_write"}

    @pytest.mark.asyncio
    async def test_execute_batch(self, ctx: ToolContext):
        registry = ToolRegistry()
        registry.register(FakeReadTool())
        registry.register(FakeWriteTool())

        calls = [
            {"name": "fake_read", "arguments": {}},
            {"name": "fake_write", "arguments": {}},
        ]
        results = await registry.execute(calls, ctx)
        assert len(results) == 2
        assert results[0].output == "read result"
        assert results[1].output == "write result"

    @pytest.mark.asyncio
    async def test_execute_handles_errors(self, ctx: ToolContext):
        registry = ToolRegistry()
        registry.register(FakeErrorTool())

        calls = [{"name": "fake_error", "arguments": {}}]
        results = await registry.execute(calls, ctx)
        assert len(results) == 1
        assert results[0].success is False
        assert "RuntimeError" in results[0].error


class TestCreateDefaultRegistry:
    def test_has_all_p0_tools(self):
        registry = create_default_registry()
        assert "bash" in registry
        assert "file_read" in registry
        assert "file_edit" in registry
        assert "file_create" in registry
        assert "search" in registry
        assert len(registry) == 5
