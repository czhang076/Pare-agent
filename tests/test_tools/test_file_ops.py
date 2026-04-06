"""Tests for FileReadTool, FileEditTool, and FileCreateTool.

These tests use real temp files to verify actual file I/O behavior.
"""

from pathlib import Path

import pytest

from forge.tools.base import ToolContext
from forge.tools.file_edit import FileCreateTool, FileEditTool
from forge.tools.file_read import FileReadTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample file with numbered content."""
    fp = tmp_path / "sample.py"
    lines = [f"line {i}: content {i}" for i in range(1, 21)]
    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------


class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_read_whole_file(self, ctx: ToolContext, sample_file: Path):
        tool = FileReadTool()
        result = await tool.execute({"file_path": "sample.py"}, ctx)
        assert result.success is True
        assert "line 1: content 1" in result.output
        assert "line 20: content 20" in result.output
        assert result.metadata["total_lines"] == 20

    @pytest.mark.asyncio
    async def test_read_line_range(self, ctx: ToolContext, sample_file: Path):
        tool = FileReadTool()
        result = await tool.execute(
            {"file_path": "sample.py", "start_line": 5, "end_line": 10}, ctx
        )
        assert result.success is True
        assert "line 5: content 5" in result.output
        assert "line 10: content 10" in result.output
        assert "line 4" not in result.output
        assert "line 11" not in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, ctx: ToolContext):
        tool = FileReadTool()
        result = await tool.execute({"file_path": "nope.py"}, ctx)
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_directory(self, ctx: ToolContext):
        tool = FileReadTool()
        (ctx.cwd / "adir").mkdir()
        result = await tool.execute({"file_path": "adir"}, ctx)
        assert result.success is False
        assert "directory" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, ctx: ToolContext):
        tool = FileReadTool()
        result = await tool.execute({"file_path": "../../etc/passwd"}, ctx)
        assert result.success is False
        assert "outside" in result.error.lower() or "denied" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_path(self, ctx: ToolContext):
        tool = FileReadTool()
        result = await tool.execute({"file_path": ""}, ctx)
        assert result.success is False


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------


class TestFileEditTool:
    @pytest.mark.asyncio
    async def test_simple_replace(self, ctx: ToolContext, sample_file: Path):
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "sample.py",
                "old_str": "line 5: content 5",
                "new_str": "line 5: MODIFIED",
            },
            ctx,
        )
        assert result.success is True
        assert "---" in result.output  # diff output
        # Verify actual file change
        content = sample_file.read_text()
        assert "line 5: MODIFIED" in content
        assert "line 5: content 5" not in content

    @pytest.mark.asyncio
    async def test_old_str_not_found(self, ctx: ToolContext, sample_file: Path):
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "sample.py",
                "old_str": "this does not exist in the file",
                "new_str": "replacement",
            },
            ctx,
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_ambiguous_match(self, ctx: ToolContext, tmp_path: Path):
        fp = tmp_path / "dup.py"
        fp.write_text("x = 1\nx = 1\n", encoding="utf-8")
        tool = FileEditTool()
        result = await tool.execute(
            {"file_path": "dup.py", "old_str": "x = 1", "new_str": "x = 2"},
            ctx,
        )
        assert result.success is False
        assert "2 times" in result.error

    @pytest.mark.asyncio
    async def test_identical_old_new(self, ctx: ToolContext, sample_file: Path):
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "sample.py",
                "old_str": "line 1: content 1",
                "new_str": "line 1: content 1",
            },
            ctx,
        )
        assert result.success is False
        assert "identical" in result.error.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, ctx: ToolContext):
        tool = FileEditTool()
        result = await tool.execute(
            {"file_path": "nope.py", "old_str": "a", "new_str": "b"},
            ctx,
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, ctx: ToolContext):
        tool = FileEditTool()
        result = await tool.execute(
            {"file_path": "../../etc/passwd", "old_str": "a", "new_str": "b"},
            ctx,
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_multiline_replace(self, ctx: ToolContext, tmp_path: Path):
        fp = tmp_path / "multi.py"
        fp.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "multi.py",
                "old_str": "def foo():\n    return 1",
                "new_str": "def foo():\n    return 42",
            },
            ctx,
        )
        assert result.success is True
        content = fp.read_text()
        assert "return 42" in content
        assert "return 2" in content  # bar() unchanged

    @pytest.mark.asyncio
    async def test_whitespace_fallback_succeeds(self, ctx: ToolContext, tmp_path: Path):
        """Whitespace-insensitive fallback matches when only whitespace differs."""
        fp = tmp_path / "ws.py"
        fp.write_text("x  =   1\ny = 2\n", encoding="utf-8")
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "ws.py",
                "old_str": "x = 1",
                "new_str": "x = 42",
            },
            ctx,
        )
        assert result.success is True
        content = fp.read_text()
        assert "42" in content

    @pytest.mark.asyncio
    async def test_whitespace_fallback_trailing_spaces(self, ctx: ToolContext, tmp_path: Path):
        """Trailing whitespace differences are handled by fallback."""
        fp = tmp_path / "trail.py"
        fp.write_text("def foo():   \n    pass\n", encoding="utf-8")
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "trail.py",
                "old_str": "def foo():\n    pass",
                "new_str": "def foo():\n    return 1",
            },
            ctx,
        )
        assert result.success is True
        content = fp.read_text()
        assert "return 1" in content

    @pytest.mark.asyncio
    async def test_whitespace_fallback_no_match(self, ctx: ToolContext, sample_file: Path):
        """Whitespace fallback doesn't match structurally different content."""
        tool = FileEditTool()
        result = await tool.execute(
            {
                "file_path": "sample.py",
                "old_str": "totally different content",
                "new_str": "replacement",
            },
            ctx,
        )
        assert result.success is False
        assert "not found" in result.error.lower()


# ---------------------------------------------------------------------------
# FileCreateTool
# ---------------------------------------------------------------------------


class TestFileCreateTool:
    @pytest.mark.asyncio
    async def test_create_new_file(self, ctx: ToolContext):
        tool = FileCreateTool()
        result = await tool.execute(
            {"file_path": "new_file.py", "content": "print('hello')\n"},
            ctx,
        )
        assert result.success is True
        fp = ctx.cwd / "new_file.py"
        assert fp.exists()
        assert fp.read_text() == "print('hello')\n"

    @pytest.mark.asyncio
    async def test_create_with_subdirectories(self, ctx: ToolContext):
        tool = FileCreateTool()
        result = await tool.execute(
            {"file_path": "src/utils/helper.py", "content": "# helper\n"},
            ctx,
        )
        assert result.success is True
        assert (ctx.cwd / "src" / "utils" / "helper.py").exists()

    @pytest.mark.asyncio
    async def test_create_existing_file_fails(self, ctx: ToolContext, sample_file: Path):
        tool = FileCreateTool()
        result = await tool.execute(
            {"file_path": "sample.py", "content": "overwrite"},
            ctx,
        )
        assert result.success is False
        assert "already exists" in result.error.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, ctx: ToolContext):
        tool = FileCreateTool()
        result = await tool.execute(
            {"file_path": "../../evil.py", "content": "bad"},
            ctx,
        )
        assert result.success is False
