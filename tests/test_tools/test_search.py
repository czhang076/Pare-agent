"""Tests for SearchTool."""

from pathlib import Path

import pytest

from pare.tools.base import ToolContext
from pare.tools.search import SearchTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


@pytest.fixture
def search_project(tmp_path: Path) -> Path:
    """Create a small project to search in."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('bye')\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def helper():\n    return 42\n"
    )
    (tmp_path / "README.md").write_text("# Hello Project\nThis is a test.\n")
    return tmp_path


class TestSearchTool:
    @pytest.mark.asyncio
    async def test_simple_search(self, ctx: ToolContext, search_project: Path):
        tool = SearchTool()
        result = await tool.execute({"pattern": "hello"}, ctx)
        assert result.success is True
        assert "hello" in result.output.lower()
        assert result.metadata["match_count"] > 0

    @pytest.mark.asyncio
    async def test_regex_search(self, ctx: ToolContext, search_project: Path):
        tool = SearchTool()
        result = await tool.execute({"pattern": "def \\w+"}, ctx)
        assert result.success is True
        assert "def hello" in result.output or "def helper" in result.output

    @pytest.mark.asyncio
    async def test_file_glob_filter(self, ctx: ToolContext, search_project: Path):
        tool = SearchTool()
        result = await tool.execute(
            {"pattern": "hello", "file_glob": "*.py"}, ctx
        )
        assert result.success is True
        # Should find hello in .py files but not in README.md
        assert "main.py" in result.output

    @pytest.mark.asyncio
    async def test_no_matches(self, ctx: ToolContext, search_project: Path):
        tool = SearchTool()
        result = await tool.execute({"pattern": "zzz_nonexistent_zzz"}, ctx)
        assert result.success is True
        assert "no matches" in result.output.lower()

    @pytest.mark.asyncio
    async def test_search_specific_path(self, ctx: ToolContext, search_project: Path):
        tool = SearchTool()
        result = await tool.execute(
            {"pattern": "helper", "path": "src"}, ctx
        )
        assert result.success is True
        assert "helper" in result.output

    @pytest.mark.asyncio
    async def test_empty_pattern(self, ctx: ToolContext):
        tool = SearchTool()
        result = await tool.execute({"pattern": ""}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, ctx: ToolContext):
        tool = SearchTool()
        result = await tool.execute(
            {"pattern": "test", "path": "../../"}, ctx
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_max_results_limit(self, ctx: ToolContext, tmp_path: Path):
        # Create a file with many matches
        content = "\n".join(f"match line {i}" for i in range(100))
        (tmp_path / "many.txt").write_text(content)
        tool = SearchTool()
        result = await tool.execute(
            {"pattern": "match", "max_results": 5}, ctx
        )
        assert result.success is True
        assert result.metadata["match_count"] <= 5
