"""Tests for the workspace resource validator.

The validator is a single choke point shared by file_read / file_edit /
file_create; these tests verify both the pure string rules and their
propagation into each of the three tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pare.tools.base import ToolContext, validate_workspace_path
from pare.tools.file_edit import FileCreateTool, FileEditTool
from pare.tools.file_read import FileReadTool


# ---------------------------------------------------------------------------
# Pure validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".pare/MEMORY.md",
        ".pare/history.jsonl",
        "./.pare/MEMORY.md",
        ".PARE/MEMORY.md",
        "__pycache__/abc.cpython-312.pyc",
        "sympy/__pycache__/abc.cpython-312.pyc",
        ".git/config",
        "sub/.git/HEAD",
        "foo/bar.pyc",
        "foo/bar.PYO",
        "m.pyd",
    ],
)
def test_validator_rejects_forbidden(path: str) -> None:
    assert validate_workspace_path(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        "sympy/geometry/point.py",
        "README.md",
        "src/module.py",
        "pare/agent/planner.py",
        # Not under .git/.pare/__pycache__ — just named similarly
        "docs/pare_usage.md",
        "scripts/git_helper.py",
        "",  # empty path handled upstream, validator must not crash
    ],
)
def test_validator_allows_normal_paths(path: str) -> None:
    assert validate_workspace_path(path) is None


# ---------------------------------------------------------------------------
# Integration: each file tool must short-circuit on forbidden paths
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, headless=True)


@pytest.mark.asyncio
async def test_file_read_rejects_pycache(ctx: ToolContext, tmp_path: Path) -> None:
    # Create the file on disk — the validator should reject *before* read.
    pkg = tmp_path / "__pycache__"
    pkg.mkdir()
    (pkg / "abc.cpython-312.pyc").write_bytes(b"\x00\x01")

    result = await FileReadTool().execute(
        {"file_path": "__pycache__/abc.cpython-312.pyc"}, ctx
    )
    assert result.success is False
    assert "bytecode" in result.error.lower() or "bookkeeping" in result.error.lower()


@pytest.mark.asyncio
async def test_file_edit_rejects_pare_internal(ctx: ToolContext, tmp_path: Path) -> None:
    (tmp_path / ".pare").mkdir()
    (tmp_path / ".pare" / "MEMORY.md").write_text("x", encoding="utf-8")

    result = await FileEditTool().execute(
        {"file_path": ".pare/MEMORY.md", "old_str": "x", "new_str": "y"}, ctx
    )
    assert result.success is False
    assert ".pare/MEMORY.md" in result.error
    # The file must not have been modified.
    assert (tmp_path / ".pare" / "MEMORY.md").read_text(encoding="utf-8") == "x"


@pytest.mark.asyncio
async def test_file_create_rejects_pyc(ctx: ToolContext, tmp_path: Path) -> None:
    result = await FileCreateTool().execute(
        {"file_path": "foo/bar.pyc", "content": "anything"}, ctx
    )
    assert result.success is False
    assert not (tmp_path / "foo" / "bar.pyc").exists()


@pytest.mark.asyncio
async def test_file_create_allows_real_source(ctx: ToolContext, tmp_path: Path) -> None:
    result = await FileCreateTool().execute(
        {"file_path": "sympy/new_mod.py", "content": "x = 1\n"}, ctx
    )
    assert result.success is True
    assert (tmp_path / "sympy" / "new_mod.py").read_text() == "x = 1\n"
