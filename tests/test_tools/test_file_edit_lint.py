"""Phase 3.13.1 post-edit lint tests for :class:`FileEditTool`.

The host branch parses new content with :func:`ast.parse` after write.
The container branch shells out to ``python -m py_compile`` inside the
container. Both paths must:

- Leave the file on disk (the agent needs the edit history to self-correct).
- Return ``success=False`` with the literal marker ``⚠ SYNTAX ERROR:`` so
  :func:`pare.trajectory.error_signal_extractor.extract_error_signal`
  classifies the event as ``SYNTAX_ERROR`` (Liu B2.2).
- Skip non-Python files (no false positives on JSON / Markdown / etc.).
- Let valid Python through unchanged (``success=True``).

We assert ``ErrorSignal.SYNTAX_ERROR`` routing too, because the whole
point of Phase 3.13.1 is "Liu B2.2 signals become observable" — if the
regex misses our marker, the entire feature is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from pare.sandbox.instance_container import ExecResult
from pare.tools.base import ToolContext
from pare.tools.file_edit import FileEditTool
from pare.trajectory.error_signal_extractor import extract_error_signal
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _host_ctx(cwd: Path) -> ToolContext:
    return ToolContext(cwd=cwd, env={}, confirmed_tools=set(), headless=True)


def _make_event_from_result(
    tool_name: str, params: dict, result
) -> ToolCallEvent:
    """Wrap a ToolResult into the event shape the extractor expects."""
    # extract_error_signal scans result_content; when success=False the
    # loop's ``_format_result_text`` produces ``"ERROR: {error}\n{output}"``.
    if result.success:
        content = result.output or "(no output)"
    elif result.output:
        content = f"ERROR: {result.error}\n{result.output}"
    else:
        content = f"ERROR: {result.error}"
    return ToolCallEvent.create(
        turn_id=0,
        call_index_in_turn=0,
        global_index=0,
        tool_name=tool_name,
        params=params,
        result_success=result.success,
        result_content=content,
        timestamp=0.0,
    )


# ---------------------------------------------------------------------------
# Host branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_syntax_error_flags_failure_but_keeps_write(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")

    params = {"file_path": "a.py", "old_str": "x = 1", "new_str": "def broken(:"}
    r = await FileEditTool().execute(params, _host_ctx(tmp_path))

    # Lint catches the error → success=False.
    assert r.success is False
    assert "⚠ SYNTAX ERROR:" in r.error
    # File stays written — that's deliberate, so the agent can see and fix.
    # Original had a trailing newline that the replacement preserved.
    assert path.read_text() == "def broken(:\n"

    # ErrorSignal routing confirms Liu B2.2 becomes observable.
    event = _make_event_from_result("file_edit", params, r)
    assert extract_error_signal(event) == ErrorSignal.SYNTAX_ERROR


@pytest.mark.asyncio
async def test_host_valid_python_still_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n")

    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "x = 1", "new_str": "x = 2"},
        _host_ctx(tmp_path),
    )
    assert r.success is True
    assert "syntax_error" not in r.metadata
    assert path.read_text() == "x = 2\n"


@pytest.mark.asyncio
async def test_host_non_py_file_skips_lint(tmp_path: Path) -> None:
    """A broken-looking edit on a .txt file must NOT fail the tool."""
    path = tmp_path / "notes.md"
    path.write_text("old\n")

    r = await FileEditTool().execute(
        {"file_path": "notes.md", "old_str": "old", "new_str": "def broken(:"},
        _host_ctx(tmp_path),
    )
    assert r.success is True
    assert path.read_text() == "def broken(:\n"


@pytest.mark.asyncio
async def test_host_indentation_error_classified_as_syntax(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("def f():\n    return 1\n")

    # Break the indentation.
    r = await FileEditTool().execute(
        {
            "file_path": "a.py",
            "old_str": "def f():\n    return 1\n",
            "new_str": "def f():\nreturn 1\n",
        },
        _host_ctx(tmp_path),
    )
    assert r.success is False
    assert "⚠ SYNTAX ERROR:" in r.error
    event = _make_event_from_result("file_edit", {}, r)
    assert extract_error_signal(event) == ErrorSignal.SYNTAX_ERROR


# ---------------------------------------------------------------------------
# Container branch — fake container routing py_compile
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LintFakeContainer:
    """Container fake that routes ``python -m py_compile`` to a scripted
    result, while keeping file I/O in memory.
    """

    workdir: str = "/testbed"
    instance_id: str = "fake"
    files: dict[str, str] = field(default_factory=dict)
    py_compile_result: ExecResult = field(
        default_factory=lambda: ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False,
        )
    )
    exec_log: list[Any] = field(default_factory=list)

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        if path not in self.files:
            raise RuntimeError(f"not found: {path}")
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    async def exec(self, cmd, *, timeout=60.0, cwd=None, env=None):
        self.exec_log.append(cmd)
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "py_compile" in cmd_str:
            return self.py_compile_result
        return ExecResult(stdout="", stderr="", exit_code=0, timed_out=False)


def _container_ctx(container: LintFakeContainer) -> ToolContext:
    return ToolContext(
        cwd=PurePosixPath(container.workdir),  # type: ignore[arg-type]
        env={},
        confirmed_tools=set(),
        headless=True,
        container=container,  # type: ignore[arg-type]
        exec_target="container",
    )


@pytest.mark.asyncio
async def test_container_syntax_error_surfaces() -> None:
    fake = LintFakeContainer(
        files={"/testbed/a.py": "x = 1\n"},
        py_compile_result=ExecResult(
            stdout="",
            stderr=(
                '  File "a.py", line 1\n'
                "    def broken(:\n"
                "               ^\n"
                "SyntaxError: invalid syntax\n"
            ),
            exit_code=1,
            timed_out=False,
        ),
    )
    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "x = 1", "new_str": "def broken(:"},
        _container_ctx(fake),
    )
    assert r.success is False
    assert "⚠ SYNTAX ERROR:" in r.error
    assert fake.files["/testbed/a.py"] == "def broken(:\n"
    # py_compile was actually called.
    assert any("py_compile" in (str(c)) for c in fake.exec_log)

    event = _make_event_from_result("file_edit", {}, r)
    assert extract_error_signal(event) == ErrorSignal.SYNTAX_ERROR


@pytest.mark.asyncio
async def test_container_valid_edit_passes_lint() -> None:
    fake = LintFakeContainer(
        files={"/testbed/a.py": "x = 1\n"},
        py_compile_result=ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False,
        ),
    )
    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "x = 1", "new_str": "x = 2"},
        _container_ctx(fake),
    )
    assert r.success is True
    assert fake.files["/testbed/a.py"] == "x = 2\n"


@pytest.mark.asyncio
async def test_container_lint_timeout_treated_as_soft_pass() -> None:
    """A lint timeout must not false-positive the edit.

    Per the contract in ``_lint_python_container``, timeouts log a
    warning and return ``None`` (no syntax error) rather than block the
    edit — timeouts indicate container trouble, not a syntax issue.
    """
    fake = LintFakeContainer(
        files={"/testbed/a.py": "x = 1\n"},
        py_compile_result=ExecResult(
            stdout="", stderr="", exit_code=124, timed_out=True,
        ),
    )
    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "x = 1", "new_str": "x = 2"},
        _container_ctx(fake),
    )
    assert r.success is True


@pytest.mark.asyncio
async def test_container_non_py_file_skips_lint() -> None:
    fake = LintFakeContainer(files={"/testbed/notes.md": "old\n"})
    r = await FileEditTool().execute(
        {"file_path": "notes.md", "old_str": "old", "new_str": "new"},
        _container_ctx(fake),
    )
    assert r.success is True
    # No py_compile invocation at all.
    assert not any("py_compile" in str(c) for c in fake.exec_log)
