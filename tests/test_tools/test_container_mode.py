"""Unit tests for the container branch of bash/file_read/file_edit/search.

Uses a hand-rolled FakeContainer (not MagicMock) so the tests double as
executable documentation of the InstanceContainer contract the tools rely
on. No Docker required — runs on any dev box.

Host-mode parity is checked by the existing suites
(test_bash/test_file_read/test_file_edit/test_search). These tests focus
on the ``exec_target == "container"`` branch only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

import pytest

from pare.sandbox.instance_container import ExecResult
from pare.tools.base import ToolContext
from pare.tools.bash import BashTool
from pare.tools.declare_done import DeclareDoneTool
from pare.tools.file_edit import FileCreateTool, FileEditTool
from pare.tools.file_read import FileReadTool
from pare.tools.search import SearchTool


# ---------------------------------------------------------------------------
# Fake container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FakeContainer:
    """Minimal stand-in for :class:`InstanceContainer`.

    - Keeps an in-memory dict file system (absolute-path keyed).
    - Scripted ``exec_responses``: popleft per call if present, else
      returns the ``default_exec`` value.
    - Records every ``exec`` command in ``exec_calls`` so tests can assert
      the tool built the command it was supposed to.
    """

    instance_id: str = "fake"
    workdir: str = "/testbed"
    files: dict[str, str] = field(default_factory=dict)
    exec_responses: list[ExecResult] = field(default_factory=list)
    default_exec: ExecResult = field(
        default_factory=lambda: ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False
        )
    )
    exec_calls: list[tuple[object, float | None]] = field(default_factory=list)

    async def exec(self, cmd, *, timeout=60.0, cwd=None, env=None):
        self.exec_calls.append((cmd, timeout))
        if self.exec_responses:
            return self.exec_responses.pop(0)
        return self.default_exec

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        if path not in self.files:
            raise RuntimeError(f"file not found: {path}")
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


def _ctx(container: FakeContainer) -> ToolContext:
    return ToolContext(
        cwd=PurePosixPath(container.workdir),  # type: ignore[arg-type]
        env={},
        confirmed_tools=set(),
        headless=True,
        container=container,
        exec_target="container",
    )


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_container_success() -> None:
    fake = FakeContainer(
        default_exec=ExecResult(stdout="hello\n", stderr="", exit_code=0, timed_out=False),
    )
    r = await BashTool().execute({"command": "echo hello"}, _ctx(fake))
    assert r.success is True
    assert "hello" in r.output
    assert r.metadata == {"return_code": 0, "timed_out": False}
    # Command was forwarded verbatim (InstanceContainer wraps strings in bash -lc).
    assert fake.exec_calls[0][0] == "echo hello"


@pytest.mark.asyncio
async def test_bash_container_nonzero_exit_merges_stderr() -> None:
    fake = FakeContainer(
        default_exec=ExecResult(
            stdout="", stderr="boom\n", exit_code=2, timed_out=False
        ),
    )
    r = await BashTool().execute({"command": "false"}, _ctx(fake))
    assert r.success is False
    assert "STDERR:" in r.output
    assert "boom" in r.output
    assert r.error == "Exit code: 2"


@pytest.mark.asyncio
async def test_bash_container_timeout() -> None:
    fake = FakeContainer(
        default_exec=ExecResult(stdout="", stderr="", exit_code=124, timed_out=True),
    )
    r = await BashTool().execute(
        {"command": "sleep 60", "timeout": 1}, _ctx(fake)
    )
    assert r.success is False
    assert "timed out" in r.error
    assert r.metadata == {"return_code": 124, "timed_out": True}


@pytest.mark.asyncio
async def test_bash_container_without_container_errors() -> None:
    ctx = ToolContext(
        cwd=PurePosixPath("/testbed"),  # type: ignore[arg-type]
        headless=True,
        container=None,
        exec_target="container",
    )
    r = await BashTool().execute({"command": "echo x"}, ctx)
    assert r.success is False
    assert "container" in r.error.lower()


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_read_container_roundtrip() -> None:
    fake = FakeContainer(files={"/testbed/foo.py": "a\nb\nc\n"})
    r = await FileReadTool().execute({"file_path": "foo.py"}, _ctx(fake))
    assert r.success is True
    assert "1\ta" in r.output
    assert "3\tc" in r.output
    assert r.metadata["total_lines"] == 3


@pytest.mark.asyncio
async def test_file_read_container_rejects_escape() -> None:
    fake = FakeContainer(files={"/etc/passwd": "x"})
    r = await FileReadTool().execute({"file_path": "/etc/passwd"}, _ctx(fake))
    assert r.success is False
    assert "Access denied" in r.error


@pytest.mark.asyncio
async def test_file_read_container_missing_file() -> None:
    fake = FakeContainer()
    r = await FileReadTool().execute({"file_path": "does_not_exist.py"}, _ctx(fake))
    assert r.success is False
    assert "Failed to read" in r.error


# ---------------------------------------------------------------------------
# file_edit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_edit_container_single_match() -> None:
    fake = FakeContainer(files={"/testbed/a.py": "alpha\nbeta\ngamma\n"})
    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "beta", "new_str": "BETA"},
        _ctx(fake),
    )
    assert r.success is True
    assert fake.files["/testbed/a.py"] == "alpha\nBETA\ngamma\n"
    assert "BETA" in r.output and "-beta" in r.output


@pytest.mark.asyncio
async def test_file_edit_container_multi_match_fails() -> None:
    fake = FakeContainer(files={"/testbed/a.py": "x\nx\n"})
    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "x", "new_str": "y"},
        _ctx(fake),
    )
    assert r.success is False
    assert "matches 2 times" in r.error


@pytest.mark.asyncio
async def test_file_edit_container_no_match() -> None:
    fake = FakeContainer(files={"/testbed/a.py": "foo\n"})
    r = await FileEditTool().execute(
        {"file_path": "a.py", "old_str": "bar", "new_str": "baz"},
        _ctx(fake),
    )
    assert r.success is False
    assert "not found" in r.error


@pytest.mark.asyncio
async def test_file_create_container_refuses_overwrite() -> None:
    fake = FakeContainer(files={"/testbed/exists.py": "old"})
    # test -e returns 0 when file exists.
    fake.default_exec = ExecResult(stdout="", stderr="", exit_code=0, timed_out=False)
    r = await FileCreateTool().execute(
        {"file_path": "exists.py", "content": "new"},
        _ctx(fake),
    )
    assert r.success is False
    assert "already exists" in r.error


@pytest.mark.asyncio
async def test_file_create_container_writes_new_file() -> None:
    fake = FakeContainer()
    # test -e returns 1 when file missing.
    fake.default_exec = ExecResult(stdout="", stderr="", exit_code=1, timed_out=False)
    r = await FileCreateTool().execute(
        {"file_path": "new.py", "content": "hi\n"},
        _ctx(fake),
    )
    assert r.success is True
    assert fake.files["/testbed/new.py"] == "hi\n"
    assert r.metadata["lines"] == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_container_uses_rg_when_present() -> None:
    fake = FakeContainer(
        exec_responses=[
            # `command -v rg` probe
            ExecResult(stdout="/usr/bin/rg\n", stderr="", exit_code=0, timed_out=False),
            # rg call
            ExecResult(
                stdout="sympy/foo.py:10:def solve():\n",
                stderr="", exit_code=0, timed_out=False,
            ),
        ],
    )
    r = await SearchTool().execute({"pattern": "def solve"}, _ctx(fake))
    assert r.success is True
    assert "def solve" in r.output
    # Second exec call must be the rg subprocess.
    second_cmd = fake.exec_calls[1][0]
    assert isinstance(second_cmd, list) and second_cmd[0] == "rg"


@pytest.mark.asyncio
async def test_search_container_falls_back_to_grep() -> None:
    fake = FakeContainer(
        exec_responses=[
            # `command -v rg` probe — rg missing.
            ExecResult(stdout="", stderr="", exit_code=1, timed_out=False),
            # grep -rnE call
            ExecResult(
                stdout="foo.py:5:match\n",
                stderr="", exit_code=0, timed_out=False,
            ),
        ],
    )
    r = await SearchTool().execute({"pattern": "match"}, _ctx(fake))
    assert r.success is True
    assert "match" in r.output
    # Second exec call is the grep pipeline string.
    second_cmd = fake.exec_calls[1][0]
    assert isinstance(second_cmd, str) and second_cmd.startswith("grep ")


@pytest.mark.asyncio
async def test_search_container_no_matches_rg_exit_1() -> None:
    fake = FakeContainer(
        exec_responses=[
            ExecResult(stdout="/usr/bin/rg\n", stderr="", exit_code=0, timed_out=False),
            ExecResult(stdout="", stderr="", exit_code=1, timed_out=False),
        ],
    )
    r = await SearchTool().execute({"pattern": "nope"}, _ctx(fake))
    assert r.success is True
    assert "No matches" in r.output


# ---------------------------------------------------------------------------
# declare_done metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declare_done_emits_both_metadata_keys() -> None:
    """Both legacy (status/summary) and new (declared_*) keys are present."""
    fake = FakeContainer()
    r = await DeclareDoneTool().execute(
        {"status": "fixed", "summary": "patched bug in foo()"},
        _ctx(fake),
    )
    assert r.success is True
    assert r.metadata["status"] == "fixed"
    assert r.metadata["declared_status"] == "fixed"
    assert r.metadata["summary"] == "patched bug in foo()"
    assert r.metadata["declared_summary"] == "patched bug in foo()"
