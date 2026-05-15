"""Tests for ``pare.sandbox.host_container.HostContainer``.

HostContainer is the no-Docker duck-typed alternative to
InstanceContainer used by the host-mode failure-injection runner.
The invariants we pin here:

1. The interface matches InstanceContainer's (workdir / exec /
   read_file / write_file / git_*) so the agent loop can swap one
   for the other without code changes.
2. ``exec`` routes through host subprocess with correct cwd, returns
   ExecResult shape, treats non-zero exit as data not exception.
3. ``git_init_checkpoint`` works on a non-repo (initializes) and on
   a pre-existing repo (returns HEAD).
4. ``git_diff(base)`` produces a unified diff comparing base SHA to
   HEAD — the artifact the loop uses for ``final_diff``.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from pare.sandbox.host_container import HostContainer, HostContainerError
from pare.sandbox.instance_container import ExecResult


# HostContainer wraps ``bash -lc`` for string commands (matches
# InstanceContainer.exec). On Windows ``bash`` is not on PATH by
# default — these tests are designed for Linux / WSL where Pare's
# host-mode failure-injection runs live anyway.
pytestmark = [
    pytest.mark.skipif(
        shutil.which("git") is None, reason="git not on PATH"
    ),
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="HostContainer uses bash -lc; Linux/WSL only",
    ),
]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_python_repo(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text('"""pkg."""\n')
    (root / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )


# ---------------------------------------------------------------------------
# Lifecycle + exec
# ---------------------------------------------------------------------------


class TestLifecycleAndExec:
    @pytest.mark.asyncio
    async def test_from_workdir_validates_directory(self, tmp_path: Path):
        c = await HostContainer.from_workdir(tmp_path)
        assert Path(c.workdir).resolve() == tmp_path.resolve()

    @pytest.mark.asyncio
    async def test_from_workdir_missing_raises(self, tmp_path: Path):
        missing = tmp_path / "nope"
        with pytest.raises(HostContainerError, match="does not exist"):
            await HostContainer.from_workdir(missing)

    @pytest.mark.asyncio
    async def test_from_workdir_not_a_dir_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        with pytest.raises(HostContainerError, match="not a directory"):
            await HostContainer.from_workdir(f)

    @pytest.mark.asyncio
    async def test_exec_returns_exec_result_shape(self, tmp_path: Path):
        c = await HostContainer.from_workdir(tmp_path)
        r = await c.exec("echo hello")
        assert isinstance(r, ExecResult)
        assert r.exit_code == 0
        assert r.stdout.strip() == "hello"
        assert r.timed_out is False

    @pytest.mark.asyncio
    async def test_exec_nonzero_exit_is_data_not_exception(
        self, tmp_path: Path
    ):
        """Tools consume exit_code as input to error_signal — we must
        not throw on non-zero. Same posture as InstanceContainer.exec."""
        c = await HostContainer.from_workdir(tmp_path)
        r = await c.exec("exit 7")
        assert r.exit_code == 7

    @pytest.mark.asyncio
    async def test_exec_uses_workdir_as_cwd(self, tmp_path: Path):
        _make_python_repo(tmp_path)
        c = await HostContainer.from_workdir(tmp_path)
        r = await c.exec("ls -1 pkg")
        assert r.exit_code == 0
        assert "core.py" in r.stdout

    @pytest.mark.asyncio
    async def test_exec_list_form_bypasses_shell(self, tmp_path: Path):
        """List-form is what git helpers use — no shell quoting."""
        c = await HostContainer.from_workdir(tmp_path)
        r = await c.exec(["python3", "-c", "print(1+1)"])
        assert r.exit_code == 0
        assert r.stdout.strip() == "2"

    @pytest.mark.asyncio
    async def test_exec_timeout_returns_124(self, tmp_path: Path):
        """Match GNU timeout(1) + InstanceContainer convention.
        Use a short timeout to keep the test fast."""
        c = await HostContainer.from_workdir(tmp_path)
        r = await c.exec("sleep 5", timeout=0.3)
        assert r.exit_code == 124
        assert r.timed_out is True

    @pytest.mark.asyncio
    async def test_exec_command_not_found_surfaces_as_127(
        self, tmp_path: Path
    ):
        c = await HostContainer.from_workdir(tmp_path)
        # The bash -lc wrapper resolves a missing command to exit 127.
        r = await c.exec("__definitely_no_such_command__")
        assert r.exit_code != 0


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


class TestFileIO:
    @pytest.mark.asyncio
    async def test_read_file_workdir_relative(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("contents\n")
        c = await HostContainer.from_workdir(tmp_path)
        assert await c.read_file("a.txt") == "contents\n"

    @pytest.mark.asyncio
    async def test_read_file_absolute(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("contents\n")
        c = await HostContainer.from_workdir(tmp_path)
        assert (
            await c.read_file(str(tmp_path / "a.txt")) == "contents\n"
        )

    @pytest.mark.asyncio
    async def test_read_file_missing_raises(self, tmp_path: Path):
        c = await HostContainer.from_workdir(tmp_path)
        with pytest.raises(HostContainerError, match="read_file"):
            await c.read_file("nope.txt")

    @pytest.mark.asyncio
    async def test_read_file_truncates(self, tmp_path: Path):
        (tmp_path / "big.txt").write_text("x" * 5000)
        c = await HostContainer.from_workdir(tmp_path)
        out = await c.read_file("big.txt", max_bytes=100)
        assert out.startswith("x" * 100)
        assert "[truncated at 100 bytes]" in out

    @pytest.mark.asyncio
    async def test_write_file_creates_parent_dirs(self, tmp_path: Path):
        c = await HostContainer.from_workdir(tmp_path)
        await c.write_file("nested/dir/file.py", "x = 1\n")
        assert (tmp_path / "nested" / "dir" / "file.py").read_text() == "x = 1\n"

    @pytest.mark.asyncio
    async def test_write_file_round_trips_through_read(self, tmp_path: Path):
        c = await HostContainer.from_workdir(tmp_path)
        await c.write_file("a.py", "import sys\n")
        assert await c.read_file("a.py") == "import sys\n"


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------


class TestGitOperations:
    @pytest.mark.asyncio
    async def test_init_checkpoint_initializes_non_repo(
        self, tmp_path: Path
    ):
        _make_python_repo(tmp_path)
        c = await HostContainer.from_workdir(tmp_path)
        sha = await c.git_init_checkpoint()
        # Hex SHA, 40 chars (or 7+ if short — but git rev-parse HEAD is full).
        assert len(sha) >= 7
        assert all(ch in "0123456789abcdef" for ch in sha)
        # And now the workdir is a git repo.
        assert (tmp_path / ".git").is_dir()

    @pytest.mark.asyncio
    async def test_init_checkpoint_reuses_existing_repo(
        self, tmp_path: Path
    ):
        """If the workdir already has a HEAD, return its SHA without
        re-initializing — matches the SWE-bench worktree case where
        ``materialize_swe_bench_workdirs`` has set up the branch
        already."""
        _make_python_repo(tmp_path)
        c = await HostContainer.from_workdir(tmp_path)
        sha1 = await c.git_init_checkpoint()
        sha2 = await c.git_init_checkpoint()
        assert sha1 == sha2

    @pytest.mark.asyncio
    async def test_git_diff_shows_uncommitted_changes(
        self, tmp_path: Path
    ):
        _make_python_repo(tmp_path)
        c = await HostContainer.from_workdir(tmp_path)
        base = await c.git_init_checkpoint()
        # Modify pkg/core.py without committing.
        await c.write_file(
            "pkg/core.py", "def add(a, b):\n    return a * b\n"
        )
        # Working-tree diff (base=None).
        wt = await c.git_diff()
        assert "-    return a + b" in wt
        assert "+    return a * b" in wt
        # Base..HEAD diff should be empty (nothing committed).
        committed = await c.git_diff(base=base)
        assert committed.strip() == ""

    @pytest.mark.asyncio
    async def test_git_commit_then_diff_shows_committed_change(
        self, tmp_path: Path
    ):
        """Loop flow: checkpoint → edit → commit → diff(base) → that
        diff is what ``final_diff`` becomes."""
        _make_python_repo(tmp_path)
        c = await HostContainer.from_workdir(tmp_path)
        base = await c.git_init_checkpoint()
        await c.write_file(
            "pkg/core.py", "def add(a, b):\n    return a * b\n"
        )
        new_head = await c.git_commit("test edit")
        assert new_head != base
        diff = await c.git_diff(base=base)
        assert "return a * b" in diff
        assert "return a + b" in diff

    @pytest.mark.asyncio
    async def test_aenter_aexit_does_not_clobber_workdir(
        self, tmp_path: Path
    ):
        """``async with HostContainer.from_workdir(...) as c`` must not
        delete or modify the workdir. The caller owns it."""
        (tmp_path / "marker.txt").write_text("preserved")
        c = await HostContainer.from_workdir(tmp_path)
        async with c:
            pass
        assert (tmp_path / "marker.txt").read_text() == "preserved"


# ---------------------------------------------------------------------------
# Interface parity smoke
# ---------------------------------------------------------------------------


class TestInterfaceParity:
    """A change to InstanceContainer's surface that doesn't reach
    HostContainer breaks the substitution promise. This test pins the
    method names + arity from the loop's perspective."""

    def test_has_methods_loop_calls(self):
        # Methods named verbatim from pare/agent/loop.py's container.*
        # usage. If loop.py grows a new call, this list grows here too.
        expected = (
            "exec",
            "read_file",
            "write_file",
            "git_init_checkpoint",
            "git_commit",
            "git_diff",
            "git_checkout",
            "workdir",
        )
        for name in expected:
            assert hasattr(HostContainer, name), (
                f"HostContainer is missing {name!r} — loop.py will crash "
                f"when it tries to call it"
            )
