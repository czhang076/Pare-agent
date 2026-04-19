"""Tests for pare/sandbox/git_checkpoint.py.

Each test gets a fresh git repo in a temp directory so tests are fully
isolated.  We use real git operations — no mocking — because the whole
point of this module is to get git right.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pare.sandbox.git_checkpoint import GitCheckpoint, GitCheckpointError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _git(cwd: Path, *args: str) -> str:
    """Run a git command in the given directory."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    assert proc.returncode == 0, f"git {' '.join(args)} failed: {stderr.decode()}"
    return stdout.decode()


async def _init_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    await _git(repo, "init")
    await _git(repo, "config", "user.email", "test@test.com")
    await _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("# Test Repo\n")
    await _git(repo, "add", ".")
    await _git(repo, "commit", "-m", "initial commit")
    return repo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo(tmp_path: Path) -> Path:
    return await _init_repo(tmp_path)


@pytest.fixture
async def cp(repo: Path) -> GitCheckpoint:
    return GitCheckpoint(repo)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSetup:
    @pytest.mark.asyncio
    async def test_setup_creates_working_branch(self, cp: GitCheckpoint, repo: Path):
        original = await cp.setup()
        assert original == "master" or original == "main"
        assert cp.is_active
        assert cp.working_branch is not None
        assert cp.working_branch.startswith("pare/working-")

        # Verify we're on the working branch
        branch = (await _git(repo, "rev-parse", "--abbrev-ref", "HEAD")).strip()
        assert branch == cp.working_branch

    @pytest.mark.asyncio
    async def test_setup_auto_commits_dirty_tree(self, cp: GitCheckpoint, repo: Path):
        # Make the tree dirty
        (repo / "dirty.txt").write_text("uncommitted\n")
        await cp.setup()

        # The dirty file should have been auto-committed before branching
        assert cp.is_active
        log = await _git(repo, "log", "--oneline", "-3")
        assert "auto-save" in log

    @pytest.mark.asyncio
    async def test_setup_twice_raises(self, cp: GitCheckpoint):
        await cp.setup()
        with pytest.raises(GitCheckpointError, match="already active"):
            await cp.setup()

    @pytest.mark.asyncio
    async def test_setup_not_a_repo(self, tmp_path: Path):
        not_repo = tmp_path / "not_a_repo"
        not_repo.mkdir()
        cp = GitCheckpoint(not_repo)
        with pytest.raises(GitCheckpointError, match="not inside a git"):
            await cp.setup()


class TestCheckpoint:
    @pytest.mark.asyncio
    async def test_checkpoint_with_changes(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()

        (repo / "new_file.py").write_text("print('hello')\n")
        sha = await cp.checkpoint("added new_file.py")

        assert sha is not None
        assert len(sha) == 40  # full SHA
        assert len(cp.checkpoints) == 1
        assert cp.checkpoints[0].message == "added new_file.py"

    @pytest.mark.asyncio
    async def test_checkpoint_no_changes_returns_none(self, cp: GitCheckpoint):
        await cp.setup()
        sha = await cp.checkpoint("nothing happened")
        assert sha is None
        assert len(cp.checkpoints) == 0

    @pytest.mark.asyncio
    async def test_multiple_checkpoints(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()

        (repo / "a.py").write_text("a\n")
        sha1 = await cp.checkpoint("step 1")

        (repo / "b.py").write_text("b\n")
        sha2 = await cp.checkpoint("step 2")

        assert sha1 != sha2
        assert len(cp.checkpoints) == 2
        assert cp.last_checkpoint_sha == sha2

    @pytest.mark.asyncio
    async def test_checkpoint_requires_active(self, cp: GitCheckpoint):
        with pytest.raises(GitCheckpointError, match="setup"):
            await cp.checkpoint("nope")


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_to_previous(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()

        (repo / "a.py").write_text("a\n")
        await cp.checkpoint("step 1")

        (repo / "b.py").write_text("b\n")
        await cp.checkpoint("step 2")

        assert (repo / "b.py").exists()

        await cp.rollback()

        # b.py should be gone after rollback
        assert (repo / "a.py").exists()
        assert not (repo / "b.py").exists()

    @pytest.mark.asyncio
    async def test_rollback_to_specific_sha(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()

        (repo / "a.py").write_text("a\n")
        sha1 = await cp.checkpoint("step 1")

        (repo / "b.py").write_text("b\n")
        await cp.checkpoint("step 2")

        (repo / "c.py").write_text("c\n")
        await cp.checkpoint("step 3")

        await cp.rollback(sha1)

        assert (repo / "a.py").exists()
        assert not (repo / "b.py").exists()
        assert not (repo / "c.py").exists()

    @pytest.mark.asyncio
    async def test_rollback_requires_active(self, cp: GitCheckpoint):
        with pytest.raises(GitCheckpointError, match="setup"):
            await cp.rollback()


class TestDiff:
    @pytest.mark.asyncio
    async def test_diff_since_original(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()

        (repo / "new.py").write_text("code\n")
        await cp.checkpoint("added new.py")

        diff = await cp.get_diff_since()
        assert "new.py" in diff

    @pytest.mark.asyncio
    async def test_full_diff(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()

        (repo / "new.py").write_text("code\n")
        await cp.checkpoint("added")

        full = await cp.get_full_diff()
        assert "+code" in full

    @pytest.mark.asyncio
    async def test_diff_empty_when_no_changes(self, cp: GitCheckpoint):
        await cp.setup()
        diff = await cp.get_diff_since()
        assert diff.strip() == ""

    @pytest.mark.asyncio
    async def test_full_diff_excludes_pycache_and_pyc(
        self, cp: GitCheckpoint, repo: Path
    ):
        """pytest regenerates .pyc files during tool calls — they must not
        reach the SWE-bench harness or `patch` chokes on binary hunks
        before applying the real .py edits.
        """
        await cp.setup()

        (repo / "real.py").write_text("print('x')\n")
        pycache = repo / "__pycache__"
        pycache.mkdir(exist_ok=True)
        (pycache / "real.cpython-312.pyc").write_bytes(b"\x00\x01fakebytes\x02")
        nested = repo / "pkg" / "__pycache__"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "mod.cpython-312.pyc").write_bytes(b"\x03morebytes")
        (repo / "stray.pyc").write_bytes(b"\x04")

        await cp.checkpoint("mixed real + pyc changes")

        full = await cp.get_full_diff()
        assert "real.py" in full
        assert "__pycache__" not in full
        assert ".pyc" not in full


class TestFinalize:
    @pytest.mark.asyncio
    async def test_finalize_squash_merges(self, cp: GitCheckpoint, repo: Path):
        original = await cp.setup()

        (repo / "a.py").write_text("a\n")
        await cp.checkpoint("step 1")

        (repo / "b.py").write_text("b\n")
        await cp.checkpoint("step 2")

        sha = await cp.finalize()

        assert sha is not None
        assert not cp.is_active

        # Should be back on original branch
        branch = (await _git(repo, "rev-parse", "--abbrev-ref", "HEAD")).strip()
        assert branch == original

        # Files should exist
        assert (repo / "a.py").exists()
        assert (repo / "b.py").exists()

        # Working branch should be deleted
        branches = await _git(repo, "branch")
        assert "pare/working" not in branches

        # Should be a single squash commit (not 2 separate)
        log = await _git(repo, "log", "--oneline", "-5")
        assert "pare: apply agent changes" in log

    @pytest.mark.asyncio
    async def test_finalize_no_changes(self, cp: GitCheckpoint, repo: Path):
        await cp.setup()
        sha = await cp.finalize()
        assert sha is None
        assert not cp.is_active

    @pytest.mark.asyncio
    async def test_finalize_requires_active(self, cp: GitCheckpoint):
        with pytest.raises(GitCheckpointError, match="setup"):
            await cp.finalize()


class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_discards_changes(self, cp: GitCheckpoint, repo: Path):
        original = await cp.setup()

        (repo / "bad.py").write_text("bad code\n")
        await cp.checkpoint("bad step")

        await cp.abort()

        assert not cp.is_active
        branch = (await _git(repo, "rev-parse", "--abbrev-ref", "HEAD")).strip()
        assert branch == original

        # Changes should NOT be on the original branch
        assert not (repo / "bad.py").exists()

    @pytest.mark.asyncio
    async def test_abort_requires_active(self, cp: GitCheckpoint):
        with pytest.raises(GitCheckpointError, match="setup"):
            await cp.abort()
