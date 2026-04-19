"""Git-based checkpoint and rollback system.

Every agent mutation is preceded by a git commit on an isolated working
branch.  If a step fails the agent can roll back atomically.  When the
task is done, the working branch is squash-merged back so the user's
history stays clean.

Lifecycle:
    setup()          → create pare/working branch from HEAD
    checkpoint(msg)  → stage all + commit (skip if clean)
    rollback(sha)    → hard-reset to a prior checkpoint
    get_diff_since() → unified diff for LLM context
    finalize()       → squash-merge back to original branch
    abort()          → discard working branch, return to original
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Git binary — can be overridden for tests
_GIT = "git"

# Pathspec exclusions appended to `git diff` calls. pytest regenerates
# .pyc files during tool-call iterations; if those land in final_diff the
# SWE-bench harness's `patch` fallback chokes on binary hunks before
# reaching the real .py edits.
_DIFF_EXCLUDES: tuple[str, ...] = (
    "--",
    ":(exclude,glob)**/__pycache__/**",
    ":(exclude,glob)**/*.pyc",
    ":(exclude,glob)**/*.pyo",
    ":(exclude,glob)**/*.pyd",
    ":(exclude,glob).pare/**",
)


@dataclass
class CheckpointInfo:
    """Metadata about a single checkpoint."""

    sha: str
    message: str


class GitCheckpointError(Exception):
    """Raised when a git operation fails unexpectedly."""


class GitCheckpoint:
    """Manages a working branch with commit-based checkpoints.

    The caller (executor/orchestrator) is responsible for calling these
    methods at the right points in the agent loop.  This class is purely
    a git-operation wrapper — it has no knowledge of plan steps or tools.
    """

    BRANCH_PREFIX = "pare/working"

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd
        self._original_branch: str | None = None
        self._working_branch: str | None = None
        self._checkpoints: list[CheckpointInfo] = []
        self._active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def setup(self) -> str:
        """Create a working branch from the current HEAD.

        Returns the original branch name (to restore later).
        Raises GitCheckpointError if the repo is not a git repo or if
        there are already uncommitted changes that can't be stashed.
        """
        if self._active:
            raise GitCheckpointError("Checkpoint already active — call finalize() or abort() first.")

        # Verify git repo
        if not await self._is_git_repo():
            raise GitCheckpointError(f"{self.cwd} is not inside a git repository.")

        # Get current branch
        self._original_branch = await self._current_branch()

        # Ensure working tree is clean (or auto-stash)
        if await self._is_dirty():
            # Auto-commit uncommitted changes so we don't lose them
            await self._run("add", "-A")
            await self._run(
                "commit", "-m", "pare: auto-save before agent session",
                "--allow-empty",
            )
            logger.info("Auto-committed uncommitted changes before setup.")

        # Create unique working branch name
        short_sha = (await self._run("rev-parse", "--short", "HEAD")).strip()
        self._working_branch = f"{self.BRANCH_PREFIX}-{short_sha}"

        # Create and switch to working branch
        await self._run("checkout", "-b", self._working_branch)
        self._active = True

        logger.info(
            "Git checkpoint active: %s → %s",
            self._original_branch,
            self._working_branch,
        )
        return self._original_branch

    async def checkpoint(self, message: str) -> str | None:
        """Stage all changes and commit.

        Returns the commit SHA, or None if there were no changes.
        """
        self._require_active()

        if not await self._is_dirty():
            logger.debug("No changes to checkpoint: %s", message)
            return None

        await self._run("add", "-A")
        await self._run("commit", "-m", f"pare: {message}")

        sha = (await self._run("rev-parse", "HEAD")).strip()
        self._checkpoints.append(CheckpointInfo(sha=sha, message=message))

        logger.info("Checkpoint %s: %s", sha[:8], message)
        return sha

    async def rollback(self, to_sha: str | None = None) -> None:
        """Hard-reset to a prior checkpoint.

        If to_sha is None, rolls back to the checkpoint before the most
        recent one (i.e. undoes the last step).  If no checkpoints exist,
        rolls back to the branch start point.
        """
        self._require_active()

        if to_sha is None:
            if len(self._checkpoints) >= 2:
                to_sha = self._checkpoints[-2].sha
            elif len(self._checkpoints) == 1:
                # Roll back to before the first checkpoint
                to_sha = f"{self._checkpoints[0].sha}~1"
            else:
                # No checkpoints — reset to branch start
                to_sha = self._original_branch

        await self._run("reset", "--hard", to_sha)

        # Remove rolled-back checkpoints from our list
        current_sha = (await self._run("rev-parse", "HEAD")).strip()
        self._checkpoints = [
            cp for cp in self._checkpoints if cp.sha == current_sha
            or await self._is_ancestor(cp.sha, current_sha)
        ]

        logger.info("Rolled back to %s", to_sha)

    async def get_diff_since(self, sha: str | None = None) -> str:
        """Get a diff summary since the given SHA.

        If sha is None, diffs against the original branch.
        Returns unified diff string (can be empty).
        """
        self._require_active()
        base = sha or self._original_branch
        try:
            return await self._run("diff", "--stat", base, "HEAD", *_DIFF_EXCLUDES)
        except GitCheckpointError:
            return ""

    async def get_full_diff(self, sha: str | None = None) -> str:
        """Get the full unified diff since the given SHA."""
        self._require_active()
        base = sha or self._original_branch
        try:
            return await self._run("diff", base, "HEAD", *_DIFF_EXCLUDES)
        except GitCheckpointError:
            return ""

    async def finalize(self) -> str | None:
        """Squash-merge working branch back to original, clean up.

        Returns the merge commit SHA, or None if there were no changes.
        """
        self._require_active()

        # Check if there are any changes to merge
        diff = await self._run("diff", self._original_branch, "HEAD", "--stat")
        if not diff.strip():
            # No changes — just switch back and delete working branch
            await self._cleanup()
            return None

        # Switch to original branch
        await self._run("checkout", self._original_branch)

        # Squash merge
        await self._run("merge", "--squash", self._working_branch)
        await self._run("commit", "-m", "pare: apply agent changes")

        sha = (await self._run("rev-parse", "HEAD")).strip()

        # Delete working branch
        await self._run("branch", "-D", self._working_branch)

        self._active = False
        self._checkpoints.clear()
        logger.info("Finalized: squash-merged to %s (%s)", self._original_branch, sha[:8])
        return sha

    async def abort(self) -> None:
        """Discard all working branch changes, return to original branch."""
        self._require_active()
        await self._cleanup()
        logger.info("Aborted: returned to %s", self._original_branch)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def original_branch(self) -> str | None:
        return self._original_branch

    @property
    def working_branch(self) -> str | None:
        return self._working_branch

    @property
    def checkpoints(self) -> list[CheckpointInfo]:
        return list(self._checkpoints)

    @property
    def last_checkpoint_sha(self) -> str | None:
        return self._checkpoints[-1].sha if self._checkpoints else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_active(self) -> None:
        if not self._active:
            raise GitCheckpointError("No active checkpoint session. Call setup() first.")

    async def _is_git_repo(self) -> bool:
        try:
            await self._run("rev-parse", "--git-dir")
            return True
        except GitCheckpointError:
            return False

    async def _current_branch(self) -> str:
        result = await self._run("rev-parse", "--abbrev-ref", "HEAD")
        return result.strip()

    async def _is_dirty(self) -> bool:
        """Check for any uncommitted changes (staged or unstaged or untracked)."""
        status = await self._run("status", "--porcelain")
        return bool(status.strip())

    async def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is an ancestor of descendant."""
        try:
            await self._run("merge-base", "--is-ancestor", ancestor, descendant)
            return True
        except GitCheckpointError:
            return False

    async def _cleanup(self) -> None:
        """Switch back to original branch and delete working branch."""
        await self._run("checkout", self._original_branch)
        try:
            await self._run("branch", "-D", self._working_branch)
        except GitCheckpointError:
            logger.warning("Could not delete working branch %s", self._working_branch)
        self._active = False
        self._checkpoints.clear()

    async def _run(self, *args: str) -> str:
        """Run a git command and return stdout.

        Raises GitCheckpointError on non-zero exit.
        """
        cmd = [_GIT, *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise GitCheckpointError(f"Git command timed out: {' '.join(cmd)}")
        except FileNotFoundError:
            raise GitCheckpointError(f"Git not found. Is git installed and on PATH?")

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            raise GitCheckpointError(
                f"git {' '.join(args)} failed (rc={proc.returncode}): {err_text}"
            )

        return stdout.decode("utf-8", errors="replace")
