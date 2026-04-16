"""Ghost Sandbox for Multiverse Execution.

Uses `git worktree` to instantiate totally isolated filesystem clones
at near-zero cost in milliseconds. This allows parallel execution
of different agent strategies without git lock contention or 
dirtying the original checkout.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator
from contextlib import asynccontextmanager

from pare.sandbox.git_checkpoint import GitCheckpointError, _GIT

logger = logging.getLogger(__name__)


@dataclass
class UniverseConfig:
    """Configuration and state for a single parallel universe."""
    universe_id: str
    worktree_path: Path
    branch_name: str


class GhostWorktreeManager:
    """Manages isolated git worktrees for multiverse parallel execution."""

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir
        self.universes: dict[str, UniverseConfig] = {}

    async def _run(self, *args: str, cwd: Path | None = None) -> str:
        """Run a git command and return stdout."""
        target_cwd = cwd or self.repo_dir
        cmd = [_GIT, *args]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(target_cwd),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise GitCheckpointError(f"Git command timed out: {' '.join(cmd)}")
            
        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise GitCheckpointError(f"Git command failed: {' '.join(cmd)}\n{err}")
        return stdout.decode()

    async def _is_dirty(self) -> bool:
        """Check for uncommitted changes in main repo."""
        status = await self._run("status", "--porcelain")
        return bool(status.strip())

    async def spawn_universe(self, universe_id: str, base_commit: str | None = None) -> UniverseConfig:
        """Create a new isolated worktree for a universe."""
        if universe_id in self.universes:
            raise ValueError(f"Universe {universe_id} already exists")

        # Auto-stash/commit primary repo if dirty to safely set base commit
        if await self._is_dirty():
            await self._run("add", "-A")
            await self._run("commit", "-m", "pare: auto-save before multiverse spawn", "--allow-empty")
            logger.info("Auto-committed changes in main repo before spawning universe %s.", universe_id)

        target_base = base_commit or "HEAD"
        branch_name = f"pare/univ-{universe_id}"
        
        # We place worktrees in the system temp directory so they don't pollute the local repo
        # but they still share the same .git object folder.
        tmp_base = Path(tempfile.gettempdir()) / "pare_universes"
        tmp_base.mkdir(parents=True, exist_ok=True)
        worktree_path = tmp_base / f"{self.repo_dir.name}_{universe_id}"

        # Clean it up if it already exists (from a crashed previous run)
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

        try:
            # git worktree add <path> -b <branch> <base>
            await self._run("worktree", "add", str(worktree_path), "-b", branch_name, target_base)
            logger.info("Spawned Universe '%s' at %s (branch: %s)", universe_id, worktree_path, branch_name)
        except GitCheckpointError as e:
            logger.error("Failed to spawn universe %s: %s", universe_id, e)
            raise

        config = UniverseConfig(
            universe_id=universe_id,
            worktree_path=worktree_path,
            branch_name=branch_name,
        )
        self.universes[universe_id] = config
        return config

    async def destroy_universe(self, universe_id: str) -> None:
        """Tear down a universe's worktree and delete its branch."""
        if universe_id not in self.universes:
            return
            
        config = self.universes.pop(universe_id)
        
        # Remove the worktree definition
        try:
            await self._run("worktree", "remove", "--force", str(config.worktree_path))
        except GitCheckpointError as e:
            logger.warning("Could not cleanly remove worktree %s: %s", config.worktree_path, e)
            # Hard delete if git worktree remove fails
            if config.worktree_path.exists():
                shutil.rmtree(config.worktree_path, ignore_errors=True)

        # Delete the underlying branch
        try:
            # Can't use self._run if we need to fall back to -D on original repo
            await self._run("branch", "-D", config.branch_name)
            logger.info("Destroyed Universe '%s' branch %s", universe_id, config.branch_name)
        except GitCheckpointError as e:
            logger.warning("Could not clean up branch %s for universe %s: %s", config.branch_name, universe_id, e)

    async def squash_merge_winner(self, winner_id: str) -> None:
        """Merge the winning universe back to the main repository."""
        if winner_id not in self.universes:
            raise KeyError(f"Winner universe {winner_id} not found.")

        config = self.universes[winner_id]
        logger.info("Merging winner Universe '%s' from branch %s", winner_id, config.branch_name)

        # Ensure winner cleanly commits its current state
        try:
            # We add and commit inside the worktree
            # Use explicit git commands since _run defaults to repo_dir
            await self._run("add", "-A", cwd=config.worktree_path)
            await self._run("commit", "-m", f"pare: winner {winner_id} final state", "--allow-empty", cwd=config.worktree_path)
        except GitCheckpointError:
            # Might just be nothing to commit
            pass

        current_branch = (await self._run("rev-parse", "--abbrev-ref", "HEAD")).strip()

        # Squash merge into original repo
        # Pulls the code from the parallel branch
        await self._run("merge", "--squash", config.branch_name)
        await self._run("commit", "-m", f"pare: fix resolved by Multiverse (Universe {winner_id})")

    @asynccontextmanager
    async def multiverse_session(self, universe_ids: list[str], base_commit: str | None = None) -> AsyncIterator[dict[str, UniverseConfig]]:
        """Context manager to spawn universes and ensure cleanup on exit/crash."""
        try:
            configs = {}
            for uid in universe_ids:
                configs[uid] = await self.spawn_universe(uid, base_commit=base_commit)
            yield configs
        finally:
            for uid in list(self.universes.keys()):
                await self.destroy_universe(uid)

