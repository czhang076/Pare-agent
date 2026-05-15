"""Host-mode container — duck-typed alternative to InstanceContainer.

Why this exists
---------------

``InstanceContainer`` boots a Docker container per SWE-bench instance
and routes every agent tool call through ``docker exec``. That gives
SWE-bench eval parity (the agent runs in the same image the harness
uses to score the diff) at the cost of Docker setup.

``HostContainer`` implements the same surface (``exec`` / ``read_file``
/ ``write_file`` / ``git_*`` / ``workdir``) but routes through plain
``asyncio.create_subprocess_exec`` on the host. Use it when:

- you don't have Docker (CI, dev box, WSL without integration)
- you don't need Tier-2 SWE-bench eval (e.g. failure-injection runs
  that judge outcome via classifier_liu on the trajectory, not via
  running the official test suite)
- you accept the loss of environment parity — Python version, OS
  libraries, installed deps are whatever the host happens to have

Non-goals
---------

- Sandboxing. We run user-defined ``bash`` commands on the host with
  the calling user's permissions. Do not run untrusted models through
  this on a host that holds anything you care about.
- Tier-2 verification. The official SWE-bench harness expects its
  own image; we don't try to replicate that on host.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Optional

# Re-export the same ExecResult shape so callers can use either
# container type interchangeably.
from pare.sandbox.instance_container import ExecResult

logger = logging.getLogger(__name__)


class HostContainerError(RuntimeError):
    """Raised on host-side I/O / process failures.

    Distinct from ``ExecResult.exit_code != 0`` — that's normal tool
    output. This exception is for situations the caller can't recover
    from (workdir missing, git not on PATH, etc.).
    """


class HostContainer:
    """Run agent tools against a host workdir, no Docker.

    Usage::

        async with await HostContainer.from_workdir(workdir) as c:
            await c.exec("ls")
            await c.write_file("/abs/path/foo.py", new_content)
            diff = await c.git_diff(base=base_sha)

    The interface intentionally matches ``InstanceContainer`` so the
    agent loop (``pare.agent.loop.run_agent``) accepts either type.
    """

    def __init__(self, workdir: str) -> None:
        # Stored as str to match InstanceContainer.workdir (also str).
        self._workdir = str(Path(workdir).resolve())

    # ---- lifecycle ---------------------------------------------------------

    @classmethod
    async def from_workdir(cls, workdir: str | Path) -> "HostContainer":
        """Construct + validate. Mirrors InstanceContainer.build()'s
        async-classmethod shape so callers can swap one for the other.

        Raises HostContainerError if workdir doesn't exist or isn't a
        directory — caller is expected to have materialized it (via
        ``experiments.materialize_swe_bench_workdirs`` or equivalent)
        before this call.
        """
        wd = Path(workdir)
        if not wd.exists():
            raise HostContainerError(f"workdir does not exist: {wd}")
        if not wd.is_dir():
            raise HostContainerError(f"workdir not a directory: {wd}")
        return cls(str(wd))

    async def __aenter__(self) -> "HostContainer":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Nothing to tear down — host workdir is owned by the caller.
        # If a fault was applied, the caller is responsible for revert.
        return None

    @property
    def workdir(self) -> str:
        return self._workdir

    # ---- exec --------------------------------------------------------------

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout: float = 60.0,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Run a command on the host with ``cwd=self.workdir``.

        Strings are wrapped in ``bash -lc`` to match InstanceContainer's
        behaviour (so shell metachars + pipes Just Work). Lists bypass
        the shell — what the git helpers use.

        Non-zero exit codes are returned as data; only host-level
        failures (PermissionError, etc.) raise. ``exit_code == 124`` is
        the timeout sentinel, matching GNU ``timeout(1)``.
        """
        run_cwd = cwd or self._workdir
        if isinstance(cmd, str):
            argv = ["bash", "-lc", cmd]
        else:
            argv = list(cmd)

        # Merge env (None means inherit the parent process env, which is
        # what InstanceContainer does for default behaviour too).
        merged_env: Optional[dict[str, str]]
        if env is None:
            merged_env = None
        else:
            merged_env = {**os.environ, **env}

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=run_cwd,
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            # bash missing, or argv[0] not on PATH. Surface as
            # exit_code != 0 so tools handle it like a normal failure.
            return ExecResult(
                stdout="",
                stderr=f"command not found: {e}",
                exit_code=127,
                timed_out=False,
            )

        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return ExecResult(
                stdout="",
                stderr=f"command timed out after {timeout:.0f}s",
                exit_code=124,
                timed_out=True,
            )

        return ExecResult(
            stdout=out_b.decode("utf-8", errors="replace"),
            stderr=err_b.decode("utf-8", errors="replace"),
            exit_code=int(proc.returncode or 0),
            timed_out=False,
        )

    # ---- file I/O ----------------------------------------------------------

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        """Read a text file from the host. Truncate at ``max_bytes``."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._workdir) / p
        try:
            data = await asyncio.to_thread(p.read_bytes)
        except FileNotFoundError as e:
            raise HostContainerError(f"read_file: {p}: {e}") from e
        if len(data) > max_bytes:
            text = data[:max_bytes].decode("utf-8", errors="replace")
            return text + f"\n[truncated at {max_bytes} bytes]"
        return data.decode("utf-8", errors="replace")

    async def write_file(self, path: str, content: str) -> None:
        """Write text to a host file. Path may be relative to workdir."""
        p = Path(path)
        if not p.is_absolute():
            p = Path(self._workdir) / p
        # Match InstanceContainer.write_file behaviour: parent must
        # exist (we don't auto-mkdir; that's the agent's responsibility
        # via a separate bash call). But we DO need the parent to exist
        # for the write to succeed at all, so be a touch more lenient:
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            p.write_text, content, "utf-8"
        )

    # ---- git ---------------------------------------------------------------
    #
    # The git helpers re-implement (rather than inherit) the
    # InstanceContainer versions because the parent class is tightly
    # coupled to docker-py state. The shell commands themselves are
    # identical — only the exec routing differs.

    async def git_init_checkpoint(self) -> str:
        """Record base commit SHA. If the workdir isn't a git repo, init one
        and create a baseline commit so ``git diff`` has something to
        diff against. SWE-bench workdirs ship as git worktrees, so the
        init branch is mostly a safety net for unit-test fixtures."""
        # Probe for git repo.
        check = await self.exec(["git", "rev-parse", "HEAD"], timeout=15.0)
        if check.exit_code != 0:
            # Initialize + commit current state as the baseline.
            init = await self.exec(["git", "init", "-q"], timeout=15.0)
            if init.exit_code != 0:
                raise HostContainerError(
                    f"git init failed: {init.stderr.strip()}"
                )
            await self.exec(["git", "add", "-A"], timeout=30.0)
            commit = await self.exec(
                [
                    "git",
                    "-c", "user.email=pare@local",
                    "-c", "user.name=pare",
                    "commit", "--allow-empty", "-q",
                    "-m", "pare: host-mode baseline",
                ],
                timeout=30.0,
            )
            if commit.exit_code != 0:
                raise HostContainerError(
                    f"git commit failed: {commit.stderr.strip()}"
                )
            check = await self.exec(["git", "rev-parse", "HEAD"], timeout=15.0)
            if check.exit_code != 0:
                raise HostContainerError(
                    f"git rev-parse after init failed: {check.stderr.strip()}"
                )
        return check.stdout.strip()

    async def git_commit(self, message: str = "pare: agent step") -> str:
        add = await self.exec(["git", "add", "-A"], timeout=30.0)
        if add.exit_code != 0:
            raise HostContainerError(
                f"git add failed: {add.stderr.strip()}"
            )
        commit = await self.exec(
            [
                "git",
                "-c", "user.email=pare@local",
                "-c", "user.name=pare",
                "commit", "--allow-empty", "-m", message,
            ],
            timeout=30.0,
        )
        if commit.exit_code != 0:
            raise HostContainerError(
                f"git commit failed: "
                f"{commit.stderr.strip() or commit.stdout.strip()}"
            )
        head = await self.exec(["git", "rev-parse", "HEAD"], timeout=15.0)
        return head.stdout.strip()

    async def git_diff(self, base: Optional[str] = None) -> str:
        if base is None:
            cmd = ["git", "diff"]
        else:
            cmd = ["git", "diff", f"{base}..HEAD"]
        r = await self.exec(cmd, timeout=30.0)
        if r.exit_code != 0:
            raise HostContainerError(f"git diff failed: {r.stderr.strip()}")
        return r.stdout

    async def git_checkout(self, ref: str) -> None:
        r = await self.exec(
            f"git checkout -- . && git checkout {shlex.quote(ref)}",
            timeout=30.0,
        )
        if r.exit_code != 0:
            raise HostContainerError(f"git checkout failed: {r.stderr.strip()}")
