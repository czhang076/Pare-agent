"""Long-lived per-instance Docker container for agent + Tier 2.

R0 scaffold — signatures only. Real implementation lands in R1.

Replaces the one-shot ``swebench.harness.run_instance`` pattern: instead of
building a container every time we verify a diff, we build one container per
instance at session start, run all agent tool calls inside it, then reuse
the same container for Tier 2 pytest. This is what makes host-vs-Docker env
mismatch go away — tool errors and test failures come from the same
Python / OS / dep versions.

Design notes (for R1 implementer):

- Async context manager. ``async with InstanceContainer.build(...)`` starts
  on entry, stops on exit (force-remove on exception path).
- ``exec(cmd, timeout, cwd)`` returns :class:`ExecResult`. Uses docker-py
  ``exec_run(stream=True, demux=True)`` so stdout/stderr are separated;
  outer ``asyncio.wait_for`` enforces timeout since docker-py's stream does
  not honour socket timeouts (verified in the R0 spike).
- File I/O: ``read_file`` uses ``exec("cat ...")`` (robust for source code
  but binary-unsafe). ``write_file`` uses docker-py ``put_archive`` with a
  one-file tar stream — binary-safe and no heredoc quoting issues.
- Git ops: subprocess-style ``exec("git ...")`` at ``/testbed``; swebench
  images ship with git + python + base deps pre-installed.

The container image is expected to be a **derived** image
``pare-eval.<iid>:latest`` built once by
:func:`pare.sandbox.image_builder.ensure_pare_image` — it layers ripgrep on
top of ``swebench/sweb.eval.x86_64.<iid>:latest``. We do **not** run
``apt-get`` at container start; that path measured 10–40 s per container
during the R0 spike, which is unacceptable for pilot wall time. If the
derived image is unavailable (e.g. offline), callers should fall back to
the bare swebench image and the search tool's ``grep -rn`` branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of a single ``exec`` call inside the container.

    Attributes:
        stdout: Decoded stdout (utf-8, errors replaced).
        stderr: Decoded stderr (utf-8, errors replaced).
        exit_code: Process exit code. 124 indicates timeout (matches the
            GNU ``timeout(1)`` convention). Non-zero exits are data, not
            exceptions — agent tools consume this as input to error_signal.
        timed_out: True if outer ``asyncio.wait_for`` tripped.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class InstanceContainerError(RuntimeError):
    """Raised for container-lifecycle or I/O failures.

    Distinct from ``ExecResult.exit_code != 0`` — the latter is normal
    tool output; this exception fires only when docker itself fails
    (daemon unreachable, container gone, put_archive refused).
    """


class InstanceContainer:
    """Long-lived Docker container scoped to one SWE-bench instance.

    Usage::

        async with await InstanceContainer.build(
            instance_id="sympy__sympy-11618",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
        ) as c:
            await c.exec("ls /testbed")
            await c.write_file("/testbed/sympy/foo.py", new_content)
            diff = await c.git_diff()

    All public methods are ``async``. Blocking docker-py calls are dispatched
    to ``asyncio.to_thread`` so they do not stall the event loop.
    """

    def __init__(
        self,
        instance_id: str,
        image_tag: str,
        client: Any,  # docker.DockerClient
        workdir: str = "/testbed",
        name: Optional[str] = None,
    ) -> None:
        self.instance_id = instance_id
        self.image_tag = image_tag
        self.workdir = workdir
        self.name = name or f"pare-{instance_id.replace('/', '_')}"
        self._client = client
        self._container: Any = None  # docker.models.containers.Container

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    async def build(
        cls,
        instance_id: str,
        *,
        dataset_name: str,
        split: str,
        namespace: str = "swebench",
    ) -> "InstanceContainer":
        """Resolve TestSpec → image_tag, create docker client, return instance.

        Does NOT start the container yet — ``__aenter__`` does. Callers that
        do not use ``async with`` must call ``_start()`` / ``_stop()``
        themselves (discouraged; tests only).
        """
        raise NotImplementedError("R1: resolve _spec_for + ensure_pare_image, return cls(...)")

    async def __aenter__(self) -> "InstanceContainer":
        raise NotImplementedError("R1: call self._start() and return self")

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        raise NotImplementedError("R1: call self._stop(); swallow errors in cleanup")

    async def _start(self) -> None:
        """Create + start container with the ``tail -f /dev/null`` keep-alive."""
        raise NotImplementedError("R1")

    async def _stop(self) -> None:
        """Kill + force-remove the container. Idempotent; safe on failure."""
        raise NotImplementedError("R1")

    # ---- exec --------------------------------------------------------------

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout: float = 60.0,
        cwd: Optional[str] = None,
    ) -> ExecResult:
        """Run a command inside the container.

        Returns ``ExecResult``; never raises on non-zero exit — agent tools
        need ``exit_code`` as data. Raises :class:`InstanceContainerError`
        only for docker-level failures.
        """
        raise NotImplementedError("R1")

    # ---- file I/O ----------------------------------------------------------

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        """Read a text file from the container. Raises if not found.

        Truncates to ``max_bytes`` with a trailing marker; caller handles
        the full-file case by reading in chunks if needed.
        """
        raise NotImplementedError("R1")

    async def write_file(self, path: str, content: str) -> None:
        """Write text to a file inside the container via ``put_archive``.

        Binary-safe (unlike heredoc); handles arbitrary content with newlines,
        quotes, backticks. Path must be absolute.
        """
        raise NotImplementedError("R1")

    # ---- git ---------------------------------------------------------------

    async def git_init_checkpoint(self) -> str:
        """Record base commit; returns the SHA for later diff/reset."""
        raise NotImplementedError("R1")

    async def git_commit(self, message: str = "pare: agent step") -> str:
        """Stage all changes and commit; returns new SHA.

        Uses ``--allow-empty`` so a no-op step does not fail the loop.
        """
        raise NotImplementedError("R1")

    async def git_diff(self, base: Optional[str] = None) -> str:
        """Return unified diff from ``base`` (or initial checkpoint) to HEAD."""
        raise NotImplementedError("R1")

    async def git_checkout(self, ref: str) -> None:
        """Discard working-tree changes and check out ``ref``."""
        raise NotImplementedError("R1")
