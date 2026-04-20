"""Long-lived per-instance Docker container for agent + Tier 2.

Replaces the one-shot ``swebench.harness.run_instance`` pattern: instead of
building a container every time we verify a diff, we build one container per
instance at session start, run all agent tool calls inside it, then reuse
the same container for Tier 2 pytest. This is what makes host-vs-Docker env
mismatch go away — tool errors and test failures come from the same
Python / OS / dep versions.

Design notes:

- Async context manager. ``async with await InstanceContainer.build(...)``
  starts on entry, stops on exit (force-remove on exception path).
- :meth:`exec` returns :class:`ExecResult`. Uses docker-py
  ``exec_run(stream=True, demux=True)`` so stdout/stderr are separated;
  outer ``asyncio.wait_for`` enforces timeout since docker-py's stream does
  not honour socket timeouts (verified in the R0 spike).
- File I/O: :meth:`read_file` uses ``exec("cat ...")`` (robust for source
  code but binary-unsafe). :meth:`write_file` uses docker-py ``put_archive``
  with a one-file tar stream — binary-safe and no heredoc quoting issues.
- Git ops: subprocess-style ``exec("git ...")`` at ``/testbed``; swebench
  images ship with git + python + base deps pre-installed.

The container image is the **derived** image ``pare-eval.<iid>:latest``
built once by :func:`pare.sandbox.image_builder.ensure_pare_image`. It
layers ripgrep on top of ``swebench/sweb.eval.x86_64.<iid>:latest`` so the
``search`` tool stays fast without per-container apt-get. If
``ensure_pare_image`` returns the base tag (offline fallback), the
``search`` tool's grep branch takes over — the container still works.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import tarfile
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


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
    to :func:`asyncio.to_thread` so they do not stall the event loop.
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
        self.name = name or _safe_container_name(instance_id)
        self._client = client
        self._container: Any = None  # docker.models.containers.Container

    # ---- lifecycle --------------------------------------------------------

    @classmethod
    async def build(
        cls,
        instance_id: str,
        *,
        dataset_name: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
        allow_offline_fallback: bool = True,
    ) -> "InstanceContainer":
        """Resolve image tag (building the derived image on demand), return instance.

        Does NOT start the container yet — ``__aenter__`` does. Callers that
        do not use ``async with`` must call ``_start()`` / ``_stop()``
        themselves (discouraged; tests only).
        """
        try:
            import docker  # type: ignore
        except ImportError as e:
            raise InstanceContainerError(
                "docker-py not installed; pip install docker"
            ) from e

        # Lazy import to keep this module usable when docker_eval's heavy
        # deps (datasets/swebench) are not installed.
        from pare.sandbox.image_builder import ensure_pare_image

        image_tag = await ensure_pare_image(
            instance_id,
            dataset_name=dataset_name,
            split=split,
            allow_offline_fallback=allow_offline_fallback,
        )

        client = docker.from_env()
        try:
            await asyncio.to_thread(client.ping)
        except Exception as e:
            raise InstanceContainerError(
                f"docker daemon unreachable: {e}"
            ) from e

        return cls(instance_id=instance_id, image_tag=image_tag, client=client)

    async def __aenter__(self) -> "InstanceContainer":
        await self._start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Always attempt cleanup, even on exception, but never mask the
        # original error by raising from _stop.
        try:
            await self._stop()
        except Exception:  # pragma: no cover — defensive
            logger.exception("cleanup failure for container %s", self.name)

    async def _start(self) -> None:
        """Create + start container with the ``tail -f /dev/null`` keep-alive.

        Idempotent: a second call when ``_container`` is already running
        is a no-op. Raises :class:`InstanceContainerError` on docker errors.
        """
        if self._container is not None:
            return

        def _create() -> Any:
            # Remove any stale container from a crashed previous run.
            try:
                old = self._client.containers.get(self.name)
                try:
                    old.kill()
                except Exception:
                    pass
                old.remove(force=True)
            except Exception:
                pass
            return self._client.containers.create(
                image=self.image_tag,
                command=["tail", "-f", "/dev/null"],
                name=self.name,
                working_dir=str(self.workdir),
                detach=True,
                tty=False,
                auto_remove=False,
            )

        try:
            self._container = await asyncio.to_thread(_create)
            await asyncio.to_thread(self._container.start)
        except Exception as e:
            raise InstanceContainerError(
                f"failed to start container {self.name}: {e}"
            ) from e
        logger.info(
            "InstanceContainer started: %s (image=%s)",
            self.name, self.image_tag,
        )

    async def _stop(self) -> None:
        """Kill + force-remove the container. Idempotent; safe on failure."""
        if self._container is None:
            return

        def _kill_and_remove() -> None:
            try:
                self._container.kill()
            except Exception:
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                pass

        await asyncio.to_thread(_kill_and_remove)
        logger.info("InstanceContainer stopped: %s", self.name)
        self._container = None

    # ---- exec --------------------------------------------------------------

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout: float = 60.0,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Run a command inside the container.

        Returns :class:`ExecResult`; never raises on non-zero exit — agent
        tools need ``exit_code`` as data. Raises
        :class:`InstanceContainerError` only for docker-level failures.

        ``cmd`` can be a string (wrapped in ``bash -lc``) or a list. Strings
        are the common case for agent bash; lists bypass shell quoting and
        are what the git helpers use.
        """
        if self._container is None:
            raise InstanceContainerError("container not started")

        workdir = cwd or self.workdir
        cmd_arg: list[str] | str
        if isinstance(cmd, str):
            cmd_arg = ["bash", "-lc", cmd]
        else:
            cmd_arg = list(cmd)

        def _run() -> tuple[bytes, bytes, int, bool]:
            exec_id = self._client.api.exec_create(
                self._container.id,
                cmd=cmd_arg,
                workdir=workdir,
                stdout=True,
                stderr=True,
                tty=False,
                environment=env or None,
            )["Id"]
            stdout_chunks: list[bytes] = []
            stderr_chunks: list[bytes] = []
            stream = self._client.api.exec_start(
                exec_id, stream=True, demux=True,
            )
            for out, err in stream:
                if out:
                    stdout_chunks.append(out)
                if err:
                    stderr_chunks.append(err)
            info = self._client.api.exec_inspect(exec_id)
            exit_code = info.get("ExitCode")
            if exit_code is None:
                # Rare: inspect raced ahead of drain. Treat as zero so we
                # don't inject a fake error signal; the stream was drained.
                exit_code = 0
            return (
                b"".join(stdout_chunks),
                b"".join(stderr_chunks),
                int(exit_code),
                False,
            )

        try:
            out, err, code, timed_out = await asyncio.wait_for(
                asyncio.to_thread(_run), timeout=timeout
            )
        except asyncio.TimeoutError:
            return ExecResult(
                stdout="",
                stderr=f"command timed out after {timeout:.0f}s",
                exit_code=124,
                timed_out=True,
            )
        except Exception as e:
            raise InstanceContainerError(
                f"exec failed for {cmd!r}: {e}"
            ) from e

        return ExecResult(
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            exit_code=code,
            timed_out=timed_out,
        )

    # ---- file I/O ----------------------------------------------------------

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        """Read a text file from the container. Raises if not found.

        Truncates to ``max_bytes`` with a trailing marker; caller handles
        the full-file case by reading in chunks if needed.
        """
        r = await self.exec(
            f"cat -- {shlex.quote(path)}",
            timeout=30.0,
        )
        if r.exit_code != 0:
            raise InstanceContainerError(
                f"read_file {path!r} failed: {r.stderr.strip() or 'exit=' + str(r.exit_code)}"
            )
        if len(r.stdout.encode("utf-8", errors="replace")) > max_bytes:
            return r.stdout[:max_bytes] + f"\n[truncated at {max_bytes} bytes]"
        return r.stdout

    async def write_file(self, path: str, content: str) -> None:
        """Write text to a file inside the container via ``put_archive``.

        Binary-safe (unlike heredoc); handles arbitrary content with
        newlines, quotes, backticks. Path must be absolute.
        """
        if not path.startswith("/"):
            raise InstanceContainerError(
                f"write_file requires absolute path, got {path!r}"
            )
        if self._container is None:
            raise InstanceContainerError("container not started")

        data = content.encode("utf-8")
        parent, _, name = path.rpartition("/")
        parent = parent or "/"

        # Ensure parent exists.
        await self.exec(f"mkdir -p {shlex.quote(parent)}", timeout=10.0)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)

        def _put() -> bool:
            return bool(self._container.put_archive(parent, buf.getvalue()))

        ok = await asyncio.to_thread(_put)
        if not ok:
            raise InstanceContainerError(
                f"put_archive refused write to {path!r}"
            )

    # ---- git ---------------------------------------------------------------

    async def git_init_checkpoint(self) -> str:
        """Record base commit; returns the SHA for later diff/reset.

        Callers typically store the returned SHA and pass it back into
        :meth:`git_diff` at session end to get the full agent-produced
        diff relative to the container's starting state.
        """
        r = await self.exec(["git", "rev-parse", "HEAD"], timeout=15.0)
        if r.exit_code != 0:
            raise InstanceContainerError(
                f"git rev-parse failed: {r.stderr.strip()}"
            )
        return r.stdout.strip()

    async def git_commit(self, message: str = "pare: agent step") -> str:
        """Stage all changes and commit; returns new SHA.

        Uses ``--allow-empty`` so a no-op step doesn't fail the loop.
        Identity is hard-coded to ``pare@local`` to avoid depending on
        git config inside the image.
        """
        add = await self.exec(["git", "add", "-A"], timeout=30.0)
        if add.exit_code != 0:
            raise InstanceContainerError(
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
            raise InstanceContainerError(
                f"git commit failed: {commit.stderr.strip() or commit.stdout.strip()}"
            )
        head = await self.exec(["git", "rev-parse", "HEAD"], timeout=15.0)
        return head.stdout.strip()

    async def git_diff(self, base: Optional[str] = None) -> str:
        """Return unified diff from ``base`` (or working tree) to HEAD.

        When ``base`` is None, returns the working-tree-vs-HEAD diff
        (uncommitted changes). When ``base`` is a SHA, returns
        ``git diff <base>..HEAD`` — i.e. everything committed since the
        checkpoint, which is what the loop's final_diff needs.
        """
        if base is None:
            cmd = ["git", "diff"]
        else:
            cmd = ["git", "diff", f"{base}..HEAD"]
        r = await self.exec(cmd, timeout=30.0)
        if r.exit_code != 0:
            raise InstanceContainerError(
                f"git diff failed: {r.stderr.strip()}"
            )
        return r.stdout

    async def git_checkout(self, ref: str) -> None:
        """Discard working-tree changes and check out ``ref``."""
        r = await self.exec(
            f"git checkout -- . && git checkout {shlex.quote(ref)}",
            timeout=30.0,
        )
        if r.exit_code != 0:
            raise InstanceContainerError(
                f"git checkout failed: {r.stderr.strip()}"
            )


def _safe_container_name(instance_id: str) -> str:
    """Docker names disallow ``/`` and some punctuation."""
    return f"pare-{instance_id.replace('/', '_')}"
