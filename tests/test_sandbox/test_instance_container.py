"""Integration tests for :class:`InstanceContainer`.

These tests require:
- Docker daemon reachable (``docker.from_env().ping()``),
- A pulled (or pullable) swebench instance image,
- Environment variable ``PARE_RUN_DOCKER_TESTS=1`` to opt in.

Without the opt-in flag the whole module skips — keeping
``pytest -q`` fast on dev laptops where Docker may be off.

Default instance is ``sympy__sympy-11618`` (small test suite, base image
pulls in ~2 min cold). Override via ``PARE_SPIKE_INSTANCE``.

The 9 cases listed in the refactor plan (§验证):

1. ``test_start_stop_idempotent``
2. ``test_exec_captures_stdout_stderr``
3. ``test_exec_nonzero_exit``
4. ``test_exec_timeout``
5. ``test_read_write_roundtrip``
6. ``test_write_creates_parents``
7. ``test_git_diff_empty_when_no_change``
8. ``test_git_diff_after_write``
9. ``test_context_manager_cleanup_on_exception``
"""

from __future__ import annotations

import os
import uuid

import pytest

from pare.sandbox.instance_container import (
    ExecResult,
    InstanceContainer,
    InstanceContainerError,
)


_RUN_FLAG = os.environ.get("PARE_RUN_DOCKER_TESTS") == "1"
_INSTANCE_ID = os.environ.get("PARE_SPIKE_INSTANCE", "sympy__sympy-11618")

pytestmark = [
    pytest.mark.skipif(not _RUN_FLAG, reason="PARE_RUN_DOCKER_TESTS != 1"),
    pytest.mark.asyncio,
]


def _unique_name() -> str:
    return f"pare-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def container():
    """Yield a started container, force-remove on teardown."""
    c = await InstanceContainer.build(_INSTANCE_ID)
    # Override name so parallel test runs don't collide.
    c.name = _unique_name()
    async with c:
        yield c


async def test_start_stop_idempotent() -> None:
    c = await InstanceContainer.build(_INSTANCE_ID)
    c.name = _unique_name()
    try:
        await c._start()
        await c._start()  # second call is a no-op
        assert c._container is not None
    finally:
        await c._stop()
        await c._stop()  # idempotent cleanup
        assert c._container is None


async def test_exec_captures_stdout_stderr(container: InstanceContainer) -> None:
    r = await container.exec("echo hi; echo oops >&2")
    assert isinstance(r, ExecResult)
    assert r.exit_code == 0
    assert "hi" in r.stdout
    assert "oops" in r.stderr


async def test_exec_nonzero_exit(container: InstanceContainer) -> None:
    r = await container.exec(["false"])
    assert r.exit_code == 1
    assert r.timed_out is False


async def test_exec_timeout(container: InstanceContainer) -> None:
    r = await container.exec(["sleep", "10"], timeout=1.0)
    assert r.timed_out is True
    assert r.exit_code == 124


async def test_read_write_roundtrip(container: InstanceContainer) -> None:
    payload = "line1\n'quoted'\n\"double\"\n`tick`\n"
    path = "/tmp/pare_probe.txt"
    await container.write_file(path, payload)
    got = await container.read_file(path)
    assert got == payload


async def test_write_creates_parents(container: InstanceContainer) -> None:
    path = "/tmp/pare_deep/a/b/c/file.txt"
    await container.write_file(path, "ok\n")
    got = await container.read_file(path)
    assert got == "ok\n"


async def test_git_diff_empty_when_no_change(container: InstanceContainer) -> None:
    # /testbed is a git repo; after a fresh start the working tree matches HEAD.
    diff = await container.git_diff()
    assert diff == ""


async def test_git_diff_after_write(container: InstanceContainer) -> None:
    # Pick any Python file in /testbed and append a comment.
    r = await container.exec(
        "ls /testbed | head -20 && find /testbed -maxdepth 2 -name '*.py' | head -1",
    )
    # Find the first .py file from the find output.
    candidates = [
        line for line in r.stdout.splitlines()
        if line.startswith("/testbed/") and line.endswith(".py")
    ]
    assert candidates, f"no .py found in /testbed: {r.stdout!r}"
    target = candidates[0]
    original = await container.read_file(target)
    await container.write_file(target, original + "\n# pare probe\n")
    diff = await container.git_diff()
    assert target.replace("/testbed/", "a/") in diff or target in diff
    assert "pare probe" in diff


async def test_context_manager_cleanup_on_exception() -> None:
    c = await InstanceContainer.build(_INSTANCE_ID)
    c.name = _unique_name()
    with pytest.raises(RuntimeError, match="boom"):
        async with c:
            raise RuntimeError("boom")
    # After the raise, container must be removed.
    assert c._container is None
