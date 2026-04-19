"""R0 docker-py spike — verify three easy-to-get-wrong bits before R1.

Run on Linux with Docker daemon reachable::

    python -m scripts.spike_container sympy__sympy-11618

Or supply any swebench image tag directly::

    PARE_SPIKE_IMAGE=swebench/sweb.eval.x86_64.sympy__sympy-11618:latest \\
        python -m scripts.spike_container

Validates:

1. ``exec_start(stream=True, demux=True)`` + outer ``asyncio.wait_for`` really
   does interrupt a long command. docker-py's stream iterator does not
   honour socket-level timeouts, so the outer ``wait_for`` is load-bearing.
2. ``exec_inspect`` returns a sensible ``ExitCode`` only after the stream
   has been fully drained. Test by streaming a short command and calling
   ``exec_inspect`` immediately after the loop exits.
3. ``put_archive`` writes a file with predictable mode. We set
   ``TarInfo.mode = 0o644`` explicitly and read it back with ``ls -l``.

Prints a compact PASS/FAIL report; non-zero exit on any failure.

This is optional scaffolding — R1's ``InstanceContainer`` implementation
bakes in the workarounds regardless. Run this only if you want to
independently verify docker-py behaviour on your host before diving
into R1.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _resolve_image(arg: str | None) -> str:
    if arg and arg.startswith("swebench/"):
        return arg
    if arg:
        return f"swebench/sweb.eval.x86_64.{arg}:latest"
    env = os.environ.get("PARE_SPIKE_IMAGE")
    if env:
        return env
    # Default: the sympy instance referenced throughout the refactor plan.
    return "swebench/sweb.eval.x86_64.sympy__sympy-11618:latest"


async def _drain_stream_async(stream_iter: Any) -> tuple[bytes, bytes]:
    """Iterate a demuxed stream on a worker thread, returning (stdout, stderr)."""

    def _drain() -> tuple[bytes, bytes]:
        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        for out, err in stream_iter:
            if out:
                out_chunks.append(out)
            if err:
                err_chunks.append(err)
        return b"".join(out_chunks), b"".join(err_chunks)

    return await asyncio.to_thread(_drain)


async def _check_timeout(client: Any, container: Any) -> CheckResult:
    """Start `sleep 30`, cancel after 2 s via outer wait_for."""
    exec_id = client.api.exec_create(
        container.id, cmd=["sleep", "30"], stdout=True, stderr=True, tty=False
    )["Id"]
    stream = client.api.exec_start(exec_id, stream=True, demux=True)
    t0 = time.monotonic()
    timed_out = False
    try:
        await asyncio.wait_for(_drain_stream_async(stream), timeout=2.0)
    except asyncio.TimeoutError:
        timed_out = True
    elapsed = time.monotonic() - t0

    if not timed_out:
        return CheckResult("timeout_interrupts_long_exec", False,
                           f"wait_for did not fire; slept {elapsed:.1f}s")
    if elapsed > 5.0:
        return CheckResult("timeout_interrupts_long_exec", False,
                           f"wait_for took {elapsed:.1f}s (expected ~2s)")
    return CheckResult("timeout_interrupts_long_exec", True,
                       f"cancelled at {elapsed:.2f}s (expected ~2s)")


async def _check_exec_inspect_after_drain(client: Any, container: Any) -> CheckResult:
    """Stream `false`, drain, then exec_inspect — ExitCode should be 1."""
    exec_id = client.api.exec_create(
        container.id, cmd=["sh", "-c", "false"],
        stdout=True, stderr=True, tty=False,
    )["Id"]
    stream = client.api.exec_start(exec_id, stream=True, demux=True)
    await _drain_stream_async(stream)
    info = client.api.exec_inspect(exec_id)
    code = info.get("ExitCode")
    if code != 1:
        return CheckResult(
            "exec_inspect_after_drain", False,
            f"ExitCode={code!r} after draining `false` (expected 1)"
        )
    return CheckResult("exec_inspect_after_drain", True,
                       "ExitCode=1 as expected after stream drain")


async def _check_put_archive_mode(client: Any, container: Any) -> CheckResult:
    """put_archive a file with mode 0o644 and verify via ls -l."""
    data = b"hello pare\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="spike_probe.txt")
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    ok = await asyncio.to_thread(
        container.put_archive, "/tmp", buf.getvalue()
    )
    if not ok:
        return CheckResult("put_archive_mode_preserved", False, "put_archive returned False")

    exec_id = client.api.exec_create(
        container.id,
        cmd=["ls", "-l", "/tmp/spike_probe.txt"],
        stdout=True, stderr=True, tty=False,
    )["Id"]
    stream = client.api.exec_start(exec_id, stream=True, demux=True)
    out, err = await _drain_stream_async(stream)
    line = out.decode("utf-8", errors="replace").strip()
    # Expect e.g. "-rw-r--r-- 1 root root 11 ..."
    if not line.startswith("-rw-r--r--"):
        return CheckResult(
            "put_archive_mode_preserved", False,
            f"ls -l reported {line!r} — expected leading '-rw-r--r--'"
        )
    return CheckResult("put_archive_mode_preserved", True,
                       "TarInfo.mode=0o644 survived put_archive")


async def main(argv: list[str]) -> int:
    try:
        import docker  # type: ignore
    except ImportError:
        print("FAIL: docker-py not installed (pip install docker)", file=sys.stderr)
        return 2

    image = _resolve_image(argv[1] if len(argv) > 1 else None)
    print(f"[spike] image = {image}")

    client = docker.from_env()
    try:
        client.ping()
    except Exception as e:
        print(f"FAIL: docker daemon unreachable: {e}", file=sys.stderr)
        return 2

    # Pull on demand — the first pull can be slow (2-5 min for sympy);
    # subsequent runs hit cache.
    try:
        client.images.get(image)
    except Exception:
        print(f"[spike] pulling {image} (first time — may take minutes)…")
        client.images.pull(image)

    container = client.containers.create(
        image=image,
        command=["tail", "-f", "/dev/null"],
        name=f"pare-spike-{os.getpid()}",
        working_dir="/testbed",
        detach=True,
        tty=False,
        auto_remove=False,
    )
    container.start()
    print(f"[spike] container up: {container.short_id}")

    checks: list[CheckResult] = []
    try:
        checks.append(await _check_timeout(client, container))
        checks.append(await _check_exec_inspect_after_drain(client, container))
        checks.append(await _check_put_archive_mode(client, container))
    finally:
        try:
            container.kill()
        except Exception:
            pass
        try:
            container.remove(force=True)
        except Exception:
            pass
        client.close()

    print()
    all_pass = True
    for c in checks:
        flag = "PASS" if c.passed else "FAIL"
        all_pass &= c.passed
        print(f"  [{flag}] {c.name:<38} {c.detail}")
    print()
    if all_pass:
        print("spike: all checks green — R1 can proceed")
        return 0
    print("spike: at least one check failed — see above", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
