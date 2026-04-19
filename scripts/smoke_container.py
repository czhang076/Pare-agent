"""R1 smoke: start an InstanceContainer and run a quick sanity loop.

Run on Linux with Docker daemon reachable::

    python -m scripts.smoke_container sympy__sympy-11618

Default instance is ``sympy__sympy-11618`` if none given. Expected
duration once the derived image is built: ~5-10 s. Cold (base pull +
derived build): 3-6 min.

The script:

1. Builds / resolves the derived ``pare-eval.<iid>:latest`` image.
2. Starts the container.
3. Runs ``python -c "import sympy; print(sympy.__version__)"``.
4. Writes a file, reads it back, asserts equality.
5. Records the base commit, appends a line to a .py file, commits,
   and prints the resulting diff.
6. Stops + removes the container.

Non-zero exit on any failure.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from pare.sandbox.instance_container import InstanceContainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def main(argv: list[str]) -> int:
    instance_id = argv[1] if len(argv) > 1 else "sympy__sympy-11618"
    print(f"[smoke] instance = {instance_id}")

    async with await InstanceContainer.build(instance_id) as c:
        print(f"[smoke] image = {c.image_tag}")

        r = await c.exec("python -c 'import sympy; print(sympy.__version__)'")
        print(f"[smoke] sympy version: {r.stdout.strip()} (exit={r.exit_code})")
        assert r.exit_code == 0, r.stderr

        path = "/tmp/pare_smoke.txt"
        payload = "hello from pare smoke\n"
        await c.write_file(path, payload)
        got = await c.read_file(path)
        assert got == payload, f"roundtrip mismatch: {got!r} != {payload!r}"
        print(f"[smoke] file roundtrip ok ({len(payload)} bytes)")

        base = await c.git_init_checkpoint()
        print(f"[smoke] base commit = {base[:12]}")

        ls = await c.exec(
            "find /testbed -maxdepth 2 -name '*.py' | head -1"
        )
        target = ls.stdout.strip().splitlines()[0]
        original = await c.read_file(target)
        await c.write_file(target, original + "\n# pare smoke probe\n")
        new_head = await c.git_commit("pare smoke: probe")
        print(f"[smoke] new HEAD = {new_head[:12]}")

        diff = await c.git_diff(base=base)
        print(f"[smoke] diff length = {len(diff)} chars")
        assert "pare smoke probe" in diff, "expected probe line in diff"

        print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
