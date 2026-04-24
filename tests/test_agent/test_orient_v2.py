"""Unit tests for :mod:`pare.agent.orient_v2`.

The repo-map pre-pass is zero-LLM, so tests mock an
:class:`InstanceContainer`-shaped object and assert on the rendered
markdown. Covers:

- README section: picks the first candidate that exists, truncates to
  ``readme_max_chars``.
- Top-listing section: ``ls`` output rendered as a bullet list, with a
  tail note when over the cap.
- Repo-map section: ``git ls-files`` → rank → signature extraction.
- Fail-open: any internal failure yields ``""``.
- Rank heuristic: shallow non-test files beat deep test files of the
  same size.
- ``_extract_signatures``: skips files with SyntaxError instead of
  exploding the pre-pass.
- ``format_orient_for_system_prompt``: no-op on empty blob.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pytest

from pare.agent.orient_v2 import (
    _extract_signatures,
    _rank_files,
    _render_readme,
    _render_repo_map,
    _render_top_listing,
    format_orient_for_system_prompt,
    run_orient,
)
from pare.sandbox.instance_container import ExecResult


# ---------------------------------------------------------------------------
# Shared fake
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FakeContainer:
    """Minimal stand-in for :class:`InstanceContainer`.

    - ``files`` keyed by absolute posix path.
    - ``exec_handler(cmd) -> ExecResult`` lets each test customise output.
    """

    workdir: str = "/testbed"
    instance_id: str = "fake"
    files: dict[str, str] = field(default_factory=dict)
    exec_handler: Callable[[object], ExecResult] = field(
        default_factory=lambda: (lambda cmd: ExecResult(
            stdout="", stderr="", exit_code=0, timed_out=False,
        ))
    )
    exec_log: list[object] = field(default_factory=list)

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        if path not in self.files:
            raise RuntimeError(f"not found: {path}")
        return self.files[path]

    async def exec(self, cmd, *, timeout=60.0, cwd=None, env=None):
        self.exec_log.append(cmd)
        return self.exec_handler(cmd)


def _ok(stdout: str, stderr: str = "") -> ExecResult:
    return ExecResult(stdout=stdout, stderr=stderr, exit_code=0, timed_out=False)


# ---------------------------------------------------------------------------
# Section 1: README
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_readme_picks_first_existing_candidate() -> None:
    fake = FakeContainer(
        files={"/testbed/README.md": "# Sympy\n\nSymbolic mathematics."},
    )
    out = await _render_readme(fake, max_chars=100)
    assert "Sympy" in out
    assert "README (README.md)" in out


@pytest.mark.asyncio
async def test_render_readme_falls_back_to_rst_when_md_missing() -> None:
    fake = FakeContainer(files={"/testbed/README.rst": "Heading\n=======\n"})
    out = await _render_readme(fake, max_chars=100)
    assert "README (README.rst)" in out


@pytest.mark.asyncio
async def test_render_readme_truncates_with_marker() -> None:
    fake = FakeContainer(
        files={"/testbed/README.md": "x" * 5000},
    )
    out = await _render_readme(fake, max_chars=100)
    assert "truncated at 100 chars" in out
    # 100-char body + separator + marker
    assert out.count("x") == 100


@pytest.mark.asyncio
async def test_render_readme_returns_empty_when_no_readme() -> None:
    fake = FakeContainer()  # no files
    assert await _render_readme(fake, max_chars=100) == ""


# ---------------------------------------------------------------------------
# Section 2: top-level listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_top_listing_bulletises_ls_output() -> None:
    listing = "sympy\ndocs\nsetup.py\nREADME.md\n"
    fake = FakeContainer(exec_handler=lambda cmd: _ok(listing))
    out = await _render_top_listing(fake, max_entries=10)
    assert out.startswith("### Top-level listing")
    for name in ("sympy", "docs", "setup.py", "README.md"):
        assert f"- {name}" in out


@pytest.mark.asyncio
async def test_render_top_listing_truncates_past_max_entries() -> None:
    listing = "\n".join(f"entry_{i}" for i in range(100)) + "\n"
    fake = FakeContainer(exec_handler=lambda cmd: _ok(listing))
    out = await _render_top_listing(fake, max_entries=5)
    assert "[... 95 more entries]" in out


@pytest.mark.asyncio
async def test_render_top_listing_empty_on_exec_failure() -> None:
    fake = FakeContainer(
        exec_handler=lambda cmd: ExecResult(
            stdout="", stderr="err", exit_code=1, timed_out=False,
        )
    )
    assert await _render_top_listing(fake, max_entries=10) == ""


# ---------------------------------------------------------------------------
# Section 3: repo map
# ---------------------------------------------------------------------------


def _pick_exec(
    ls_files_stdout: str = "",
    wc_stdout: str = "",
) -> Callable[[object], ExecResult]:
    """Route ``git ls-files`` vs ``wc -l`` to separate stdouts."""

    def handler(cmd) -> ExecResult:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "ls-files" in cmd_str:
            return _ok(ls_files_stdout)
        if cmd_str.startswith("wc -l"):
            return _ok(wc_stdout)
        return _ok("")

    return handler


@pytest.mark.asyncio
async def test_render_repo_map_emits_ranked_signatures() -> None:
    # git ls-files prints NUL-terminated
    ls_out = "sympy/core.py\x00tests/test_core.py\x00"
    wc_out = (
        "  120 sympy/core.py\n"
        "   50 tests/test_core.py\n"
        "  170 total\n"
    )
    fake = FakeContainer(
        files={
            "/testbed/sympy/core.py": (
                '"""core."""\n'
                "class Expr:\n"
                "    def diff(self, x):\n"
                "        return 0\n"
                "def solve(eq, x):\n"
                "    return []\n"
            ),
            "/testbed/tests/test_core.py": (
                "def test_diff():\n"
                "    assert True\n"
            ),
        },
        exec_handler=_pick_exec(ls_out, wc_out),
    )
    out = await _render_repo_map(fake, top_k=10, max_sigs=20)
    assert out.startswith("### Repo map")
    # Non-test file ranks first.
    assert out.index("sympy/core.py") < out.index("tests/test_core.py")
    # Class signature and method signature rendered.
    assert "class Expr:" in out
    assert "def diff(self, x)" in out
    assert "def solve(eq, x)" in out


@pytest.mark.asyncio
async def test_render_repo_map_empty_when_git_fails() -> None:
    fake = FakeContainer(
        exec_handler=lambda cmd: ExecResult(
            stdout="", stderr="fatal", exit_code=128, timed_out=False,
        )
    )
    assert await _render_repo_map(fake, top_k=10, max_sigs=20) == ""


def test_rank_files_boosts_shallow_non_test() -> None:
    ranked = _rank_files([
        ("sympy/core.py", 100),
        ("tests/test_deep/a/b/c/test_x.py", 100),
        ("docs/example.py", 100),
        ("src/stub.py", 1),           # filtered out — trivially small
    ])
    paths = [fr.path for fr in ranked]
    assert paths[0] == "sympy/core.py"
    # The stub file is dropped entirely.
    assert "src/stub.py" not in paths


@pytest.mark.asyncio
async def test_extract_signatures_skips_file_with_syntax_error() -> None:
    fake = FakeContainer(
        files={"/testbed/broken.py": "def oops(:\n    pass\n"},
    )
    sigs = await _extract_signatures(fake, "broken.py", max_sigs=10)
    assert sigs == []


@pytest.mark.asyncio
async def test_extract_signatures_emits_class_and_def_headers() -> None:
    src = (
        "import os\n"
        "class A:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    async def fetch(self, url):\n"
        "        return None\n"
        "def top_level():\n"
        "    pass\n"
    )
    fake = FakeContainer(files={"/testbed/m.py": src})
    sigs = await _extract_signatures(fake, "m.py", max_sigs=10)
    assert "class A:" in sigs
    assert any("def __init__(self)" in s for s in sigs)
    assert any("async def fetch(self, url)" in s for s in sigs)
    assert "def top_level()" in sigs


# ---------------------------------------------------------------------------
# run_orient (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_orient_combines_sections() -> None:
    fake = FakeContainer(
        files={
            "/testbed/README.md": "# Proj\n\nHello world.",
            "/testbed/core.py": "def main():\n    return 1\n",
        },
        exec_handler=_pick_exec(
            ls_files_stdout="core.py\x00",
            wc_stdout="  2 core.py\n  2 total\n",
        ),
    )
    # The top-listing exec is the same "ls -1" command; route it too.
    # Simplest: override exec_handler to include all three.
    original = fake.exec_handler

    def combined(cmd):
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "ls -1" in cmd_str and "ls-files" not in cmd_str:
            return _ok("README.md\ncore.py\n")
        return original(cmd)

    fake.exec_handler = combined
    out = await run_orient(fake, readme_max_chars=50, top_list_max=5)
    # All three sections present.
    assert "### README" in out
    assert "### Top-level listing" in out
    assert "### Repo map" in out


@pytest.mark.asyncio
async def test_run_orient_fail_open_on_read_failure() -> None:
    """An exception inside a section must not crash the whole pre-pass."""

    class ExplodingContainer(FakeContainer):
        async def read_file(self, path, *, max_bytes=1_000_000):
            raise RuntimeError("disk on fire")

        async def exec(self, cmd, *, timeout=60.0, cwd=None, env=None):
            raise RuntimeError("docker on fire")

    fake = ExplodingContainer()
    # run_orient catches at top level and returns "".
    assert await run_orient(fake) == ""


@pytest.mark.asyncio
async def test_run_orient_can_skip_repo_map() -> None:
    fake = FakeContainer(
        files={"/testbed/README.md": "hi"},
        exec_handler=lambda cmd: _ok("README.md\n") if "ls -1" in (
            cmd if isinstance(cmd, str) else " ".join(cmd)
        ) else _ok(""),
    )
    out = await run_orient(fake, use_repo_map=False)
    assert "### Repo map" not in out
    assert "### README" in out


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def test_format_orient_for_empty_returns_empty() -> None:
    assert format_orient_for_system_prompt("") == ""
    assert format_orient_for_system_prompt("  \n  ") == ""


def test_format_orient_wraps_section_header() -> None:
    out = format_orient_for_system_prompt("### README\n\nhi")
    assert out.startswith("\n\n## Repository Context")
    assert "### README" in out
