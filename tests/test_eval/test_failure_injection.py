"""Tests for ``pare.eval.failure_injection``.

Lock the scaffold before we start calling it from a CLI + an agent
runner. The invariants we care about:

1. ``apply → revert`` is byte-perfect for every registered fault on a
   fresh workdir — if the revert leaks state into the next trial, the
   per-fault comparisons in the headline table are contaminated.
2. ``run_with_fault`` reverts **even when the agent raises**. Leaving
   a faulted file in a shared git worktree is the worst possible
   outcome; the revert-always contract is what makes this module safe
   to batch over many tasks.
3. Registry is unique-by-name; re-registering the same name at import
   time would be a silent fault-overwrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pare.eval.failure_injection import (
    FaultInjectionResult,
    InjectedFault,
    REGISTRY,
    _register,
    apply_fault,
    revert_fault,
    run_with_fault,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _snapshot_tree(root: Path) -> dict[str, str]:
    """Cheap recursive snapshot for byte-perfect revert assertions.

    Keys are paths relative to ``root``; values are file contents.
    Directories are recorded as ``<DIR>``. We intentionally ignore
    mtime / permissions — tests only care that text content round-trips.
    """
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root)).replace("\\", "/")
        if p.is_dir():
            out[rel] = "<DIR>"
        else:
            try:
                out[rel] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                out[rel] = f"<BINARY:{p.stat().st_size}>"
    return out


def _make_minimal_repo(workdir: Path) -> None:
    """A workdir that looks enough like a real Python repo for faults to
    find targets. Faults that scan for ``*.py`` need a non-test file."""
    (workdir / "pkg").mkdir()
    (workdir / "pkg" / "__init__.py").write_text('"""pkg."""\n', encoding="utf-8")
    (workdir / "pkg" / "core.py").write_text(
        '"""module core."""\n\ndef add(a, b):\n    return a + b\n',
        encoding="utf-8",
    )
    (workdir / "tests").mkdir()
    (workdir / "tests" / "test_core.py").write_text(
        "from pkg.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# REGISTRY invariants
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_known_faults_present(self):
        """The 3 scaffold faults we've written must be loaded.

        ``fake_test_success`` replaced ``stale_test_cache`` in v0.2.0
        because the cache fault was effectively a no-op against Pare's
        prompts (which don't use --last-failed)."""
        assert "fake_test_success" in REGISTRY
        assert "wrong_import" in REGISTRY
        assert "empty_baseline" in REGISTRY
        # Old name must NOT be re-registered — keeps a future maintainer
        # from accidentally re-introducing the design bug.
        assert "stale_test_cache" not in REGISTRY

    def test_register_rejects_duplicate_name(self):
        """A second registration under an existing name raises ValueError
        — this would silently overwrite a fault if we let it slip."""
        duplicate = InjectedFault(
            name="wrong_import",  # already registered
            description="collision",
            applies_to_liu="",
            apply_fn=lambda _p: {},
            revert_fn=lambda _p, _t: None,
        )
        with pytest.raises(ValueError, match="duplicate fault name"):
            _register(duplicate)

    def test_apply_unknown_fault_raises_keyerror(self, tmp_path: Path):
        with pytest.raises(KeyError, match="unknown fault"):
            apply_fault("does_not_exist", tmp_path)


# ---------------------------------------------------------------------------
# Per-fault apply/revert round-trips
# ---------------------------------------------------------------------------


class TestFakeTestSuccessFault:
    """v0.2.0 — direct C2 Premature Success probe.

    Rewrites every ``def test_*`` body to ``pass``. The agent that
    trusts pytest's green exit code without re-reading the test
    bodies will declare victory on an unfixed bug — exactly the
    failure mode classifier_liu.C2 should catch."""

    def test_apply_rewrites_test_function_bodies_to_pass(
        self, tmp_path: Path
    ):
        """Real assertions become ``pass``; the original file content is
        stashed in the token for revert."""
        _make_minimal_repo(tmp_path)
        test_file = tmp_path / "tests" / "test_core.py"
        original = test_file.read_text(encoding="utf-8")
        # Sanity-check the fixture: original test has a real assertion.
        assert "assert add(1, 2) == 3" in original

        token = apply_fault("fake_test_success", tmp_path)

        post = test_file.read_text(encoding="utf-8")
        # Assertion is gone — body is just `pass`.
        assert "assert" not in post
        assert "def test_add" in post
        # Module-level imports are preserved by ast round-trip.
        assert "from pkg.core import add" in post
        # Token records the pre-fault content for revert.
        assert str(test_file) in token["backups"]
        assert token["backups"][str(test_file)] == original

    def test_revert_restores_byte_identical_content(self, tmp_path: Path):
        _make_minimal_repo(tmp_path)
        before = _snapshot_tree(tmp_path)

        token = apply_fault("fake_test_success", tmp_path)
        # Confirm something actually changed.
        assert _snapshot_tree(tmp_path) != before

        revert_fault("fake_test_success", tmp_path, token)
        assert _snapshot_tree(tmp_path) == before

    def test_raises_when_no_test_file_available(self, tmp_path: Path):
        """A workdir with zero test files has no C2 target — loud
        error, not silent no-op that masquerades as empty_baseline."""
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")

        with pytest.raises(RuntimeError, match="no test_.*\\.py"):
            apply_fault("fake_test_success", tmp_path)

    def test_handles_multiple_test_files(self, tmp_path: Path):
        """Both test_X.py and Y_test.py patterns are rewritten."""
        _make_minimal_repo(tmp_path)
        # Add a second test file with the *_test.py suffix variant.
        extra = tmp_path / "tests" / "helper_test.py"
        extra.write_text(
            "def test_helper():\n    assert 1 == 1\n", encoding="utf-8"
        )

        token = apply_fault("fake_test_success", tmp_path)
        assert len(token["backups"]) == 2

        helper_post = extra.read_text(encoding="utf-8")
        assert "assert 1 == 1" not in helper_post
        # Revert restores both.
        revert_fault("fake_test_success", tmp_path, token)
        assert extra.read_text(encoding="utf-8") == (
            "def test_helper():\n    assert 1 == 1\n"
        )

    def test_skips_files_under_dot_git(self, tmp_path: Path):
        """A .git/hooks/test_pre_commit.py would be a horrible target —
        the filter excludes anything under .git/ exactly so we don't
        end up rewriting git's internal scripts."""
        _make_minimal_repo(tmp_path)
        (tmp_path / ".git").mkdir(exist_ok=True)
        (tmp_path / ".git" / "hooks").mkdir(exist_ok=True)
        sneaky = tmp_path / ".git" / "hooks" / "test_pre_commit.py"
        sneaky.write_text(
            "def test_thing():\n    assert True\n", encoding="utf-8"
        )

        token = apply_fault("fake_test_success", tmp_path)
        # The .git-hosted file must NOT appear in the backup dict.
        assert all(".git" not in p for p in token["backups"])
        # Its content is preserved (apply did not touch it).
        assert sneaky.read_text(encoding="utf-8") == (
            "def test_thing():\n    assert True\n"
        )
        revert_fault("fake_test_success", tmp_path, token)


class TestWrongImportFault:
    def test_apply_prepends_bad_import_to_first_non_test_py(
        self, tmp_path: Path
    ):
        _make_minimal_repo(tmp_path)
        token = apply_fault("wrong_import", tmp_path)

        target = Path(token["target"])
        assert "/tests/" not in str(target).replace("\\", "/"), (
            "must not target a test file — would break test collection "
            "in ways that conflate with the real fault signal"
        )
        assert target.read_text(encoding="utf-8").startswith(
            "import _pare_synthetic_missing_module"
        )

    def test_revert_restores_original_content(self, tmp_path: Path):
        _make_minimal_repo(tmp_path)
        before = _snapshot_tree(tmp_path)

        token = apply_fault("wrong_import", tmp_path)
        revert_fault("wrong_import", tmp_path, token)
        assert _snapshot_tree(tmp_path) == before

    def test_raises_when_no_python_files_available(self, tmp_path: Path):
        """A workdir with only test files has nothing the fault can
        target — we want a loud error, not silent no-op success that
        later confuses the results table."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_only.py").write_text("pass\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="no non-test .py file"):
            apply_fault("wrong_import", tmp_path)

    def test_skips_conftest_setup_init_at_workdir_root(self, tmp_path: Path):
        """Repo-bootstrap files (``conftest.py``, ``setup.py``,
        ``__init__.py``) must not be selected as wrong_import targets.
        Poisoning them changes the failure mode from "ModuleNotFoundError
        at exec" to "pytest collection error" / "pip install break",
        which is not the B2.2 scenario the fault claims to probe.

        Real trigger: SWE-bench repos (django, sympy, astropy) all have
        a root-level ``conftest.py`` whose name sorts before most
        package files lexicographically, so the previous implementation
        would pick it first."""
        # Bootstrap files at root — all must be skipped.
        (tmp_path / "conftest.py").write_text(
            "import pytest\n", encoding="utf-8"
        )
        (tmp_path / "setup.py").write_text(
            "from setuptools import setup\nsetup()\n", encoding="utf-8"
        )
        (tmp_path / "__init__.py").write_text("", encoding="utf-8")
        # The one legitimate target lives in a package directory.
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "core.py").write_text(
            '"""module."""\n', encoding="utf-8"
        )

        token = apply_fault("wrong_import", tmp_path)

        target = Path(token["target"])
        assert target.name == "core.py", (
            f"expected pkg/core.py, got {target!r} — bootstrap files "
            "leaked through the filter"
        )
        revert_fault("wrong_import", tmp_path, token)


class TestEmptyBaselineFault:
    def test_apply_and_revert_are_no_ops(self, tmp_path: Path):
        """empty_baseline is the false-positive sanity arm — any
        observable change here means we broke it."""
        _make_minimal_repo(tmp_path)
        before = _snapshot_tree(tmp_path)

        token = apply_fault("empty_baseline", tmp_path)
        assert _snapshot_tree(tmp_path) == before

        revert_fault("empty_baseline", tmp_path, token)
        assert _snapshot_tree(tmp_path) == before


# ---------------------------------------------------------------------------
# run_with_fault orchestration
# ---------------------------------------------------------------------------


class TestRunWithFault:
    def test_happy_path_returns_result_and_reverts(self, tmp_path: Path):
        _make_minimal_repo(tmp_path)
        before = _snapshot_tree(tmp_path)

        def stub_runner(
            instance_id: str, workdir: Path
        ) -> tuple[int, dict[str, Any]]:
            # While the agent "runs", exactly one non-test .py file must
            # carry the injected import (which file depends on sort
            # order — we don't pin it here, only that the fault is live).
            poisoned = [
                p for p in workdir.rglob("*.py")
                if p.read_text(encoding="utf-8").startswith(
                    "import _pare_synthetic_missing_module"
                )
            ]
            assert len(poisoned) == 1, (
                f"expected 1 poisoned file mid-run, found {len(poisoned)}"
            )
            return 0, {"trajectory_id": "t_fake", "instance_id": instance_id}

        result = run_with_fault(
            fault_name="wrong_import",
            instance_id="swe-1",
            workdir=tmp_path,
            agent_runner=stub_runner,
        )

        assert isinstance(result, FaultInjectionResult)
        assert result.fault_name == "wrong_import"
        assert result.agent_exit_code == 0
        assert result.trajectory["trajectory_id"] == "t_fake"
        assert result.error == ""
        assert result.agent_duration_s >= 0
        # Critical: revert happened.
        assert _snapshot_tree(tmp_path) == before

    def test_reverts_even_when_agent_raises(self, tmp_path: Path):
        """The revert-always contract: if the agent blows up, we still
        restore the workdir. Leaving a faulted file in a shared git
        worktree is the worst outcome this module can produce."""
        _make_minimal_repo(tmp_path)
        before = _snapshot_tree(tmp_path)

        def exploding_runner(*_args, **_kwargs):
            raise RuntimeError("llm provider dropped the call")

        result = run_with_fault(
            fault_name="wrong_import",
            instance_id="swe-1",
            workdir=tmp_path,
            agent_runner=exploding_runner,
        )

        # Sentinel contract: None means "runner raised", distinct from
        # any real integer exit code (including -1 from subprocess).
        assert result.agent_exit_code is None
        assert "llm provider dropped" in result.error
        # Even after the agent raised, the workdir is clean.
        assert _snapshot_tree(tmp_path) == before

    def test_real_negative_one_exit_is_not_the_raise_sentinel(
        self, tmp_path: Path
    ):
        """A runner that legitimately returns -1 must produce
        ``agent_exit_code == -1`` (an int), NOT the ``None`` sentinel
        reserved for "runner raised". Distinguishes subprocess failure
        from callback failure."""
        _make_minimal_repo(tmp_path)

        def subprocess_style_runner(*_args, **_kwargs):
            # Real subprocess wrappers can legitimately surface -1.
            return -1, {"trajectory_id": "t_x"}

        result = run_with_fault(
            fault_name="wrong_import",
            instance_id="swe-sub",
            workdir=tmp_path,
            agent_runner=subprocess_style_runner,
        )

        assert result.agent_exit_code == -1
        assert result.agent_exit_code is not None
        assert result.error == ""

    def test_records_revert_failure_in_error_field(self, tmp_path: Path):
        """If the revert itself fails, both the agent error (if any)
        and the revert failure must surface in the result — silently
        swallowing the revert exception is how corrupted workdirs
        accumulate over a long batch run."""
        calls: dict[str, int] = {"revert": 0}

        def bad_revert(_workdir: Path, _token: Any) -> None:
            calls["revert"] += 1
            raise OSError("disk full")

        fault = InjectedFault(
            name="_test_bad_revert",
            description="test-only fault whose revert raises",
            applies_to_liu="",
            apply_fn=lambda _p: {},
            revert_fn=bad_revert,
        )
        # Register manually for this test; unregister in finally.
        _register(fault)
        try:
            result = run_with_fault(
                fault_name="_test_bad_revert",
                instance_id="swe-x",
                workdir=tmp_path,
                agent_runner=lambda _i, _w: (0, {}),
            )
            assert "revert_failed" in result.error
            assert "disk full" in result.error
            assert calls["revert"] == 1
        finally:
            REGISTRY.pop("_test_bad_revert", None)

    def test_unknown_fault_raises_before_applying(self, tmp_path: Path):
        """Unknown fault name → KeyError before anything touches the
        workdir, so no cleanup is needed."""
        before = _snapshot_tree(tmp_path)
        with pytest.raises(KeyError):
            run_with_fault(
                fault_name="no_such_fault",
                instance_id="swe-1",
                workdir=tmp_path,
                agent_runner=lambda _i, _w: (0, {}),
            )
        assert _snapshot_tree(tmp_path) == before


# ---------------------------------------------------------------------------
# Container-mode fault apply/revert (mock InstanceContainer)
# ---------------------------------------------------------------------------


class _MockContainerExecResult:
    """Duck-types pare.sandbox.instance_container.ExecResult — just the
    fields container-mode fault code reads."""

    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = False


class _MockContainer:
    """In-memory container stand-in for unit-testing container-mode faults.

    Stores a virtual filesystem as ``{absolute_path: file_content}``.
    ``exec`` only implements ``find`` because that's all the fault
    apply code calls it for; everything else is read_file/write_file
    against the dict.

    Why not use a real ``InstanceContainer``: it needs Docker + a built
    SWE-bench image (~2 GB pull), defeats the purpose of fast unit
    tests. Container interactions in tests are all
    workdir-prefix / find-based; the mock captures that surface
    faithfully without the boot cost.
    """

    def __init__(self, workdir: str, files: dict[str, str] | None = None):
        self.workdir = workdir
        self.files: dict[str, str] = dict(files or {})
        self.exec_calls: list[str] = []

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout: float = 60.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _MockContainerExecResult:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.exec_calls.append(cmd_str)
        # Handle the one shape of find invocation the fault code uses:
        #   find <wd> -type f -name '<pattern>' [-o -name ...]
        if cmd_str.startswith("find "):
            patterns: list[str] = []
            tokens = cmd_str.split()
            i = 0
            while i < len(tokens):
                if tokens[i] == "-name" and i + 1 < len(tokens):
                    pat = tokens[i + 1].strip("'\"\\")
                    patterns.append(pat)
                    i += 2
                else:
                    i += 1
            import fnmatch
            lines = [
                p
                for p in self.files
                if any(
                    fnmatch.fnmatch(p.rsplit("/", 1)[-1], pat)
                    for pat in patterns
                )
            ]
            return _MockContainerExecResult(
                stdout="\n".join(sorted(lines)) + ("\n" if lines else "")
            )
        # Anything else: unsupported in this mock — fail loud so the
        # test surfaces the assumption violation.
        raise NotImplementedError(
            f"_MockContainer.exec doesn't handle: {cmd_str!r}"
        )

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:
        self.files[path] = content


def _run(coro):
    """Tiny asyncio.run wrapper — keeps tests readable."""
    import asyncio
    return asyncio.run(coro)


class TestContainerModeFaults:
    """Lock down the (apply -> agent sees fault -> revert -> byte-perfect)
    contract for the container-mode parallel of each fault. Uses a
    mock container so no Docker required."""

    def _make_repo(self, wd: str = "/testbed") -> _MockContainer:
        return _MockContainer(
            workdir=wd,
            files={
                f"{wd}/pkg/__init__.py": '"""pkg."""\n',
                f"{wd}/pkg/core.py": "def add(a, b):\n    return a + b\n",
                f"{wd}/tests/test_core.py": (
                    "from pkg.core import add\n\n"
                    "def test_add():\n    assert add(1, 2) == 3\n"
                ),
            },
        )

    def test_empty_baseline_in_container_is_noop(self):
        from pare.eval.failure_injection import (
            apply_fault_in_container, revert_fault_in_container,
        )
        c = self._make_repo()
        before = dict(c.files)
        token = _run(apply_fault_in_container("empty_baseline", c))
        assert token == {}
        assert c.files == before
        _run(revert_fault_in_container("empty_baseline", c, token))
        assert c.files == before

    def test_wrong_import_in_container_targets_non_test_py(self):
        from pare.eval.failure_injection import apply_fault_in_container
        c = self._make_repo()
        token = _run(apply_fault_in_container("wrong_import", c))
        # Target must be a non-test .py. pkg/__init__.py is filtered out
        # (bootstrap exclusion), so pkg/core.py is the only candidate.
        assert token["target"].endswith("/pkg/core.py")
        post = c.files[token["target"]]
        assert post.startswith("import _pare_synthetic_missing_module")
        # Token preserves original for revert.
        assert token["original"] == "def add(a, b):\n    return a + b\n"

    def test_wrong_import_in_container_revert_is_byte_identical(self):
        from pare.eval.failure_injection import (
            apply_fault_in_container, revert_fault_in_container,
        )
        c = self._make_repo()
        before = dict(c.files)
        token = _run(apply_fault_in_container("wrong_import", c))
        assert c.files != before  # confirm apply changed something
        _run(revert_fault_in_container("wrong_import", c, token))
        assert c.files == before

    def test_wrong_import_in_container_raises_when_no_target(self):
        """A container with only test files mirrors host-mode's loud
        failure mode."""
        from pare.eval.failure_injection import apply_fault_in_container
        c = _MockContainer(
            workdir="/testbed",
            files={
                "/testbed/tests/test_only.py": "def test_x(): pass\n",
            },
        )
        with pytest.raises(RuntimeError, match="no non-test"):
            _run(apply_fault_in_container("wrong_import", c))

    def test_fake_test_success_in_container_rewrites_test_bodies(self):
        from pare.eval.failure_injection import apply_fault_in_container
        c = self._make_repo()
        token = _run(apply_fault_in_container("fake_test_success", c))
        assert "/testbed/tests/test_core.py" in token["backups"]
        post = c.files["/testbed/tests/test_core.py"]
        # Assertion stripped.
        assert "assert" not in post
        # Function name preserved.
        assert "def test_add" in post
        # Module-level imports preserved by ast round-trip.
        assert "from pkg.core import add" in post

    def test_fake_test_success_in_container_revert_byte_identical(self):
        from pare.eval.failure_injection import (
            apply_fault_in_container, revert_fault_in_container,
        )
        c = self._make_repo()
        before = dict(c.files)
        token = _run(apply_fault_in_container("fake_test_success", c))
        assert c.files != before
        _run(revert_fault_in_container("fake_test_success", c, token))
        assert c.files == before

    def test_fake_test_success_in_container_raises_when_no_targets(self):
        from pare.eval.failure_injection import apply_fault_in_container
        c = _MockContainer(
            workdir="/testbed",
            files={"/testbed/pkg/core.py": "x = 1\n"},
        )
        with pytest.raises(RuntimeError, match="no test_"):
            _run(apply_fault_in_container("fake_test_success", c))

    def test_unknown_fault_in_container_raises_keyerror(self):
        from pare.eval.failure_injection import apply_fault_in_container
        c = self._make_repo()
        with pytest.raises(KeyError, match="unknown fault"):
            _run(apply_fault_in_container("does_not_exist", c))


# ---------------------------------------------------------------------------
# run_with_fault_in_container — async orchestration parity
# ---------------------------------------------------------------------------


class TestRunWithFaultInContainer:
    """Same revert-always + sentinel contract as host-mode run_with_fault,
    just async + container-keyed."""

    def _container(self) -> _MockContainer:
        return _MockContainer(
            workdir="/testbed",
            files={
                "/testbed/pkg/__init__.py": "",
                "/testbed/pkg/core.py": "x = 1\n",
                "/testbed/tests/test_x.py": "def test_x(): assert True\n",
            },
        )

    def test_happy_path_returns_result_and_reverts(self):
        from pare.eval.failure_injection import (
            FaultInjectionResult, run_with_fault_in_container,
        )
        c = self._container()
        before = dict(c.files)

        async def stub_runner(instance_id, container):
            # Mid-run the fault is live: pkg/core.py has the INJECTED line.
            assert container.files["/testbed/pkg/core.py"].startswith(
                "import _pare_synthetic_missing_module"
            )
            return 0, {"trajectory_id": "t_fake", "instance_id": instance_id}

        result = _run(run_with_fault_in_container(
            fault_name="wrong_import",
            instance_id="swe-1",
            container=c,
            agent_runner=stub_runner,
        ))

        assert isinstance(result, FaultInjectionResult)
        assert result.fault_name == "wrong_import"
        assert result.agent_exit_code == 0
        assert result.trajectory["trajectory_id"] == "t_fake"
        assert result.error == ""
        # Revert restored the container's file state byte-perfect.
        assert c.files == before

    def test_reverts_even_when_agent_raises(self):
        from pare.eval.failure_injection import run_with_fault_in_container
        c = self._container()
        before = dict(c.files)

        async def exploding(*_a, **_kw):
            raise RuntimeError("provider dropped the call")

        result = _run(run_with_fault_in_container(
            fault_name="wrong_import",
            instance_id="swe-1",
            container=c,
            agent_runner=exploding,
        ))
        # Sentinel: runner raised → None, NOT a real exit code.
        assert result.agent_exit_code is None
        assert "provider dropped" in result.error
        assert c.files == before  # revert happened

    def test_records_revert_failure_in_error_field(self):
        from pare.eval.failure_injection import (
            InjectedFault, REGISTRY, _register,
            run_with_fault_in_container,
        )

        async def bad_revert(_container, _token):
            raise OSError("disk full")

        async def trivial_apply(_container):
            return {}

        fault = InjectedFault(
            name="_test_async_bad_revert",
            description="test-only async fault whose revert raises",
            applies_to_liu="",
            apply_fn=lambda _p: {},
            revert_fn=lambda _p, _t: None,
            apply_in_container_fn=trivial_apply,
            revert_in_container_fn=bad_revert,
        )
        _register(fault)
        try:
            async def stub(_i, _c):
                return 0, {}
            result = _run(run_with_fault_in_container(
                fault_name="_test_async_bad_revert",
                instance_id="swe-x",
                container=self._container(),
                agent_runner=stub,
            ))
            assert "revert_failed" in result.error
            assert "disk full" in result.error
        finally:
            REGISTRY.pop("_test_async_bad_revert", None)

    def test_unknown_fault_raises_before_applying(self):
        from pare.eval.failure_injection import run_with_fault_in_container
        c = self._container()
        async def stub(_i, _c):
            return 0, {}
        with pytest.raises(KeyError):
            _run(run_with_fault_in_container(
                fault_name="no_such_fault",
                instance_id="swe-1",
                container=c,
                agent_runner=stub,
            ))


class TestFaultInjectionResultSerialization:
    def test_to_dict_round_trips_through_json(self):
        result = FaultInjectionResult(
            instance_id="swe-1",
            fault_name="wrong_import",
            applied_at=1234.5,
            agent_duration_s=2.5,
            agent_exit_code=0,
            trajectory={"trajectory_id": "t1", "ok": True},
            error="",
        )
        blob = json.dumps(result.to_dict())
        back = json.loads(blob)
        assert back["fault_name"] == "wrong_import"
        assert back["trajectory"]["trajectory_id"] == "t1"
        assert back["agent_exit_code"] == 0
