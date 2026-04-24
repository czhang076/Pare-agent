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
        """The 3 scaffold faults we've written must be loaded."""
        assert "stale_test_cache" in REGISTRY
        assert "wrong_import" in REGISTRY
        assert "empty_baseline" in REGISTRY

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


class TestStaleTestCacheFault:
    def test_apply_creates_cache_then_revert_removes_it(self, tmp_path: Path):
        """On a workdir with no prior .pytest_cache, apply creates one,
        revert takes the workdir back to byte-identical state."""
        _make_minimal_repo(tmp_path)
        before = _snapshot_tree(tmp_path)

        token = apply_fault("stale_test_cache", tmp_path)

        nodeids = tmp_path / ".pytest_cache" / "v" / "cache" / "nodeids"
        assert nodeids.exists(), "fault must plant nodeids file"
        # The planted content should parse as JSON and contain fake tests.
        planted = json.loads(nodeids.read_text(encoding="utf-8"))
        assert any("test_that_does_not_exist" in t for t in planted)

        revert_fault("stale_test_cache", tmp_path, token)
        after = _snapshot_tree(tmp_path)
        assert after == before, "revert must restore byte-identical tree"

    def test_apply_preserves_pre_existing_cache_content(self, tmp_path: Path):
        """If the workdir already has a .pytest_cache/v/cache/nodeids,
        revert must restore its original content (not delete it).

        This is the real-world case: users run pytest before the agent
        ever touches the workdir, so stomping on an existing cache file
        is destructive."""
        _make_minimal_repo(tmp_path)
        cache_dir = tmp_path / ".pytest_cache" / "v" / "cache"
        cache_dir.mkdir(parents=True)
        nodeids = cache_dir / "nodeids"
        nodeids.write_text('["real_test"]', encoding="utf-8")

        token = apply_fault("stale_test_cache", tmp_path)
        # Fault content is in place.
        assert "test_that_does_not_exist" in nodeids.read_text(encoding="utf-8")

        revert_fault("stale_test_cache", tmp_path, token)
        assert nodeids.read_text(encoding="utf-8") == '["real_test"]'


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
