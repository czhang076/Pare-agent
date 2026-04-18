"""Tests for pare/sandbox/docker_eval.py.

swebench/docker/datasets are NOT imported by these tests — they're installed
only via the `docker-eval` extra. We monkeypatch DockerEvalSession._require_extra
to inject stub callables so tests run in any Pare dev environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pare.sandbox.docker_eval import (
    DockerEvalConfig,
    DockerEvalSession,
    DockerEvalUnavailable,
    build_session,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _stub_dataset(instance_ids: list[str]) -> list[dict]:
    return [{"instance_id": iid, "repo": "x/y", "base_commit": "abc"} for iid in instance_ids]


def _install_stubs(
    session: DockerEvalSession,
    *,
    run_instance: MagicMock | None = None,
    instance_ids: list[str] | None = None,
    docker_ping_ok: bool = True,
) -> SimpleNamespace:
    """Bypass _require_extra by injecting stubs for swebench/docker/datasets."""
    run_instance = run_instance or MagicMock(return_value=None)
    make_test_spec = MagicMock(side_effect=lambda row: SimpleNamespace(instance_id=row["instance_id"]))
    load_dataset = MagicMock(return_value=_stub_dataset(instance_ids or ["i1"]))

    docker_client = MagicMock()
    if docker_ping_ok:
        docker_client.ping = MagicMock(return_value=True)
    else:
        docker_client.ping = MagicMock(side_effect=RuntimeError("daemon down"))
    docker_module = MagicMock()
    docker_module.from_env = MagicMock(return_value=docker_client)

    session._require_extra = MagicMock(  # type: ignore[method-assign]
        return_value=(run_instance, make_test_spec, load_dataset, docker_module),
    )
    return SimpleNamespace(
        run_instance=run_instance,
        make_test_spec=make_test_spec,
        load_dataset=load_dataset,
        docker_client=docker_client,
    )


def _write_report(logs_root: Path, config: DockerEvalConfig, instance_id: str, payload: dict) -> None:
    path = (
        logs_root
        / config.run_id
        / config.model_name
        / instance_id
        / "report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_session(tmp_path: Path, **cfg_overrides) -> DockerEvalSession:
    cfg = DockerEvalConfig(logs_root=tmp_path / "logs", **cfg_overrides)
    return DockerEvalSession(cfg)


# ---------------------------------------------------------------------------
# build_session / config
# ---------------------------------------------------------------------------


def test_build_session_defaults() -> None:
    s = build_session()
    assert isinstance(s, DockerEvalSession)
    assert s.config.model_name == "pare_v6"


# ---------------------------------------------------------------------------
# verify_diff happy paths
# ---------------------------------------------------------------------------


def test_verify_diff_resolved_true_maps_to_passed(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    stubs = _install_stubs(session, instance_ids=["sympy__sympy-12489"])
    _write_report(
        session.config.logs_root,
        session.config,
        "sympy__sympy-12489",
        {"resolved": True, "FAIL_TO_PASS": {}, "PASS_TO_PASS": {}},
    )

    result = session.verify_diff("sympy__sympy-12489", "diff --git a/x b/x\n")

    assert result.enabled is True
    assert result.passed is True
    assert result.return_code == 0
    assert result.error == ""
    assert "resolved" in result.output
    assert result.command == "swebench:sympy__sympy-12489"
    assert stubs.run_instance.call_count == 1


def test_verify_diff_resolved_false_maps_to_failed_with_output(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _install_stubs(session, instance_ids=["sympy__sympy-12489"])
    _write_report(
        session.config.logs_root,
        session.config,
        "sympy__sympy-12489",
        {"resolved": False, "FAIL_TO_PASS": {"test_a": "failed"}},
    )

    result = session.verify_diff("sympy__sympy-12489", "some diff\n")

    assert result.passed is False
    assert result.return_code == 1
    assert "failed" in result.output
    assert result.error == ""


def test_verify_diff_unwraps_instance_keyed_report(tmp_path: Path) -> None:
    """Some harness versions wrap report as {instance_id: {...}}."""
    session = _make_session(tmp_path)
    _install_stubs(session, instance_ids=["sympy__sympy-12489"])
    _write_report(
        session.config.logs_root,
        session.config,
        "sympy__sympy-12489",
        {"sympy__sympy-12489": {"resolved": True}},
    )

    result = session.verify_diff("sympy__sympy-12489", "diff\n")
    assert result.passed is True


# ---------------------------------------------------------------------------
# short-circuits & error mapping
# ---------------------------------------------------------------------------


def test_empty_diff_short_circuits_without_docker_call(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    stubs = _install_stubs(session, instance_ids=["sympy__sympy-12489"])

    result = session.verify_diff("sympy__sympy-12489", "")

    assert result.passed is False
    assert result.error == "empty_diff_skipped_docker"
    stubs.run_instance.assert_not_called()
    # Dataset must not have been loaded either.
    stubs.load_dataset.assert_not_called()


def test_whitespace_only_diff_also_short_circuits(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    stubs = _install_stubs(session, instance_ids=["sympy__sympy-12489"])

    result = session.verify_diff("sympy__sympy-12489", "   \n\n  \t")

    assert result.error == "empty_diff_skipped_docker"
    stubs.run_instance.assert_not_called()


def test_missing_extra_returns_error_not_crash(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session._require_extra = MagicMock(  # type: ignore[method-assign]
        side_effect=DockerEvalUnavailable("docker-eval extra missing"),
    )

    result = session.verify_diff("sympy__sympy-12489", "diff\n")

    assert result.passed is False
    assert "docker-eval extra missing" in result.error


def test_daemon_down_returns_error_not_crash(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _install_stubs(session, instance_ids=["sympy__sympy-12489"], docker_ping_ok=False)

    result = session.verify_diff("sympy__sympy-12489", "diff\n")

    assert result.passed is False
    assert "Docker daemon unreachable" in result.error


def test_unknown_instance_id_returns_error(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _install_stubs(session, instance_ids=["sympy__sympy-12489"])

    result = session.verify_diff("does-not-exist-1", "diff\n")

    assert result.passed is False
    assert "not in dataset" in result.error


def test_run_instance_exception_returns_tier2_error(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    run_instance = MagicMock(side_effect=RuntimeError("patch apply failed"))
    _install_stubs(session, instance_ids=["sympy__sympy-12489"], run_instance=run_instance)

    result = session.verify_diff("sympy__sympy-12489", "diff\n")

    assert result.passed is False
    assert "run_instance_failed" in result.error
    assert "patch apply failed" in result.error


def test_report_missing_is_error_not_silent_fail(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _install_stubs(session, instance_ids=["sympy__sympy-12489"])
    # No report written → run_instance succeeds but logs dir is empty.

    result = session.verify_diff("sympy__sympy-12489", "diff\n")

    assert result.passed is False
    assert "report_missing" in result.error


def test_report_parse_failed_is_caught(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    _install_stubs(session, instance_ids=["sympy__sympy-12489"])
    path = (
        session.config.logs_root
        / session.config.run_id
        / session.config.model_name
        / "sympy__sympy-12489"
        / "report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    result = session.verify_diff("sympy__sympy-12489", "diff\n")

    assert result.passed is False
    assert "report_parse_failed" in result.error


# ---------------------------------------------------------------------------
# caching behaviour
# ---------------------------------------------------------------------------


def test_dataset_loaded_once_for_multiple_instances(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    stubs = _install_stubs(session, instance_ids=["i1", "i2", "i3"])
    for iid in ("i1", "i2", "i3"):
        _write_report(session.config.logs_root, session.config, iid, {"resolved": True})

    for iid in ("i1", "i2", "i3"):
        session.verify_diff(iid, "diff\n")

    assert stubs.load_dataset.call_count == 1
    assert stubs.run_instance.call_count == 3


def test_test_spec_memoized(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    stubs = _install_stubs(session, instance_ids=["i1"])
    _write_report(session.config.logs_root, session.config, "i1", {"resolved": True})

    session.verify_diff("i1", "diff1\n")
    session.verify_diff("i1", "diff2\n")

    # Dataset loaded once (session init), make_test_spec called once
    # (second call hit the lru_cache).
    assert stubs.load_dataset.call_count == 1
    assert stubs.make_test_spec.call_count == 1
    assert stubs.run_instance.call_count == 2


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------


def test_close_is_idempotent(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    stubs = _install_stubs(session, instance_ids=["i1"])
    _write_report(session.config.logs_root, session.config, "i1", {"resolved": True})
    session.verify_diff("i1", "diff\n")

    session.close()
    session.close()  # second call must not raise
    assert stubs.docker_client.close.call_count == 1


def test_close_before_any_call_is_safe(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    session.close()  # no _ensure_ready was ever called


# ---------------------------------------------------------------------------
# integration guard (skips by default)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "PARE_DOCKER_TIER2_E2E" not in __import__("os").environ,
    reason="set PARE_DOCKER_TIER2_E2E=1 to run real docker harness (requires sympy image + daemon)",
)
def test_e2e_noop_patch_fails_resolve() -> None:  # pragma: no cover — runs only in Linux E2E
    """Smoke: empty patch to a real instance must return resolved=False.

    This is the minimal real-docker path — if this passes in CI/Linux,
    the wiring from DockerEvalSession to the harness is correct.
    """
    session = build_session(DockerEvalConfig(run_id="pare-tier2-e2e"))
    try:
        result = session.verify_diff("sympy__sympy-20639", "noop\n")
        assert result.enabled is True
        # Noop patch can't resolve FAIL_TO_PASS; either patch-apply fails
        # or pytest reports failures. Both must map to passed=False.
        assert result.passed is False
    finally:
        session.close()
