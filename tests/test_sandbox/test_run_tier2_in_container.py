"""Tests for ``pare.sandbox.docker_eval.run_tier2_in_container``.

These tests lock the wire from the loop's tier2 finalize hook to
:class:`DockerEvalSession`. The prior stub returned ``passed=False`` with
empty output unconditionally, silently invalidating every Tier-2 verdict in
every pilot — the regressions below would have caught it immediately.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pare.agent.verify import Tier2CheckResult
from pare.sandbox import docker_eval
from pare.sandbox.docker_eval import (
    DockerEvalUnavailable,
    run_tier2_in_container,
)


@pytest.fixture(autouse=True)
def _clear_session_cache():
    """Each test gets a fresh session cache — no bleed across cases."""
    docker_eval._TIER2_SESSIONS.clear()
    yield
    docker_eval._TIER2_SESSIONS.clear()


def _fake_container(
    *,
    instance_id: str = "sympy__sympy-12489",
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
) -> SimpleNamespace:
    return SimpleNamespace(
        instance_id=instance_id,
        dataset_name=dataset_name,
        split=split,
    )


def _passed_result(iid: str) -> Tier2CheckResult:
    return Tier2CheckResult(
        enabled=True,
        command=f"swebench:{iid}",
        passed=True,
        return_code=0,
        output='{"resolved": true}',
        error="",
    )


async def test_delegates_to_session_verify_diff(monkeypatch):
    container = _fake_container()
    mock_session = MagicMock()
    mock_session.verify_diff = MagicMock(
        return_value=_passed_result(container.instance_id),
    )
    captured: dict = {}

    def fake_get(**kwargs):
        captured.update(kwargs)
        return mock_session

    monkeypatch.setattr(docker_eval, "_get_tier2_session", fake_get)

    result = await run_tier2_in_container(container, "diff --git a/x b/x\n")

    assert captured == {
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
    }
    mock_session.verify_diff.assert_called_once_with(
        "sympy__sympy-12489", "diff --git a/x b/x\n",
    )
    assert result.passed is True
    assert result.return_code == 0
    assert "resolved" in result.output


async def test_threads_container_dataset_coords(monkeypatch):
    """dataset_name/split on container must reach _get_tier2_session."""
    container = _fake_container(
        instance_id="django__django-11099",
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="dev",
    )
    mock_session = MagicMock()
    mock_session.verify_diff = MagicMock(
        return_value=_passed_result("django__django-11099"),
    )
    captured: dict = {}
    monkeypatch.setattr(
        docker_eval,
        "_get_tier2_session",
        lambda **kw: (captured.update(kw) or mock_session),
    )

    await run_tier2_in_container(container, "diff\n")

    assert captured == {
        "dataset_name": "princeton-nlp/SWE-bench_Lite",
        "split": "dev",
    }


async def test_returns_docker_unavailable_gracefully(monkeypatch):
    """Missing docker-eval extra must map to a structured error, not raise."""
    container = _fake_container()

    def boom(**_kwargs):
        raise DockerEvalUnavailable("docker-eval extra not installed")

    monkeypatch.setattr(docker_eval, "_get_tier2_session", boom)

    result = await run_tier2_in_container(container, "diff\n")

    assert result.enabled is True
    assert result.passed is False
    assert result.command == "swebench:sympy__sympy-12489"
    assert "docker-eval extra not installed" in result.error


async def test_empty_diff_is_passed_through_to_session(monkeypatch):
    """Empty-diff short-circuit lives inside verify_diff — run_tier2_in_container
    must not second-guess it. Previously the stub had its own early exit, but
    routing the call through the session keeps one source of truth."""
    container = _fake_container()
    mock_session = MagicMock()
    mock_session.verify_diff = MagicMock(
        return_value=Tier2CheckResult(
            enabled=True,
            command=f"swebench:{container.instance_id}",
            passed=False,
            return_code=None,
            output="",
            error="empty_diff_skipped_docker",
        ),
    )
    monkeypatch.setattr(
        docker_eval, "_get_tier2_session", lambda **_kw: mock_session,
    )

    result = await run_tier2_in_container(container, "")

    mock_session.verify_diff.assert_called_once_with(
        "sympy__sympy-12489", "",
    )
    assert result.passed is False
    assert result.error == "empty_diff_skipped_docker"


async def test_session_is_cached_across_calls(monkeypatch):
    """Two calls with the same (dataset, split) must reuse one session —
    otherwise every tier2 invocation eats a fresh HF dataset load + docker
    ping handshake (~5s+ each)."""
    container = _fake_container()

    build_calls: list[tuple[str, str]] = []

    class FakeSession:
        def __init__(self, dataset_name: str, split: str):
            self.dataset_name = dataset_name
            self.split = split

        def verify_diff(self, iid: str, diff: str) -> Tier2CheckResult:
            return _passed_result(iid)

        def close(self) -> None:
            pass

    def fake_build(config):
        build_calls.append((config.dataset_name, config.split))
        return FakeSession(config.dataset_name, config.split)

    monkeypatch.setattr(docker_eval, "build_session", fake_build)

    await run_tier2_in_container(container, "diff A\n")
    await run_tier2_in_container(container, "diff B\n")

    assert build_calls == [("princeton-nlp/SWE-bench_Verified", "test")]

    # Different coords ⇒ fresh session.
    other = _fake_container(
        dataset_name="princeton-nlp/SWE-bench_Lite", split="dev",
    )
    await run_tier2_in_container(other, "diff C\n")
    assert build_calls == [
        ("princeton-nlp/SWE-bench_Verified", "test"),
        ("princeton-nlp/SWE-bench_Lite", "dev"),
    ]


async def test_falls_back_to_defaults_when_container_lacks_coords(monkeypatch):
    """Defensive: if a caller passes a container-like object without the
    dataset fields (e.g. a mock in an older test), we still resolve the
    default SWE-bench_Verified/test coords rather than blowing up."""
    container = SimpleNamespace(instance_id="sympy__sympy-12489")
    mock_session = MagicMock()
    mock_session.verify_diff = MagicMock(
        return_value=_passed_result("sympy__sympy-12489"),
    )
    captured: dict = {}
    monkeypatch.setattr(
        docker_eval,
        "_get_tier2_session",
        lambda **kw: (captured.update(kw) or mock_session),
    )

    await run_tier2_in_container(container, "diff\n")

    assert captured == {
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
    }
