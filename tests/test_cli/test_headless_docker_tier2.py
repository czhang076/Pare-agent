"""Tests for the docker tier2 seam in run_headless.

Verify that when tier2_mode="docker":
- The in-agent host tier2 subprocess is disabled
  (AgentConfig.tier2_test_command gets forced to None).
- The injected tier2_verifier is called after final_diff capture with
  (instance_id, final_diff).
- The verifier's Tier2CheckResult is merged into the trajectory JSONL's
  verification.tier2_* fields.
- tier2_mode="off" disables tier2 entirely.
- tier2_mode="docker" without a verifier warns but doesn't crash.

The agent loop itself is short-circuited with a MockHeadlessLLM that
emits an end-turn with no tool calls — we're testing the wiring, not
the agent's behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from pare.agent.verify import Tier2CheckResult
from pare.cli.headless import run_headless
from pare.llm.base import (
    LLMAdapter,
    LLMResponse,
    ModelProfile,
    StopReason,
    StreamChunk,
    TokenUsage,
)
from pare.trajectory.schema import load_trajectory_jsonl


_USAGE = TokenUsage(input_tokens=10, output_tokens=5)


class _NoopLLM(LLMAdapter):
    def __init__(self) -> None:
        super().__init__(model="mock-tier2", profile=ModelProfile())

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        return LLMResponse(
            content="Done.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=_USAGE,
        )

    async def chat_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    def count_tokens(self, messages) -> int:
        return 10


def _read_one_record(path: Path):
    records = load_trajectory_jsonl(path)
    assert len(records) == 1, f"expected exactly one record in {path}"
    return records[0]


class TestDockerTier2Seam:
    @pytest.mark.asyncio
    async def test_docker_verifier_result_populates_trajectory(self, tmp_path: Path) -> None:
        calls: list[tuple[str, str]] = []

        def fake_verifier(instance_id: str, diff: str) -> Tier2CheckResult:
            calls.append((instance_id, diff))
            return Tier2CheckResult(
                enabled=True,
                command="swebench:sympy__sympy-12489",
                passed=True,
                return_code=0,
                output='{"resolved": true}',
                error="",
            )

        traj = tmp_path / "trajectory.jsonl"
        with patch("pare.cli.headless.create_llm", return_value=_NoopLLM()):
            code = await run_headless(
                task="make it work",
                api_key="test-key",
                cwd=tmp_path,
                trajectory_path=traj,
                instance_id="sympy__sympy-12489",
                tier2_mode="docker",
                tier2_verifier=fake_verifier,
            )

        assert code == 0
        assert len(calls) == 1
        assert calls[0][0] == "sympy__sympy-12489"

        rec = _read_one_record(traj)
        assert rec.verification.tier2_pass is True
        assert rec.verification.tier2_command == "swebench:sympy__sympy-12489"
        assert rec.metadata.get("tier2_output") == '{"resolved": true}'

    @pytest.mark.asyncio
    async def test_docker_failed_verification_recorded(self, tmp_path: Path) -> None:
        def fake_verifier(iid: str, diff: str) -> Tier2CheckResult:
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{iid}",
                passed=False,
                return_code=1,
                output='{"resolved": false}',
                error="",
            )

        traj = tmp_path / "t.jsonl"
        with patch("pare.cli.headless.create_llm", return_value=_NoopLLM()):
            await run_headless(
                task="try",
                api_key="test-key",
                cwd=tmp_path,
                trajectory_path=traj,
                instance_id="django__django-1000",
                tier2_mode="docker",
                tier2_verifier=fake_verifier,
            )

        rec = _read_one_record(traj)
        assert rec.verification.tier2_pass is False
        assert rec.verification.tier2_command == "swebench:django__django-1000"
        assert rec.metadata.get("tier2_output") == '{"resolved": false}'

    @pytest.mark.asyncio
    async def test_docker_mode_forces_agent_config_tier2_to_none(
        self, tmp_path: Path
    ) -> None:
        """Even if caller passes a host test_command, docker mode must null it out
        in AgentConfig so the subprocess-based host checker never runs."""
        captured: list[object] = []

        real_agent_ctor = None

        def _capture_agent_config(*args, **kwargs):
            captured.append(kwargs.get("config"))
            return real_agent_ctor(*args, **kwargs)

        import pare.cli.headless as headless_mod
        real_agent_ctor = headless_mod.Agent

        verifier_calls: list[tuple[str, str]] = []

        def fake_verifier(iid: str, diff: str) -> Tier2CheckResult:
            verifier_calls.append((iid, diff))
            return Tier2CheckResult(enabled=True, passed=True)

        with (
            patch("pare.cli.headless.create_llm", return_value=_NoopLLM()),
            patch("pare.cli.headless.Agent", side_effect=_capture_agent_config),
        ):
            await run_headless(
                task="x",
                api_key="test-key",
                cwd=tmp_path,
                instance_id="i1",
                test_command="pytest tests/",  # would be used in host mode
                tier2_mode="docker",
                tier2_verifier=fake_verifier,
            )

        assert captured, "Agent was never constructed"
        agent_config = captured[0]
        assert agent_config.tier2_test_command is None, (
            "docker mode must disable host tier2 subprocess"
        )

    @pytest.mark.asyncio
    async def test_docker_mode_missing_verifier_warns_not_crashes(
        self, tmp_path: Path, capsys
    ) -> None:
        traj = tmp_path / "t.jsonl"
        with patch("pare.cli.headless.create_llm", return_value=_NoopLLM()):
            code = await run_headless(
                task="x",
                api_key="test-key",
                cwd=tmp_path,
                trajectory_path=traj,
                instance_id="i1",
                tier2_mode="docker",
                tier2_verifier=None,
            )

        assert code == 0
        captured = capsys.readouterr()
        assert "no tier2_verifier" in captured.err

        # Trajectory still written; tier2 not populated since verifier was absent.
        rec = _read_one_record(traj)
        assert rec.verification.tier2_pass is False
        assert rec.verification.tier2_command == ""

    @pytest.mark.asyncio
    async def test_off_mode_skips_all_tier2(self, tmp_path: Path) -> None:
        """tier2_mode=off: host tier2 disabled, no verifier called."""
        calls: list = []

        def verifier(iid: str, diff: str) -> Tier2CheckResult:
            calls.append((iid, diff))
            return Tier2CheckResult(enabled=True, passed=True)

        traj = tmp_path / "t.jsonl"
        with patch("pare.cli.headless.create_llm", return_value=_NoopLLM()):
            await run_headless(
                task="x",
                api_key="test-key",
                cwd=tmp_path,
                trajectory_path=traj,
                instance_id="i1",
                test_command="pytest",
                tier2_mode="off",
                tier2_verifier=verifier,
            )

        assert calls == [], "verifier must not be called in off mode"
        rec = _read_one_record(traj)
        assert rec.verification.tier2_pass is False
        assert rec.verification.tier2_command == ""

    @pytest.mark.asyncio
    async def test_host_mode_does_not_call_docker_verifier(
        self, tmp_path: Path
    ) -> None:
        """Back-compat: default tier2_mode=host must not invoke the docker verifier
        even if one is passed."""
        calls: list = []

        def verifier(iid: str, diff: str) -> Tier2CheckResult:
            calls.append((iid, diff))
            return Tier2CheckResult(enabled=True, passed=True)

        traj = tmp_path / "t.jsonl"
        with patch("pare.cli.headless.create_llm", return_value=_NoopLLM()):
            await run_headless(
                task="x",
                api_key="test-key",
                cwd=tmp_path,
                trajectory_path=traj,
                instance_id="i1",
                tier2_verifier=verifier,
                # tier2_mode defaults to "host"
            )

        assert calls == [], "host mode must not call docker verifier"
