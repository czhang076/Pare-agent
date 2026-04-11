"""Tests for headless batch mode.

Tests cover:
- Arg parser: task + output flags
- run_headless(): JSON output, exit codes, stderr logging
- _result_to_dict(): serialization of ExecutionResult
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from pare.agent.executor import ExecutionResult
from pare.cli.headless import _result_to_dict, run_headless
from pare.llm.base import (
    LLMAdapter,
    LLMResponse,
    ModelProfile,
    StopReason,
    StreamChunk,
    TokenUsage,
)
from pare.main import build_parser


# ---------------------------------------------------------------------------
# Arg parser tests
# ---------------------------------------------------------------------------


class TestHeadlessArgs:
    def test_task_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_output_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "result.json", "fix bug"])
        assert args.output == "result.json"

    def test_output_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-o", "out.json", "fix bug"])
        assert args.output == "out.json"

    def test_defaults_with_task(self):
        parser = build_parser()
        args = parser.parse_args(["fix bug"])
        assert args.task == "fix bug"
        assert args.provider == "openai"
        assert args.output is None


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


class TestResultToDict:
    def test_success_result(self):
        result = ExecutionResult(
            success=True,
            output="Fixed the bug.",
            messages=[],
            tool_call_count=3,
            stop_reason="end_turn",
            total_usage=TokenUsage(
                input_tokens=500, output_tokens=200,
                cache_read_tokens=100, cache_create_tokens=50,
            ),
        )
        d = _result_to_dict(result)

        assert d["success"] is True
        assert d["output"] == "Fixed the bug."
        assert d["stop_reason"] == "end_turn"
        assert d["tool_call_count"] == 3
        assert d["usage"]["input_tokens"] == 500
        assert d["usage"]["output_tokens"] == 200
        assert d["usage"]["total_tokens"] == 700
        assert d["usage"]["cache_read_tokens"] == 100
        assert d["usage"]["cache_create_tokens"] == 50

    def test_failure_result(self):
        result = ExecutionResult(
            success=False,
            output="",
            messages=[],
            tool_call_count=0,
            stop_reason="error",
            total_usage=TokenUsage(input_tokens=100, output_tokens=0),
        )
        d = _result_to_dict(result)

        assert d["success"] is False
        assert d["stop_reason"] == "error"
        assert d["usage"]["total_tokens"] == 100


# ---------------------------------------------------------------------------
# Mock LLM for integration tests
# ---------------------------------------------------------------------------

_USAGE = TokenUsage(input_tokens=100, output_tokens=50)


class MockHeadlessLLM(LLMAdapter):
    """LLM that returns a single text response (no tool calls)."""

    def __init__(self, text: str = "Done.", success: bool = True) -> None:
        super().__init__(model="mock-headless", profile=ModelProfile())
        self._text = text

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        return LLMResponse(
            content=self._text,
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=_USAGE,
        )

    async def chat_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    def count_tokens(self, messages) -> int:
        return 100


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunHeadless:
    @pytest.mark.asyncio
    async def test_success_writes_json(self, tmp_path: Path):
        """Headless run writes structured JSON to output path."""
        output = tmp_path / "result.json"

        with patch("pare.cli.headless.create_llm", return_value=MockHeadlessLLM("All fixed.")):
            code = await run_headless(
                task="Fix the typo",
                api_key="test-key",
                cwd=tmp_path,
                output_path=output,
            )

        assert code == 0
        assert output.exists()

        data = json.loads(output.read_text())
        assert data["success"] is True
        assert data["output"] == "All fixed."
        assert data["usage"]["input_tokens"] == 100
        assert data["usage"]["output_tokens"] == 50
        assert "elapsed_seconds" in data

    @pytest.mark.asyncio
    async def test_success_no_output_file(self, tmp_path: Path, capsys):
        """Headless run without --output still prints final text to stdout."""
        with patch("pare.cli.headless.create_llm", return_value=MockHeadlessLLM("Result text.")):
            code = await run_headless(
                task="Do something",
                api_key="test-key",
                cwd=tmp_path,
            )

        assert code == 0
        captured = capsys.readouterr()
        assert "Result text." in captured.out

    @pytest.mark.asyncio
    async def test_no_api_key_returns_2(self, tmp_path: Path, monkeypatch):
        """Missing API key returns exit code 2."""
        # Clear env var for default openai provider.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        code = await run_headless(
            task="Fix bug",
            provider="openai",
            api_key=None,
            cwd=tmp_path,
        )

        assert code == 2

    @pytest.mark.asyncio
    async def test_stderr_logging(self, tmp_path: Path, capsys):
        """Headless run logs progress to stderr."""
        with patch("pare.cli.headless.create_llm", return_value=MockHeadlessLLM("Done.")):
            await run_headless(
                task="Fix the bug",
                api_key="test-key",
                cwd=tmp_path,
            )

        captured = capsys.readouterr()
        assert "[start]" in captured.err
        assert "[config]" in captured.err
        assert "[done]" in captured.err

    @pytest.mark.asyncio
    async def test_output_creates_parent_dirs(self, tmp_path: Path):
        """Output path with nested dirs creates them automatically."""
        output = tmp_path / "nested" / "dir" / "result.json"

        with patch("pare.cli.headless.create_llm", return_value=MockHeadlessLLM("Done.")):
            code = await run_headless(
                task="Task",
                api_key="test-key",
                cwd=tmp_path,
                output_path=output,
            )

        assert code == 0
        assert output.exists()

    @pytest.mark.asyncio
    async def test_json_is_valid_and_complete(self, tmp_path: Path):
        """Verify all expected keys are present in the JSON output."""
        output = tmp_path / "result.json"

        with patch("pare.cli.headless.create_llm", return_value=MockHeadlessLLM("Done.")):
            await run_headless(
                task="Task",
                api_key="test-key",
                cwd=tmp_path,
                output_path=output,
            )

        data = json.loads(output.read_text())
        expected_keys = {"success", "output", "stop_reason", "tool_call_count", "usage", "elapsed_seconds"}
        assert set(data.keys()) == expected_keys

        usage_keys = {"input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_create_tokens"}
        assert set(data["usage"].keys()) == usage_keys
