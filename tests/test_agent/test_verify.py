"""Tests for hard verification — Tier 1 checks.

Tests cover:
- syntax_check(): Python compile() validation after edits
- git_diff_check(): detect empty diffs
- Executor integration: syntax errors appended to tool results
- Executor integration: diff-empty warning injected when agent finishes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from pare.agent.executor import ReActExecutor
from pare.agent.verify import git_diff_check, syntax_check
from pare.llm.base import (
    LLMAdapter,
    LLMResponse,
    Message,
    ModelProfile,
    StopReason,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
    ToolSchema,
)
from pare.agent.guardrails import Guardrails
from pare.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)

_USAGE = TokenUsage(input_tokens=10, output_tokens=5)


def _guardrails_allowing(*files: str) -> Guardrails:
    """Create guardrails with files pre-marked as read (bypasses read-before-write)."""
    g = Guardrails()
    for f in files:
        g.state.read_files.add(f)
    return g


# ---------------------------------------------------------------------------
# Unit tests: syntax_check
# ---------------------------------------------------------------------------


class TestSyntaxCheck:
    def test_valid_python(self, tmp_path: Path):
        f = tmp_path / "good.py"
        f.write_text("x = 1\nprint(x)\n")
        assert syntax_check(f) is None

    def test_syntax_error(self, tmp_path: Path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n")
        result = syntax_check(f)
        assert result is not None
        assert "SyntaxError" in result
        assert "bad.py" in result

    def test_non_python_file_skipped(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text("{invalid json")
        assert syntax_check(f) is None

    def test_empty_python_file(self, tmp_path: Path):
        f = tmp_path / "empty.py"
        f.write_text("")
        assert syntax_check(f) is None

    def test_indentation_error(self, tmp_path: Path):
        f = tmp_path / "indent.py"
        f.write_text("def foo():\nx = 1\n")
        result = syntax_check(f)
        assert result is not None
        assert "indent.py" in result

    def test_missing_file_returns_none(self, tmp_path: Path):
        f = tmp_path / "nope.py"
        assert syntax_check(f) is None


# ---------------------------------------------------------------------------
# Unit tests: git_diff_check
# ---------------------------------------------------------------------------


class TestGitDiffCheck:
    def test_has_changes(self, tmp_path: Path):
        """In a git repo with uncommitted changes, returns True."""
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

        # Create and commit a file
        (tmp_path / "a.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        # Modify the file
        (tmp_path / "a.txt").write_text("hello world")

        assert git_diff_check(tmp_path) is True

    def test_no_changes(self, tmp_path: Path):
        """In a clean git repo, returns False."""
        import subprocess
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

        (tmp_path / "a.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)

        assert git_diff_check(tmp_path) is False

    def test_not_a_git_repo(self, tmp_path: Path):
        """Non-git directory returns True (fail-open)."""
        assert git_diff_check(tmp_path) is True


# ---------------------------------------------------------------------------
# Mock tools for executor integration tests
# ---------------------------------------------------------------------------


class MockFileEditTool(Tool):
    """Simulates file_edit: writes a file and returns a diff."""

    name = "file_edit"
    description = "Edit a file"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        },
    }
    mutation_type = MutationType.WRITE
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        fp = params.get("file_path", "")
        new_str = params.get("new_str", "")
        full = (context.cwd / fp).resolve()
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(new_str, encoding="utf-8")
        return ToolResult(success=True, output=f"Edited {fp}", metadata={"file_path": fp})


class MockFileCreateTool(Tool):
    """Simulates file_create."""

    name = "file_create"
    description = "Create a file"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
        },
    }
    mutation_type = MutationType.WRITE
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        fp = params.get("file_path", "")
        content = params.get("content", "")
        full = (context.cwd / fp).resolve()
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return ToolResult(success=True, output=f"Created {fp}", metadata={"file_path": fp})


class MockLLM(LLMAdapter):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="mock", profile=ModelProfile())
        self._responses = list(responses)
        self._idx = 0

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        if self._idx >= len(self._responses):
            return LLMResponse(content="Done.", tool_calls=[], stop_reason=StopReason.END_TURN, usage=_USAGE)
        r = self._responses[self._idx]
        self._idx += 1
        return r

    async def chat_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    def count_tokens(self, messages) -> int:
        return 100


def _tc(tool_name: str, args: dict) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id=f"tc_{tool_name}", name=tool_name, arguments=args)],
        stop_reason=StopReason.TOOL_USE,
        usage=_USAGE,
    )


def _text(t: str) -> LLMResponse:
    return LLMResponse(content=t, tool_calls=[], stop_reason=StopReason.END_TURN, usage=_USAGE)


# ---------------------------------------------------------------------------
# Executor integration: syntax check
# ---------------------------------------------------------------------------


class TestExecutorSyntaxCheck:
    @pytest.mark.asyncio
    async def test_syntax_error_appended_to_result(self, tmp_path: Path):
        """When file_edit creates invalid Python, the syntax error appears in tool result."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        llm = MockLLM([
            _tc("file_edit", {"file_path": "bad.py", "old_str": "", "new_str": "def foo(\n"}),
            _text("I edited the file."),
        ])

        executor = ReActExecutor(llm, registry, guardrails=_guardrails_allowing("bad.py"))
        ctx = ToolContext(cwd=tmp_path, headless=True)
        result = await executor.run("system", "task", ctx)

        # The conversation should contain the syntax error in tool result
        tool_result_msg = [m for m in result.messages if m.role == "tool_result"]
        assert len(tool_result_msg) >= 1

        # Find the tool result text
        blocks = tool_result_msg[0].content
        assert any("SYNTAX ERROR" in b.text for b in blocks)

    @pytest.mark.asyncio
    async def test_valid_python_no_warning(self, tmp_path: Path):
        """When file_edit creates valid Python, no syntax warning."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        llm = MockLLM([
            _tc("file_edit", {"file_path": "good.py", "old_str": "", "new_str": "x = 1\n"}),
            _text("Done."),
        ])

        executor = ReActExecutor(llm, registry, guardrails=_guardrails_allowing("good.py"))
        ctx = ToolContext(cwd=tmp_path, headless=True)
        result = await executor.run("system", "task", ctx)

        tool_result_msg = [m for m in result.messages if m.role == "tool_result"]
        blocks = tool_result_msg[0].content
        assert not any("SYNTAX ERROR" in b.text for b in blocks)

    @pytest.mark.asyncio
    async def test_non_python_file_no_check(self, tmp_path: Path):
        """Editing a non-.py file skips syntax check."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        llm = MockLLM([
            _tc("file_edit", {"file_path": "data.json", "old_str": "", "new_str": "{invalid"}),
            _text("Done."),
        ])

        executor = ReActExecutor(llm, registry, guardrails=_guardrails_allowing("data.json"))
        ctx = ToolContext(cwd=tmp_path, headless=True)
        result = await executor.run("system", "task", ctx)

        tool_result_msg = [m for m in result.messages if m.role == "tool_result"]
        blocks = tool_result_msg[0].content
        assert not any("SYNTAX ERROR" in b.text for b in blocks)

    @pytest.mark.asyncio
    async def test_file_create_also_checked(self, tmp_path: Path):
        """file_create on .py files also gets syntax checked."""
        registry = ToolRegistry()
        registry.register(MockFileCreateTool())

        llm = MockLLM([
            _tc("file_create", {"file_path": "new.py", "content": "class Foo\n"}),
            _text("Created."),
        ])

        # file_create doesn't need read-before-write, but pass guardrails for consistency
        executor = ReActExecutor(llm, registry)
        ctx = ToolContext(cwd=tmp_path, headless=True)
        result = await executor.run("system", "task", ctx)

        tool_result_msg = [m for m in result.messages if m.role == "tool_result"]
        blocks = tool_result_msg[0].content
        assert any("SYNTAX ERROR" in b.text for b in blocks)


# ---------------------------------------------------------------------------
# Executor integration: diff check
# ---------------------------------------------------------------------------


class TestExecutorDiffCheck:
    @pytest.mark.asyncio
    async def test_diff_empty_warning_injected(self, tmp_path: Path):
        """When agent calls tools but git shows no diff, a warning is injected."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        # LLM: calls file_edit, then says done, then (after warning) says done again
        llm = MockLLM([
            _tc("file_edit", {"file_path": "x.py", "old_str": "", "new_str": "x=1\n"}),
            _text("I fixed it."),
            _text("Yes, the task is done."),
        ])

        executor = ReActExecutor(llm, registry, guardrails=_guardrails_allowing("x.py"))
        ctx = ToolContext(cwd=tmp_path, headless=True)

        # Patch git_diff_check to return False (no changes detected)
        with patch("pare.agent.executor.git_diff_check", return_value=False):
            result = await executor.run("system", "task", ctx)

        # The warning message should be in the conversation
        user_msgs = [m for m in result.messages if m.role == "user"]
        assert any("git diff" in (m.content if isinstance(m.content, str) else "") for m in user_msgs)

    @pytest.mark.asyncio
    async def test_diff_present_no_warning(self, tmp_path: Path):
        """When git shows changes, no diff warning."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        llm = MockLLM([
            _tc("file_edit", {"file_path": "x.py", "old_str": "", "new_str": "x=1\n"}),
            _text("Done."),
        ])

        executor = ReActExecutor(llm, registry, guardrails=_guardrails_allowing("x.py"))
        ctx = ToolContext(cwd=tmp_path, headless=True)

        # git_diff_check returns True (changes exist)
        with patch("pare.agent.executor.git_diff_check", return_value=True):
            result = await executor.run("system", "task", ctx)

        user_msgs = [m for m in result.messages if m.role == "user"]
        assert not any("git diff" in (m.content if isinstance(m.content, str) else "") for m in user_msgs)

    @pytest.mark.asyncio
    async def test_diff_warning_only_once(self, tmp_path: Path):
        """Diff warning is only injected once, not on every end_turn."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        # After first warning, LLM says done again — should not get a second warning
        llm = MockLLM([
            _tc("file_edit", {"file_path": "x.py", "old_str": "", "new_str": "x=1\n"}),
            _text("Done."),        # Triggers diff warning
            _text("Confirmed."),   # Should NOT trigger again
        ])

        executor = ReActExecutor(llm, registry, guardrails=_guardrails_allowing("x.py"))
        ctx = ToolContext(cwd=tmp_path, headless=True)

        with patch("pare.agent.executor.git_diff_check", return_value=False):
            result = await executor.run("system", "task", ctx)

        # Only one diff warning
        user_msgs = [
            m for m in result.messages
            if m.role == "user" and isinstance(m.content, str) and "git diff" in m.content
        ]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_no_tool_calls_no_diff_check(self, tmp_path: Path):
        """When no tools were called, diff check is skipped."""
        registry = ToolRegistry()
        registry.register(MockFileEditTool())

        llm = MockLLM([
            _text("Nothing to do."),
        ])

        executor = ReActExecutor(llm, registry)
        ctx = ToolContext(cwd=tmp_path, headless=True)

        with patch("pare.agent.executor.git_diff_check", return_value=False) as mock_diff:
            result = await executor.run("system", "task", ctx)

        # git_diff_check should not be called since total_calls == 0
        mock_diff.assert_not_called()
