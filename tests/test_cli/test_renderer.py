"""Tests for pare/cli/renderer.py."""

from io import StringIO

import pytest

from rich.console import Console

from pare.agent.executor import ToolCallEvent
from pare.cli.renderer import StreamRenderer
from pare.tools.base import ToolResult


@pytest.fixture
def output() -> StringIO:
    return StringIO()


@pytest.fixture
def renderer(output: StringIO) -> StreamRenderer:
    console = Console(file=output, force_terminal=True, width=120)
    return StreamRenderer(console)


class TestStreamRenderer:
    def test_text_delta_starts_block(self, renderer, output):
        renderer.on_text_delta("Hello ")
        renderer.on_text_delta("world")
        renderer.finish()
        text = output.getvalue()
        assert "◆" in text
        assert "Hello " in text
        assert "world" in text

    def test_tool_call_success(self, renderer, output):
        event = ToolCallEvent(
            tool_name="file_read",
            arguments={"file_path": "main.py"},
            result=ToolResult(success=True, output="x = 1"),
            duration=0.05,
        )
        renderer.on_tool_call(event)
        text = output.getvalue()
        assert "file_read" in text
        assert "main.py" in text
        assert "✓" in text

    def test_tool_call_failure(self, renderer, output):
        event = ToolCallEvent(
            tool_name="bash",
            arguments={"command": "python test.py"},
            result=ToolResult(success=False, output="", error="Exit code: 1"),
            duration=1.2,
        )
        renderer.on_tool_call(event)
        text = output.getvalue()
        assert "bash" in text
        assert "Exit code: 1" in text
        assert "✗" in text

    def test_tool_call_blocked(self, renderer, output):
        event = ToolCallEvent(
            tool_name="file_edit",
            arguments={"file_path": "main.py"},
            blocked_reason="you must read it first",
            duration=0.0,
        )
        renderer.on_tool_call(event)
        text = output.getvalue()
        assert "BLOCKED" in text
        assert "read it first" in text

    def test_text_then_tool_then_text(self, renderer, output):
        """Verify clean transitions between text and tool blocks."""
        renderer.on_text_delta("Thinking...")
        renderer.on_tool_call(ToolCallEvent(
            tool_name="search",
            arguments={"pattern": "def main"},
            result=ToolResult(success=True, output="main.py:1:def main():"),
        ))
        renderer.on_text_delta("Found it.")
        renderer.finish()
        text = output.getvalue()
        assert "Thinking..." in text
        assert "search" in text
        assert "Found it." in text

    def test_long_output_truncated(self, renderer, output):
        """Tool output > 10 lines shows count and first 5."""
        long_output = "\n".join(f"line {i}" for i in range(50))
        event = ToolCallEvent(
            tool_name="bash",
            arguments={"command": "ls"},
            result=ToolResult(success=True, output=long_output),
        )
        renderer.on_tool_call(event)
        text = output.getvalue()
        assert "50 lines" in text
        assert "more lines" in text

    def test_no_output_tool(self, renderer, output):
        event = ToolCallEvent(
            tool_name="file_edit",
            arguments={"file_path": "x.py", "old_str": "a", "new_str": "b"},
            result=ToolResult(success=True, output=""),
        )
        renderer.on_tool_call(event)
        text = output.getvalue()
        assert "no output" in text

    def test_duration_shown_for_slow_tools(self, renderer, output):
        event = ToolCallEvent(
            tool_name="bash",
            arguments={"command": "sleep 2"},
            result=ToolResult(success=True, output=""),
            duration=2.1,
        )
        renderer.on_tool_call(event)
        text = output.getvalue()
        assert "2.1s" in text


class TestSummarizeArgs:
    """Test the argument summarization for different tools."""

    def test_file_read_with_range(self, renderer):
        s = renderer._summarize_args("file_read", {"file_path": "x.py", "start_line": 10, "end_line": 20})
        assert "x.py" in s
        assert "10-20" in s

    def test_bash_truncation(self, renderer):
        long_cmd = "python -m pytest tests/ -v --tb=long " + "x" * 100
        s = renderer._summarize_args("bash", {"command": long_cmd})
        assert len(s) <= 63  # 60 + "..."
        assert s.endswith("...")

    def test_search_with_glob(self, renderer):
        s = renderer._summarize_args("search", {"pattern": "def main", "file_glob": "*.py"})
        assert "def main" in s
        assert "*.py" in s
