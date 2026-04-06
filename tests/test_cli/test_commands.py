"""Tests for forge/cli/commands.py."""

from pathlib import Path
from io import StringIO

import pytest

from rich.console import Console

from forge.cli.commands import CommandHandler
from forge.telemetry import EventLog


@pytest.fixture
def console() -> Console:
    return Console(file=StringIO(), force_terminal=True)


@pytest.fixture
def event_log(tmp_path: Path) -> EventLog:
    log = EventLog(tmp_path / "test.jsonl")
    yield log
    log.close()


class TestCommandHandler:
    def test_non_command_returns_not_handled(self, console):
        handler = CommandHandler(console)
        result = handler.handle("fix the bug")
        assert result.handled is False

    def test_help_command(self, console):
        handler = CommandHandler(console)
        result = handler.handle("/help")
        assert result.handled is True
        assert not result.should_exit

    def test_exit_command(self, console):
        handler = CommandHandler(console)
        result = handler.handle("/exit")
        assert result.handled is True
        assert result.should_exit is True

    def test_quit_command(self, console):
        handler = CommandHandler(console)
        result = handler.handle("/quit")
        assert result.handled is True
        assert result.should_exit is True

    def test_clear_command(self, console):
        handler = CommandHandler(console)
        result = handler.handle("/clear")
        assert result.handled is True
        assert result.should_clear is True

    def test_unknown_command(self, console):
        handler = CommandHandler(console)
        result = handler.handle("/unknown_cmd")
        assert result.handled is True  # Handled (showed error message)
        assert not result.should_exit

    def test_cost_with_no_log(self, console):
        handler = CommandHandler(console, event_log=None)
        result = handler.handle("/cost")
        assert result.handled is True

    def test_cost_with_data(self, console, event_log):
        event_log.log("llm_response", usage={"input_tokens": 100, "output_tokens": 50})
        handler = CommandHandler(console, event_log)
        result = handler.handle("/cost")
        assert result.handled is True

    def test_history_with_no_events(self, console, event_log):
        handler = CommandHandler(console, event_log)
        result = handler.handle("/history")
        assert result.handled is True

    def test_history_with_events(self, console, event_log):
        event_log.log("tool_call", tool="bash", params={"command": "ls"})
        event_log.log("tool_call", tool="file_read", params={"file_path": "main.py"})
        handler = CommandHandler(console, event_log)
        result = handler.handle("/history")
        assert result.handled is True

    def test_case_insensitive(self, console):
        handler = CommandHandler(console)
        result = handler.handle("/HELP")
        assert result.handled is True

    def test_command_with_extra_spaces(self, console):
        handler = CommandHandler(console)
        result = handler.handle("  /help  ")
        assert result.handled is True
