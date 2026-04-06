"""Slash command handlers for the interactive CLI.

MVP commands:
    /help     — show available commands
    /cost     — show token usage and estimated cost
    /history  — show recent tool call history
    /clear    — clear conversation and start fresh
    /exit     — quit the CLI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich.text import Text

from forge.cli import themes
from forge.telemetry import EventLog


@dataclass
class CommandResult:
    """Result of a slash command execution."""

    handled: bool = True       # Was this a recognized command?
    should_exit: bool = False  # Should the CLI exit?
    should_clear: bool = False # Should conversation be cleared?


class CommandHandler:
    """Handles slash commands typed by the user.

    Usage:
        handler = CommandHandler(console, event_log)
        result = handler.handle("/cost")
        if result.should_exit:
            break
    """

    def __init__(
        self,
        console: Console,
        event_log: EventLog | None = None,
    ) -> None:
        self.console = console
        self.event_log = event_log
        self._commands: dict[str, Callable[[list[str]], CommandResult]] = {
            "/help": self._cmd_help,
            "/cost": self._cmd_cost,
            "/history": self._cmd_history,
            "/clear": self._cmd_clear,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
        }

    def handle(self, input_text: str) -> CommandResult:
        """Try to handle input as a slash command.

        Returns CommandResult with handled=False if not a command.
        """
        stripped = input_text.strip()
        if not stripped.startswith("/"):
            return CommandResult(handled=False)

        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        handler = self._commands.get(cmd)
        if handler is None:
            self.console.print(
                Text(f"Unknown command: {cmd}. Type /help for available commands.",
                     style=themes.WARNING)
            )
            return CommandResult(handled=True)

        return handler(args)

    # -----------------------------------------------------------------------
    # Command implementations
    # -----------------------------------------------------------------------

    def _cmd_help(self, args: list[str]) -> CommandResult:
        table = Table(title="Available Commands", show_header=True, border_style=themes.BORDER)
        table.add_column("Command", style=themes.TOOL_NAME)
        table.add_column("Description")

        table.add_row("/help", "Show this help message")
        table.add_row("/cost", "Show token usage and estimated cost")
        table.add_row("/history", "Show recent tool call history")
        table.add_row("/clear", "Clear conversation and start fresh")
        table.add_row("/exit", "Quit Forge")

        self.console.print(table)
        return CommandResult()

    def _cmd_cost(self, args: list[str]) -> CommandResult:
        if not self.event_log:
            self.console.print(Text("No telemetry data available.", style=themes.INFO))
            return CommandResult()

        tokens = self.event_log.total_tokens()
        total_input = tokens["input"]
        total_output = tokens["output"]
        cache_read = tokens["cache_read"]

        table = Table(title="Token Usage", show_header=True, border_style=themes.BORDER)
        table.add_column("Metric", style=themes.HEADER)
        table.add_column("Count", justify="right", style=themes.COST)

        table.add_row("Input tokens", f"{total_input:,}")
        table.add_row("Output tokens", f"{total_output:,}")
        if cache_read:
            table.add_row("Cache read tokens", f"{cache_read:,}")
        table.add_row("Total tokens", f"{total_input + total_output:,}")

        self.console.print(table)

        # Show LLM call count
        llm_events = self.event_log.read_events("llm_response")
        self.console.print(
            Text(f"LLM calls: {len(llm_events)}", style=themes.INFO)
        )

        return CommandResult()

    def _cmd_history(self, args: list[str]) -> CommandResult:
        if not self.event_log:
            self.console.print(Text("No telemetry data available.", style=themes.INFO))
            return CommandResult()

        events = self.event_log.read_events("tool_call")
        if not events:
            self.console.print(Text("No tool calls yet.", style=themes.INFO))
            return CommandResult()

        # Show last 20
        recent = events[-20:]

        table = Table(title="Recent Tool Calls", show_header=True, border_style=themes.BORDER)
        table.add_column("#", justify="right", style=themes.INFO)
        table.add_column("Tool", style=themes.TOOL_NAME)
        table.add_column("Params")

        for i, event in enumerate(recent, start=len(events) - len(recent) + 1):
            tool = event.data.get("tool", "?")
            params = event.data.get("params", {})
            # Concise param summary
            param_str = ", ".join(
                f"{k}={str(v)[:30]}" for k, v in list(params.items())[:3]
            )
            table.add_row(str(i), tool, param_str)

        self.console.print(table)
        self.console.print(
            Text(f"Total tool calls: {len(events)}", style=themes.INFO)
        )
        return CommandResult()

    def _cmd_clear(self, args: list[str]) -> CommandResult:
        self.console.print(Text("Conversation cleared.", style=themes.SUCCESS))
        return CommandResult(should_clear=True)

    def _cmd_exit(self, args: list[str]) -> CommandResult:
        self.console.print(Text("Goodbye!", style=themes.INFO))
        return CommandResult(should_exit=True)
