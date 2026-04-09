"""Stream renderer — real-time display of LLM text and tool call events.

Consumes callbacks from the ReActExecutor (on_text_delta, on_tool_call)
and renders them to the terminal using Rich. The renderer manages a
state machine to handle interleaved text and tool calls cleanly.

Display format:
    ◆ [agent text streams here in real-time...]

    ├─ [file_read] main.py (lines 1-20)
    │  ✓ 20 lines read
    ├─ [file_edit] main.py — str_replace
    │  ✓ Applied (diff: +2, -1)
    ├─ [bash] python -m pytest
    │  ✗ Exit code: 1

    ◆ [more agent text...]
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.text import Text

from pare.agent.executor import ToolCallEvent
from pare.cli import themes


class StreamRenderer:
    """Renders agent execution to the terminal in real-time.

    Usage:
        renderer = StreamRenderer()
        result = await executor.run(
            ...,
            on_text_delta=renderer.on_text_delta,
            on_tool_call=renderer.on_tool_call,
        )
        renderer.finish()
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._in_text_block = False
        self._text_started = False

    def on_text_delta(self, text: str) -> None:
        """Handle streaming text from the LLM."""
        if not self._in_text_block:
            # Start a new text block
            if self._text_started:
                # Not the first text block — add spacing
                self.console.print()
            self.console.print(Text("◆ ", style=themes.PHASE), end="")
            self._in_text_block = True
            self._text_started = True

        # Print raw text without newline (streaming)
        self.console.print(text, end="", highlight=False)

    def on_tool_call(self, event: ToolCallEvent) -> None:
        """Handle a completed tool call event."""
        # End any in-progress text block
        if self._in_text_block:
            self.console.print()  # newline after streamed text
            self._in_text_block = False

        # Tool call header
        tool_display = self._format_tool_header(event)
        self.console.print(Text("  ├─ ", style=themes.BORDER), end="")
        self.console.print(tool_display)

        # Result or blocked
        if event.blocked_reason:
            self.console.print(
                Text("  │  ", style=themes.BORDER),
                Text(f"✗ BLOCKED: {event.blocked_reason}", style=themes.TOOL_BLOCKED),
            )
        elif event.result:
            self._render_result(event)

    def finish(self) -> None:
        """Finalize rendering — ensure clean state."""
        if self._in_text_block:
            self.console.print()  # newline after final text
            self._in_text_block = False

    def print_status(self, message: str, style: str = themes.INFO) -> None:
        """Print a status message outside of streaming."""
        self.console.print(Text(message, style=style))

    def print_error(self, message: str) -> None:
        self.console.print(Text(f"✗ {message}", style=themes.FAILURE))

    def print_success(self, message: str) -> None:
        self.console.print(Text(f"✓ {message}", style=themes.SUCCESS))

    def print_cost(self, tokens: dict[str, int], cost_usd: float | None = None) -> None:
        """Print token usage summary."""
        parts = [
            f"Input: {tokens.get('input', 0):,}",
            f"Output: {tokens.get('output', 0):,}",
        ]
        if tokens.get("cache_read", 0):
            parts.append(f"Cache read: {tokens['cache_read']:,}")
        msg = " · ".join(parts)
        if cost_usd is not None:
            msg += f" · Est. cost: ${cost_usd:.4f}"
        self.console.print(Text(msg, style=themes.COST))

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _format_tool_header(self, event: ToolCallEvent) -> Text:
        """Format the tool call header line."""
        text = Text()
        text.append("[", style=themes.BORDER)
        text.append(event.tool_name, style=themes.TOOL_NAME)
        text.append("] ", style=themes.BORDER)

        # Add concise argument summary
        summary = self._summarize_args(event.tool_name, event.arguments)
        if summary:
            text.append(summary, style="")

        if event.duration > 0.1:
            text.append(f" ({event.duration:.1f}s)", style=themes.INFO)

        return text

    def _render_result(self, event: ToolCallEvent) -> None:
        """Render tool result with truncation."""
        result = event.result
        if result is None:
            return

        prefix = Text("  │  ", style=themes.BORDER)

        if result.success:
            # Show concise output
            output = result.output.strip()
            lines = output.splitlines()
            if len(lines) == 0:
                self.console.print(prefix, Text("✓ (no output)", style=themes.TOOL_RESULT_OK))
            elif len(lines) <= 10:
                self.console.print(prefix, Text("✓", style=themes.TOOL_RESULT_OK))
                for line in lines:
                    self.console.print(
                        Text("  │  ", style=themes.BORDER),
                        escape(line),
                        highlight=False,
                    )
            else:
                self.console.print(
                    prefix,
                    Text(f"✓ ({len(lines)} lines)", style=themes.TOOL_RESULT_OK),
                )
                # Show first 5 lines
                for line in lines[:5]:
                    self.console.print(
                        Text("  │  ", style=themes.BORDER),
                        escape(line),
                        highlight=False,
                    )
                self.console.print(
                    Text("  │  ", style=themes.BORDER),
                    Text(f"  ... [{len(lines) - 5} more lines]", style=themes.INFO),
                )
        else:
            self.console.print(
                prefix,
                Text(f"✗ {result.error}", style=themes.TOOL_RESULT_ERR),
            )

    @staticmethod
    def _summarize_args(tool_name: str, args: dict) -> str:
        """Create a concise one-line summary of tool arguments."""
        match tool_name:
            case "file_read":
                path = args.get("file_path", "")
                start = args.get("start_line")
                end = args.get("end_line")
                if start and end:
                    return f"{path} (lines {start}-{end})"
                return path

            case "file_edit":
                path = args.get("file_path", "")
                old = args.get("old_str", "")
                preview = old[:40].replace("\n", "\\n")
                if len(old) > 40:
                    preview += "..."
                return f"{path} — {preview!r}"

            case "file_create":
                path = args.get("file_path", "")
                content = args.get("content", "")
                lines = content.count("\n") + 1
                return f"{path} ({lines} lines)"

            case "bash":
                cmd = args.get("command", "")
                if len(cmd) > 60:
                    return cmd[:60] + "..."
                return cmd

            case "search":
                pattern = args.get("pattern", "")
                path = args.get("path", "")
                glob = args.get("file_glob", "")
                parts = [f"/{pattern}/"]
                if path:
                    parts.append(f"in {path}")
                if glob:
                    parts.append(f"({glob})")
                return " ".join(parts)

            case _:
                # Generic: show first key=value pair
                if not args:
                    return ""
                items = list(args.items())[:2]
                return ", ".join(f"{k}={_short_val(v)}" for k, v in items)


def _short_val(v: Any) -> str:
    """Truncate a value for display."""
    s = str(v)
    if len(s) > 40:
        return s[:40] + "..."
    return s
