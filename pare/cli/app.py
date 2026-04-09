"""Main CLI application — interactive terminal loop.

Two modes:
1. One-shot: `pare "fix the bug in main.py"` — run task, exit.
2. Interactive: `pare` — REPL loop, multi-turn conversation.

The app wires together the Agent, StreamRenderer, and CommandHandler
into a cohesive user experience.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.text import Text

from pare.agent.orchestrator import Agent, AgentConfig
from pare.cli import themes
from pare.cli.commands import CommandHandler
from pare.cli.renderer import StreamRenderer
from pare.llm import create_llm
from pare.llm.base import LLMAdapter, Message
from pare.telemetry import EventLog

logger = logging.getLogger(__name__)


def _resolve_api_key(provider: str) -> str | None:
    """Resolve API key from environment variables."""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.environ.get(env_var)


class PareApp:
    """The main Pare CLI application.

    Usage:
        app = PareApp(provider="minimax", model="MiniMax-M2.5")
        await app.run_interactive()
        # or
        await app.run_once("fix the bug in main.py")
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.console = Console()
        self.cwd = (cwd or Path(".")).resolve()

        # Resolve API key
        resolved_key = api_key or _resolve_api_key(provider)
        if not resolved_key:
            env_var = f"{provider.upper()}_API_KEY"
            self.console.print(
                Text(f"Error: No API key found. Set {env_var} environment variable.",
                     style=themes.FAILURE)
            )
            sys.exit(1)

        # Create LLM adapter
        self.llm = create_llm(
            provider=provider,
            model=model,
            api_key=resolved_key,
            base_url=base_url,
        )

        # Session telemetry
        session_dir = self.cwd / ".pare"
        self.event_log = EventLog(session_dir / f"session_{int(time.time())}.jsonl")

        # Agent
        self.agent = Agent(
            llm=self.llm,
            cwd=self.cwd,
            event_log=self.event_log,
        )

        # CLI components
        self.renderer = StreamRenderer(self.console)
        self.commands = CommandHandler(self.console, self.event_log)

        # Conversation state
        self._messages: list[Message] = []
        self._first_turn = True

    async def run_once(self, task: str) -> int:
        """Run a single task and exit. Returns exit code (0=success)."""
        self._print_header()
        self.console.print(Text(f"Task: {task}", style=themes.HEADER))
        self.console.print()

        result = await self.agent.run(
            task,
            on_text_delta=self.renderer.on_text_delta,
            on_tool_call=self.renderer.on_tool_call,
        )

        self.renderer.finish()
        self.console.print()
        self._print_summary(result.tool_call_count, result.success)

        self.event_log.close()
        return 0 if result.success else 1

    async def run_interactive(self) -> None:
        """Run the interactive REPL loop."""
        self._print_header()
        self.console.print(
            Text("Type a task or question. Use /help for commands, /exit to quit.",
                 style=themes.INFO)
        )
        self.console.print()

        try:
            while True:
                # Get user input
                try:
                    user_input = await self._get_input()
                except (EOFError, KeyboardInterrupt):
                    self.console.print()
                    self.console.print(Text("Goodbye!", style=themes.INFO))
                    break

                if not user_input.strip():
                    continue

                # Check for slash commands
                cmd_result = self.commands.handle(user_input)
                if cmd_result.handled:
                    if cmd_result.should_exit:
                        break
                    if cmd_result.should_clear:
                        self._messages.clear()
                        self._first_turn = True
                    continue

                # Run the agent
                self.console.print()

                if self._first_turn:
                    result = await self.agent.run(
                        user_input,
                        on_text_delta=self.renderer.on_text_delta,
                        on_tool_call=self.renderer.on_tool_call,
                    )
                    self._messages = result.messages
                    self._first_turn = False
                else:
                    result = await self.agent.chat(
                        user_input,
                        self._messages,
                        on_text_delta=self.renderer.on_text_delta,
                        on_tool_call=self.renderer.on_tool_call,
                    )
                    self._messages = result.messages

                self.renderer.finish()
                self.console.print()

                if not result.success:
                    if result.stop_reason == "budget_exhausted":
                        self.console.print(
                            Text("⚠ Tool call budget exhausted.", style=themes.WARNING)
                        )
                    elif result.stop_reason == "error":
                        self.console.print(
                            Text("⚠ An error occurred. Check your API key and network.",
                                 style=themes.FAILURE)
                        )

        finally:
            self.event_log.close()

    async def _get_input(self) -> str:
        """Get user input with a styled prompt (async-safe)."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory

            if not hasattr(self, "_pt_session"):
                self._pt_session = PromptSession(history=InMemoryHistory())

            return await self._pt_session.prompt_async("pare> ")
        except ImportError:
            # Fallback to basic input if prompt_toolkit not available
            # Run in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: input("pare> "))

    def _print_header(self) -> None:
        """Print the Pare banner."""
        self.console.print(
            Text("Pare", style="bold"),
            Text(f" v{self._get_version()}", style=themes.INFO),
            Text(f" — {self.llm.model}", style=themes.INFO),
        )
        self.console.print(
            Text(f"Working directory: {self.cwd}", style=themes.INFO)
        )

    def _print_summary(self, tool_calls: int, success: bool) -> None:
        """Print execution summary."""
        tokens = self.event_log.total_tokens()
        total = tokens["input"] + tokens["output"]

        status = Text("✓ Done", style=themes.SUCCESS) if success else Text("✗ Failed", style=themes.FAILURE)
        self.console.print(status, end="")
        self.console.print(
            Text(f" · {tool_calls} tool calls · {total:,} tokens", style=themes.INFO)
        )

    @staticmethod
    def _get_version() -> str:
        try:
            from pare import __version__
            return __version__
        except ImportError:
            return "dev"
