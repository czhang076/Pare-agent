"""Agent orchestrator — top-level entry point that wires everything together.

For Phase 1 (MVP), this is a simple wrapper around ReActExecutor with a
system prompt. No Orient/Plan phases yet — those come in Phase 2.

The orchestrator handles:
- Creating the LLM adapter, tool registry, and guardrails
- Constructing the system prompt
- Running the ReAct loop
- Returning the result

Phase 2 will add:
- Orient phase (repo scanning)
- Plan phase (LLM-generated structured plan)
- Execute with per-step bounded ReAct
- Replan on failure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from forge.agent.executor import (
    ExecutionResult,
    OnTextDelta,
    OnToolCall,
    ReActExecutor,
)
from forge.agent.guardrails import GuardrailConfig, Guardrails
from forge.llm.base import LLMAdapter, Message
from forge.sandbox.git_checkpoint import GitCheckpoint, GitCheckpointError
from forge.telemetry import EventLog
from forge.tools.base import ToolContext, ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Forge, an expert coding agent. You help users with software \
engineering tasks by reading, searching, editing, and creating files \
in their project.

## Rules
- ALWAYS read a file before editing it. You must call file_read on a \
file before calling file_edit on it.
- When editing, make the smallest change that achieves the goal. Do not \
refactor surrounding code unless asked.
- After making changes, verify them (e.g., run tests if available).
- If you are unsure about something, search the codebase first.
- When you are done, explain what you changed and why.
- If you cannot complete the task, explain what you tried and what blocked you.

## Available Tools
You have access to tools for: running shell commands (bash), reading files \
(file_read), editing files (file_edit), creating new files (file_create), \
and searching code (search).
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentConfig:
    """Configuration for the agent orchestrator."""

    system_prompt: str = _SYSTEM_PROMPT
    guardrail_config: GuardrailConfig | None = None
    max_tool_result_lines: int = 200
    git_checkpoint: bool = True  # Enable git checkpoint safety net


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Agent:
    """Top-level agent that accepts a task and produces a result.

    Usage:
        from forge.llm import create_llm

        llm = create_llm("minimax", model="MiniMax-M2.5", api_key="...")
        agent = Agent(llm=llm, cwd=Path("."))
        result = await agent.run("Fix the typo in main.py")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        cwd: Path | None = None,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        event_log: EventLog | None = None,
        headless: bool = False,
    ) -> None:
        self.llm = llm
        self.cwd = (cwd or Path(".")).resolve()
        self.registry = registry or create_default_registry()
        self.config = config or AgentConfig()
        self.event_log = event_log
        self.headless = headless
        self._checkpoint: GitCheckpoint | None = None

    async def run(
        self,
        task: str,
        *,
        on_tool_call: OnToolCall | None = None,
        on_text_delta: OnTextDelta | None = None,
    ) -> ExecutionResult:
        """Run the agent on a task.

        Args:
            task: Natural language description of what to do.
            on_tool_call: Callback for each tool call (for CLI rendering).
            on_text_delta: Callback for streaming LLM text (for CLI rendering).

        Returns:
            ExecutionResult with the final state and conversation history.
        """
        self._log("agent_event", phase="start", task=task[:200])

        # Set up git checkpoint if enabled and in a git repo
        await self._setup_checkpoint()

        context = ToolContext(
            cwd=self.cwd,
            headless=self.headless,
        )

        guardrails = Guardrails(self.config.guardrail_config)

        executor = ReActExecutor(
            llm=self.llm,
            registry=self.registry,
            guardrails=guardrails,
            event_log=self.event_log,
        )

        # Append project context to system prompt
        system_prompt = self.config.system_prompt
        project_context = self._build_project_context()
        if project_context:
            system_prompt = f"{system_prompt}\n\n## Project Context\n{project_context}"

        # Pre-execution checkpoint
        if self._checkpoint:
            await self._checkpoint.checkpoint("before task execution")

        result = await executor.run(
            system_prompt=system_prompt,
            user_message=task,
            context=context,
            on_tool_call=on_tool_call,
            on_text_delta=on_text_delta,
        )

        # Post-execution: checkpoint success or rollback failure
        if self._checkpoint:
            if result.success:
                await self._checkpoint.checkpoint("task completed")
            # Don't finalize automatically in one-shot mode —
            # let the CLI decide whether to apply or discard

        self._log(
            "agent_event",
            phase="complete",
            success=result.success,
            tool_calls=result.tool_call_count,
            stop_reason=result.stop_reason,
        )

        return result

    async def chat(
        self,
        message: str,
        messages: list[Message],
        *,
        on_tool_call: OnToolCall | None = None,
        on_text_delta: OnTextDelta | None = None,
    ) -> ExecutionResult:
        """Continue an existing conversation (for interactive CLI mode).

        Unlike run(), this takes existing message history and appends the
        new user message. Used by the CLI for multi-turn conversations.
        """
        context = ToolContext(cwd=self.cwd, headless=self.headless)
        guardrails = Guardrails(self.config.guardrail_config)

        executor = ReActExecutor(
            llm=self.llm,
            registry=self.registry,
            guardrails=guardrails,
            event_log=self.event_log,
        )

        # Append user message to history
        conversation = list(messages)
        conversation.append(Message(role="user", content=message))

        return await executor.run(
            system_prompt="",  # Already in conversation history
            user_message="",
            context=context,
            messages=conversation,
            on_tool_call=on_tool_call,
            on_text_delta=on_text_delta,
        )

    async def _setup_checkpoint(self) -> None:
        """Initialize git checkpoint if enabled. Silently skips if not a git repo."""
        if not self.config.git_checkpoint or self._checkpoint is not None:
            return

        cp = GitCheckpoint(self.cwd)
        try:
            await cp.setup()
            self._checkpoint = cp
            self._log("git_checkpoint", action="setup", branch=cp.working_branch)
        except GitCheckpointError as e:
            logger.debug("Git checkpoint not available: %s", e)
            # Not a git repo or git not installed — continue without checkpoints

    async def finalize_checkpoint(self) -> str | None:
        """Squash-merge working branch back to original. Called by CLI."""
        if not self._checkpoint:
            return None
        try:
            sha = await self._checkpoint.finalize()
            self._log("git_checkpoint", action="finalize", sha=sha)
            self._checkpoint = None
            return sha
        except GitCheckpointError as e:
            logger.error("Failed to finalize checkpoint: %s", e)
            return None

    async def abort_checkpoint(self) -> None:
        """Discard all agent changes, return to original branch. Called by CLI."""
        if not self._checkpoint:
            return
        try:
            await self._checkpoint.abort()
            self._log("git_checkpoint", action="abort")
            self._checkpoint = None
        except GitCheckpointError as e:
            logger.error("Failed to abort checkpoint: %s", e)

    async def rollback_last_step(self) -> bool:
        """Roll back to the previous checkpoint. Returns True if successful."""
        if not self._checkpoint:
            return False
        try:
            await self._checkpoint.rollback()
            self._log("git_checkpoint", action="rollback")
            return True
        except GitCheckpointError as e:
            logger.error("Rollback failed: %s", e)
            return False

    @property
    def checkpoint(self) -> GitCheckpoint | None:
        return self._checkpoint

    def _build_project_context(self) -> str:
        """Build a lightweight project context string.

        This is the Phase 1 version — just the working directory and
        a file listing. Phase 2 will replace this with the Orient phase
        (repo scanner with code signatures).
        """
        parts = [f"Working directory: {self.cwd}"]

        # List top-level files (quick and cheap)
        try:
            entries = sorted(self.cwd.iterdir())
            files = [e.name for e in entries if e.is_file() and not e.name.startswith(".")]
            dirs = [
                e.name
                for e in entries
                if e.is_dir() and not e.name.startswith(".")
                and e.name not in ("node_modules", "__pycache__", ".git", "venv", ".venv")
            ]
            if dirs:
                parts.append(f"Directories: {', '.join(dirs[:20])}")
            if files:
                parts.append(f"Files: {', '.join(files[:20])}")
        except OSError:
            pass

        return "\n".join(parts)

    def _log(self, event_type: str, **data) -> None:
        if self.event_log:
            self.event_log.log(event_type, **data)
