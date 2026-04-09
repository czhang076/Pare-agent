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

from pare.agent.executor import (
    ExecutionResult,
    OnTextDelta,
    OnToolCall,
    ReActExecutor,
)
from pare.agent.guardrails import GuardrailConfig, Guardrails
from pare.agent.orient import RepoScanner
from pare.agent.planner import Plan, PlanStep, Planner
from pare.context.compactor import CompactionConfig
from pare.context.manager import ContextManager
from pare.llm.base import LLMAdapter, Message
from pare.sandbox.git_checkpoint import GitCheckpoint, GitCheckpointError
from pare.telemetry import EventLog
from pare.tools.base import ToolContext, ToolRegistry, create_default_registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are Pare, an expert coding agent. You help users with software \
engineering tasks by reading, searching, editing, and creating files \
in their project.

## Rules
- ALWAYS complete the task fully before stopping. Do NOT pause to ask \
the user if you should continue — just do the work. Only stop when the \
task is done or you are genuinely blocked and need clarification.
- ALWAYS read a file before editing it. You must call file_read on a \
file before calling file_edit on it.
- When editing, make the smallest change that achieves the goal. Do not \
refactor surrounding code unless asked.
- After making changes, verify them (e.g., run tests if available).
- If you are unsure about something, search the codebase first.
- When you are done, explain what you changed and why.
- If you cannot complete the task, explain what you tried and what blocked you.
- Do NOT output text saying what you plan to do and then stop. Either \
do it (call tools) or explain why you cannot.

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
    use_planning: bool = False  # Enable Orient → Plan → Execute hybrid loop
    max_replans: int = 3  # Max replans before giving up


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Agent:
    """Top-level agent that accepts a task and produces a result.

    Usage:
        from pare.llm import create_llm

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
        # Shared guardrails across chat turns — preserves read_files memory
        self._guardrails = Guardrails(self.config.guardrail_config)
        # Context manager — memory index, history, compaction
        max_ctx = getattr(llm.profile, "max_context_tokens", 128_000)
        self._context = ContextManager(
            cwd=self.cwd,
            llm=self.llm,
            compaction_config=CompactionConfig(max_context_tokens=max_ctx),
        )

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

        # Run Orient phase (zero LLM calls) to scan the repo
        project_context = await self._orient()
        system_prompt = self.config.system_prompt
        if project_context:
            system_prompt = f"{system_prompt}\n\n## Project Context\n{project_context}"
        self._context.set_system_prompt(system_prompt)
        self._context.set_task(task[:200])

        if self.config.use_planning:
            result = await self._run_hybrid(
                task, system_prompt,
                on_tool_call=on_tool_call,
                on_text_delta=on_text_delta,
            )
        else:
            result = await self._run_flat(
                task, system_prompt,
                on_tool_call=on_tool_call,
                on_text_delta=on_text_delta,
            )

        self._log(
            "agent_event",
            phase="complete",
            success=result.success,
            tool_calls=result.tool_call_count,
            stop_reason=result.stop_reason,
        )

        return result

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    async def _run_flat(
        self,
        task: str,
        system_prompt: str,
        *,
        on_tool_call: OnToolCall | None = None,
        on_text_delta: OnTextDelta | None = None,
    ) -> ExecutionResult:
        """Flat ReAct loop — no planning, just execute."""
        context = ToolContext(cwd=self.cwd, headless=self.headless)
        self._guardrails.reset_step()

        executor = ReActExecutor(
            llm=self.llm,
            registry=self.registry,
            guardrails=self._guardrails,
            event_log=self.event_log,
        )

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

        # Record conversation to history
        for msg in result.messages:
            self._context.history.append(msg)

        # Post-execution checkpoint
        if self._checkpoint and result.success:
            await self._checkpoint.checkpoint("task completed")

        return result

    async def _run_hybrid(
        self,
        task: str,
        system_prompt: str,
        *,
        on_tool_call: OnToolCall | None = None,
        on_text_delta: OnTextDelta | None = None,
    ) -> ExecutionResult:
        """Hybrid Orient → Plan → Execute loop with per-step budgets and replan."""
        # Generate plan
        planner = Planner(self.llm)
        memory_md = self._context.memory.get_content()
        plan = await planner.create_plan(task, memory_index=memory_md)
        self._log("agent_event", phase="plan", summary=plan.summary,
                  steps=len(plan.steps), complexity=plan.estimated_complexity)

        replans = 0
        all_messages: list[Message] = []
        total_tool_calls = 0

        while not plan.is_complete:
            step = plan.current_step
            if step is None:
                break

            step.status = "in_progress"
            self._log("agent_event", phase="execute_step",
                      step=step.step_number, goal=step.goal, budget=step.budget)

            # Build step-specific prompt with plan context
            step_prompt = (
                f"{task}\n\n"
                f"## Current Plan\n{plan.to_markdown()}\n\n"
                f"## Current Step\n"
                f"Step {step.step_number}: {step.goal}\n"
            )
            if step.success_criteria:
                step_prompt += f"Success criteria: {step.success_criteria}\n"
            if step.target_files:
                step_prompt += f"Target files: {', '.join(step.target_files)}\n"

            # Per-step checkpoint
            if self._checkpoint:
                await self._checkpoint.checkpoint(
                    f"before step {step.step_number}: {step.goal[:50]}"
                )

            # Execute step with its own budget
            context = ToolContext(cwd=self.cwd, headless=self.headless)
            self._guardrails.reset_step()

            executor = ReActExecutor(
                llm=self.llm,
                registry=self.registry,
                guardrails=self._guardrails,
                event_log=self.event_log,
                max_iterations=step.budget,
            )

            result = await executor.run(
                system_prompt=system_prompt,
                user_message=step_prompt,
                context=context,
                on_tool_call=on_tool_call,
                on_text_delta=on_text_delta,
            )

            all_messages.extend(result.messages)
            total_tool_calls += result.tool_call_count

            # Record to history
            for msg in result.messages:
                self._context.history.append(msg)

            if result.success:
                step.status = "completed"
                step.summary = (result.output or "")[:200]
                self._log("agent_event", phase="step_complete",
                          step=step.step_number,
                          tool_calls=result.tool_call_count)

                if self._checkpoint:
                    await self._checkpoint.checkpoint(
                        f"step {step.step_number} completed"
                    )
            else:
                # Step failed or exceeded budget
                step.status = "failed"
                step.failure_reason = result.stop_reason or "unknown"
                self._log("agent_event", phase="step_failed",
                          step=step.step_number,
                          reason=step.failure_reason,
                          tool_calls=result.tool_call_count)

                # Attempt replan
                if replans < self.config.max_replans:
                    replans += 1
                    remaining = [
                        s for s in plan.steps
                        if s.status == "pending"
                    ]
                    diff_summary = ""
                    if self._checkpoint:
                        try:
                            diff_summary = await self._checkpoint.get_full_diff()
                        except Exception:
                            pass

                    self._log("agent_event", phase="replan",
                              attempt=replans, remaining=len(remaining))

                    plan = await planner.replan(
                        failed_step=step,
                        remaining_steps=remaining,
                        diff_summary=diff_summary,
                        tool_calls_used=result.tool_call_count,
                    )
                else:
                    self._log("agent_event", phase="replan_exhausted",
                              replans=replans)
                    break

        # Build final result
        success = plan.is_complete
        final_text = ""
        if all_messages:
            # Use the last assistant message as final text
            for msg in reversed(all_messages):
                if msg.role == "assistant" and msg.content:
                    content = msg.content
                    if isinstance(content, str):
                        final_text = content
                    elif isinstance(content, list):
                        # Extract text from ContentBlocks
                        final_text = " ".join(
                            b.text for b in content if b.text
                        )
                    break

        if self._checkpoint and success:
            await self._checkpoint.checkpoint("all steps completed")

        return ExecutionResult(
            success=success,
            output=final_text,
            messages=all_messages,
            tool_call_count=total_tool_calls,
            stop_reason="plan_complete" if success else "plan_failed",
        )

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

        # Reset per-step counters but keep read_files across turns
        self._guardrails.reset_step()

        executor = ReActExecutor(
            llm=self.llm,
            registry=self.registry,
            guardrails=self._guardrails,
            event_log=self.event_log,
        )

        # Append user message to history
        conversation = list(messages)
        conversation.append(Message(role="user", content=message))

        # Check if compaction is needed before the LLM call
        self._context.messages = conversation[1:]  # Skip system prompt
        if self._context.needs_compaction():
            self._log("compaction", trigger="pre_turn")
            await self._context.compact()
            # Rebuild conversation from compacted state
            assembled = self._context.get_messages()
            conversation = assembled

        result = await executor.run(
            system_prompt="",  # Already in conversation history
            user_message="",
            context=context,
            messages=conversation,
            on_tool_call=on_tool_call,
            on_text_delta=on_text_delta,
        )

        # Record to history
        self._context.history.append(Message(role="user", content=message))
        for msg in result.messages[len(conversation):]:
            self._context.history.append(msg)

        return result

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

    @property
    def context(self) -> ContextManager:
        return self._context

    async def _orient(self) -> str:
        """Orient phase — scan the repository for context (zero LLM calls).

        Returns a Markdown string with repo structure, code signatures,
        and git status for injection into the system prompt.
        """
        try:
            scanner = RepoScanner(self.cwd)
            repo_ctx = await scanner.scan()

            # Write to memory index for persistence across turns
            md = repo_ctx.to_markdown()
            if md:
                self._context.update_memory("Repository", md)
                self._log("agent_event", phase="orient",
                          files=repo_ctx.total_files,
                          dirs=repo_ctx.total_dirs,
                          signatures=len(repo_ctx.signatures))

            return md
        except Exception as e:
            logger.warning("Orient phase failed: %s", e)
            return f"Working directory: {self.cwd}"

    def _log(self, event_type: str, **data) -> None:
        if self.event_log:
            self.event_log.log(event_type, **data)
