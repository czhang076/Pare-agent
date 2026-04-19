"""Agent orchestrator — top-level entry point for headless execution.

Wires together Orient → Plan → Execute (no replan) with git checkpoints
and context management. Two execution modes:
- Flat: pure bounded ReAct (for simple tasks or ablation)
- Hybrid: Orient → Plan → Execute with per-step budgets

No interactive features — headless only.
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
from pare.agent.verify import Tier2CheckResult, run_tier2_check
from pare.context.compactor import CompactionConfig
from pare.context.manager import ContextManager
from pare.llm.base import LLMAdapter, Message, TokenUsage
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
- When you are finished, call the `declare_done` tool exactly once with \
status='fixed' (you believe your edits resolve the task), 'cannot_fix' \
(this is not addressable by a code change, or you failed to localize \
within budget), or 'need_info' (user clarification required). Simply \
stopping without calling `declare_done` is treated as a silent give-up \
and is indistinguishable from failure — always declare.
- Do NOT output text saying what you plan to do and then stop. Either \
do it (call tools) or call `declare_done` with status='cannot_fix' and \
explain in the summary.

## Available Tools
You have access to tools for: running shell commands (bash), reading files \
(file_read), editing files (file_edit), creating new files (file_create), \
searching code (search), and declaring completion (declare_done).
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
    git_checkpoint: bool = True
    use_planning: bool = False  # True = Orient → Plan → Execute
    tier2_test_command: str | None = None
    tier2_timeout_seconds: int = 300


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Agent:
    """Top-level agent that accepts a task and produces a result.

    Usage:
        from pare.llm import create_llm

        llm = create_llm("minimax", model="MiniMax-M2.5", api_key="...")
        agent = Agent(llm=llm, cwd=Path("."), headless=True)
        result = await agent.run("Fix the typo in main.py")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        cwd: Path | None = None,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        event_log: EventLog | None = None,
        headless: bool = True,
    ) -> None:
        self.llm = llm
        self.cwd = (cwd or Path(".")).resolve()
        self.registry = registry or create_default_registry()
        self.config = config or AgentConfig()
        self.event_log = event_log
        self.headless = headless
        self._checkpoint: GitCheckpoint | None = None
        self._guardrails = Guardrails(self.config.guardrail_config)
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
        """Run the agent on a task."""
        self._log("agent_event", phase="start", task=task[:200])

        await self._setup_checkpoint()

        # Orient phase (zero LLM calls)
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

        if self._checkpoint:
            await self._checkpoint.checkpoint("before task execution")

        result = await executor.run(
            system_prompt=system_prompt,
            user_message=task,
            context=context,
            on_tool_call=on_tool_call,
            on_text_delta=on_text_delta,
        )

        tier2 = Tier2CheckResult(enabled=False)
        if result.success:
            tier2 = self._run_tier2_verification()
            if tier2.enabled:
                self._log(
                    "verify",
                    check="tier2",
                    command=tier2.command,
                    passed=tier2.passed,
                    return_code=tier2.return_code,
                )

            if tier2.enabled and not tier2.passed:
                result = ExecutionResult(
                    success=False,
                    output=result.output,
                    messages=result.messages,
                    tool_call_count=result.tool_call_count,
                    stop_reason="tier2_failed",
                    total_usage=result.total_usage,
                    tier1_pass=result.tier1_pass,
                    tier2_enabled=True,
                    tier2_pass=False,
                    tier2_command=tier2.command,
                    tier2_output=tier2.output,
                    tier2_return_code=tier2.return_code,
                    tier2_error=tier2.error,
                    llm_claimed_success=result.llm_claimed_success,
                    attempts=result.attempts,
                    tool_call_events=result.tool_call_events,
                )
            else:
                result = ExecutionResult(
                    success=result.success,
                    output=result.output,
                    messages=result.messages,
                    tool_call_count=result.tool_call_count,
                    stop_reason=result.stop_reason,
                    total_usage=result.total_usage,
                    tier1_pass=result.tier1_pass,
                    tier2_enabled=tier2.enabled,
                    tier2_pass=tier2.passed if tier2.enabled else False,
                    tier2_command=tier2.command,
                    tier2_output=tier2.output,
                    tier2_return_code=tier2.return_code,
                    tier2_error=tier2.error,
                    llm_claimed_success=result.llm_claimed_success,
                    attempts=result.attempts,
                    tool_call_events=result.tool_call_events,
                )

        result = self._with_flat_attempt(task, result)

        for msg in result.messages:
            self._context.history.append(msg)

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
        """Hybrid Orient → Plan → Execute. No replan — deterministic execution."""
        planner = Planner(self.llm)
        memory_md = self._context.memory.get_content()
        plan = await planner.create_plan(task, memory_index=memory_md)
        self._log("agent_event", phase="plan", summary=plan.summary,
                  steps=len(plan.steps), complexity=plan.estimated_complexity)

        all_messages: list[Message] = []
        all_tool_call_events = []
        total_tool_calls = 0
        total_usage = TokenUsage(input_tokens=0, output_tokens=0)
        tier1_pass = True
        tier2 = Tier2CheckResult(enabled=False)
        tier2_failed = False
        llm_claimed_success = True

        for step in plan.steps:
            step.status = "in_progress"
            self._log("agent_event", phase="execute_step",
                      step=step.step_number, goal=step.goal, budget=step.budget)

            step_prompt = (
                f"{task}\n\n"
                f"## Current Plan\n{plan.to_markdown()}\n\n"
                f"## Current Step\n"
                f"Step {step.step_number}: {step.goal}\n"
                f"Expected effort: ~{step.budget} tool calls\n"
            )
            if step.success_criteria:
                step_prompt += f"Success criteria: {step.success_criteria}\n"
            if step.target_files:
                step_prompt += f"Target files: {', '.join(step.target_files)}\n"

            if self._checkpoint:
                await self._checkpoint.checkpoint(
                    f"before step {step.step_number}: {step.goal[:50]}"
                )

            context = ToolContext(cwd=self.cwd, headless=self.headless)
            self._guardrails.reset_step()

            # Budget policy: `step.budget` is a *soft* effort target,
            # surfaced to the LLM via step_prompt only. The executor does
            # NOT use it as a hard cap — that would let the planner's own
            # optimistic estimate kill its own in-progress execution
            # (observed repeatedly in pilot20 v1 and sympy20 v1 smokes:
            # planner sets budget=3-5 for "find the class", LLM does 6+
            # cautious reads, step_failed=budget_exhausted → plan_failed,
            # tier2 never runs). The real hard protection is the
            # per-step guardrail ceiling (`max_tool_calls_per_step`) which
            # still fires via `Guardrails.check_before_tool_call`, plus
            # the global `max_tool_calls` cap across the whole task.
            executor = ReActExecutor(
                llm=self.llm,
                registry=self.registry,
                guardrails=self._guardrails,
                event_log=self.event_log,
                max_iterations=self._guardrails.config.max_tool_calls_per_step,
            )

            result = await executor.run(
                system_prompt=system_prompt,
                user_message=step_prompt,
                context=context,
                on_tool_call=on_tool_call,
                on_text_delta=on_text_delta,
            )

            all_messages.extend(result.messages)
            all_tool_call_events.extend(result.tool_call_events)
            total_tool_calls += result.tool_call_count
            total_usage = total_usage + result.total_usage
            tier1_pass = tier1_pass and result.tier1_pass
            llm_claimed_success = llm_claimed_success and result.llm_claimed_success

            for msg in result.messages:
                self._context.history.append(msg)

            if result.success:
                tier2 = self._run_tier2_verification()
                if tier2.enabled:
                    self._log(
                        "verify",
                        check="tier2",
                        step=step.step_number,
                        command=tier2.command,
                        passed=tier2.passed,
                        return_code=tier2.return_code,
                    )

                    if not tier2.passed:
                        tier2_failed = True
                        step.status = "failed"
                        step.failure_reason = "tier2_failed"
                        self._log(
                            "agent_event",
                            phase="step_failed",
                            step=step.step_number,
                            reason="tier2_failed",
                            tool_calls=result.tool_call_count,
                        )
                        break

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
                # Step failed — record and stop (no replan)
                step.status = "failed"
                step.failure_reason = result.stop_reason or "unknown"
                self._log("agent_event", phase="step_failed",
                          step=step.step_number,
                          reason=step.failure_reason,
                          tool_calls=result.tool_call_count)
                break

        # Finalize-time tier2: if tier2 was never invoked by a successful
        # step (e.g., step 1 exhausted its budget and the loop broke early),
        # run it once against the current code state. This decouples
        # "agent claimed done" (plan_complete, subjective) from
        # "agent's final_diff passes gold tests" (tier2_pass, objective).
        # Without this, any plan_failed outcome silently leaves tier2_pass=False
        # with no command executed — conflating "agent didn't finish" with
        # "code doesn't work" in the research data.
        #
        # The finalize check is a *measurement*, not a control-flow gate:
        # it does NOT set `tier2_failed` (which drives stop_reason), so a
        # plan_failed run with a finalize-tier2 pass still reports
        # stop_reason="plan_failed" — both signals are preserved
        # independently for classifier consumers.
        if not tier2.enabled and self.config.tier2_test_command:
            tier2 = self._run_tier2_verification()
            if tier2.enabled:
                self._log(
                    "verify",
                    check="tier2",
                    phase="finalize",
                    command=tier2.command,
                    passed=tier2.passed,
                    return_code=tier2.return_code,
                )

        # Build final result
        success = plan.is_complete and not tier2_failed
        final_text = ""
        if all_messages:
            for msg in reversed(all_messages):
                if msg.role == "assistant" and msg.content:
                    content = msg.content
                    if isinstance(content, str):
                        final_text = content
                    elif isinstance(content, list):
                        final_text = " ".join(
                            b.text for b in content if b.text
                        )
                    break

        if self._checkpoint and success:
            await self._checkpoint.checkpoint("all steps completed")

        total_usage = total_usage + planner.total_usage

        return ExecutionResult(
            success=success,
            output=final_text,
            messages=all_messages,
            tool_call_count=total_tool_calls,
            stop_reason=(
                "plan_complete" if success
                else ("tier2_failed" if tier2_failed else "plan_failed")
            ),
            total_usage=total_usage,
            tier1_pass=tier1_pass,
            tier2_enabled=tier2.enabled,
            # tier2_pass reflects the code state alone, independent of
            # whether the agent declared plan-complete. A plan_failed run
            # whose partial final_diff happens to pass gold tests will
            # report tier2_pass=True — meaningful research signal.
            tier2_pass=tier2.enabled and tier2.passed,
            tier2_command=tier2.command,
            tier2_output=tier2.output,
            tier2_return_code=tier2.return_code,
            tier2_error=tier2.error,
            llm_claimed_success=llm_claimed_success,
            attempts=self._plan_to_attempts(plan),
            tool_call_events=all_tool_call_events,
        )

    # ------------------------------------------------------------------
    # Git checkpoint management
    # ------------------------------------------------------------------

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

    async def finalize_checkpoint(self) -> str | None:
        """Squash-merge working branch back to original."""
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
        """Discard all agent changes, return to original branch."""
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
        """Orient phase — scan the repository for context (zero LLM calls)."""
        try:
            scanner = RepoScanner(self.cwd)
            repo_ctx = await scanner.scan()

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

    def _run_tier2_verification(self) -> Tier2CheckResult:
        return run_tier2_check(
            self.cwd,
            self.config.tier2_test_command,
            timeout_seconds=self.config.tier2_timeout_seconds,
        )

    @staticmethod
    def _status_from_result(result: ExecutionResult) -> str:
        if result.success:
            return "success"
        if result.stop_reason == "budget_exhausted":
            return "budget_exhausted"
        if result.stop_reason == "error":
            return "error"
        return "failed"

    def _with_flat_attempt(self, task: str, result: ExecutionResult) -> ExecutionResult:
        if result.attempts:
            return result

        status = self._status_from_result(result)
        attempts = [
            {
                "step_number": 1,
                "attempt_number": 1,
                "goal": task[:200],
                "status": status,
                "target_files": [],
                "tool_names": [],
                "failure_reason": "" if status == "success" else result.stop_reason,
            }
        ]
        return ExecutionResult(
            success=result.success,
            output=result.output,
            messages=result.messages,
            tool_call_count=result.tool_call_count,
            stop_reason=result.stop_reason,
            total_usage=result.total_usage,
            tier1_pass=result.tier1_pass,
            tier2_enabled=result.tier2_enabled,
            tier2_pass=result.tier2_pass,
            tier2_command=result.tier2_command,
            tier2_output=result.tier2_output,
            tier2_return_code=result.tier2_return_code,
            tier2_error=result.tier2_error,
            llm_claimed_success=result.llm_claimed_success,
            attempts=attempts,
            tool_call_events=result.tool_call_events,
        )

    def _plan_to_attempts(self, plan: Plan) -> list[dict]:
        attempts: list[dict] = []
        for step in plan.steps:
            if step.status == "pending":
                continue

            status = "failed"
            if step.status == "completed":
                status = "success"
            elif step.status == "budget_exceeded" or step.failure_reason == "budget_exhausted":
                status = "budget_exhausted"
            elif step.failure_reason == "error":
                status = "error"

            attempts.append(
                {
                    "step_number": step.step_number,
                    "attempt_number": 1,
                    "goal": step.goal,
                    "status": status,
                    "target_files": list(step.target_files),
                    "tool_names": list(step.expected_tools),
                    "failure_reason": step.failure_reason,
                }
            )

        return attempts

    def _log(self, event_type: str, **data) -> None:
        if self.event_log:
            self.event_log.log(event_type, **data)
