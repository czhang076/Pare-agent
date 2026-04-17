"""Tests for the Agent orchestrator — flat and hybrid execution modes.

Uses mock LLM + mock tools to verify the orchestrator correctly
dispatches between flat ReAct and hybrid Orient→Plan→Execute loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import patch

import pytest

from pare.agent.executor import ExecutionResult
from pare.agent.guardrails import GuardrailConfig
from pare.agent.orchestrator import Agent, AgentConfig
from pare.agent.verify import Tier2CheckResult
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
from pare.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)

_USAGE = TokenUsage(input_tokens=100, output_tokens=50)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockAgentLLM(LLMAdapter):
    """LLM that returns a predefined sequence of responses.

    The first N responses are consumed by planning (if hybrid mode).
    Remaining responses feed the executor loop.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(model="mock", profile=ModelProfile())
        self._responses = list(responses)
        self._call_count = 0

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        if self._call_count >= len(self._responses):
            return LLMResponse(
                content="Done.",
                tool_calls=[],
                stop_reason=StopReason.END_TURN,
                usage=_USAGE,
            )
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp

    async def chat_stream(self, messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError

    def count_tokens(self, messages) -> int:
        return 100


# ---------------------------------------------------------------------------
# Mock tool
# ---------------------------------------------------------------------------


class EchoTool(Tool):
    name = "echo"
    description = "Echo input"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output=f"echo: {params.get('text', '')}")


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
        usage=_USAGE,
    )


def _tool_call_response(name: str, args: dict, text: str = "") -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[ToolCallRequest(id=f"tc_{name}", name=name, arguments=args)],
        stop_reason=StopReason.TOOL_USE,
        usage=_USAGE,
    )


def _plan_response(summary: str, steps: list[dict]) -> LLMResponse:
    """Create a mock LLM response that returns a plan JSON."""
    return LLMResponse(
        content=json.dumps({"summary": summary, "steps": steps}),
        tool_calls=[],
        stop_reason=StopReason.END_TURN,
        usage=_USAGE,
    )


# ---------------------------------------------------------------------------
# Tests: Flat mode (use_planning=False)
# ---------------------------------------------------------------------------


class TestFlatMode:
    @pytest.mark.asyncio
    async def test_flat_run_simple(self, tmp_path: Path):
        """Flat mode: LLM responds without tool calls."""
        llm = MockAgentLLM([_text_response("Nothing to fix.")])
        config = AgentConfig(git_checkpoint=False, use_planning=False)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Fix typo")

        assert result.success is True
        assert result.output == "Nothing to fix."
        assert len(result.attempts) == 1
        assert result.attempts[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_flat_run_with_tool(self, tmp_path: Path):
        """Flat mode: LLM calls a tool, then finishes."""
        llm = MockAgentLLM([
            _tool_call_response("echo", {"text": "hello"}),
            _text_response("Done echoing."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=False)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Echo hello")

        assert result.success is True
        assert result.tool_call_count == 1

    @pytest.mark.asyncio
    async def test_flat_tier2_failure_marks_task_failed(self, tmp_path: Path):
        llm = MockAgentLLM([_text_response("Finished.")])
        config = AgentConfig(
            git_checkpoint=False,
            use_planning=False,
            tier2_test_command="pytest -q",
        )
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        with patch(
            "pare.agent.orchestrator.run_tier2_check",
            return_value=Tier2CheckResult(
                enabled=True,
                command="pytest -q",
                passed=False,
                return_code=1,
                output="failed",
            ),
        ):
            result = await agent.run("Run task")

        assert result.success is False
        assert result.stop_reason == "tier2_failed"
        assert result.tier2_enabled is True
        assert result.tier2_pass is False


# ---------------------------------------------------------------------------
# Tests: Hybrid mode (use_planning=True)
# ---------------------------------------------------------------------------


class TestHybridMode:
    @pytest.mark.asyncio
    async def test_hybrid_single_step_plan(self, tmp_path: Path):
        """Hybrid mode: planner creates 1-step plan, executor runs it."""
        # Response sequence:
        # 1. Planner create_plan → returns plan JSON
        # 2. Executor step 1 → calls echo tool
        # 3. Executor step 1 → returns text (step complete)
        llm = MockAgentLLM([
            _plan_response("Echo task", [
                {"step_number": 1, "goal": "Echo hello", "budget": 5},
            ]),
            _tool_call_response("echo", {"text": "hello"}),
            _text_response("Step 1 done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Echo hello")

        assert result.success is True
        assert result.stop_reason == "plan_complete"
        assert result.tool_call_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_multi_step_plan(self, tmp_path: Path):
        """Hybrid mode: 2-step plan, both complete successfully."""
        llm = MockAgentLLM([
            # Planner
            _plan_response("Two step task", [
                {"step_number": 1, "goal": "First echo", "budget": 5},
                {"step_number": 2, "goal": "Second echo", "budget": 5},
            ]),
            # Step 1 executor
            _tool_call_response("echo", {"text": "first"}),
            _text_response("Step 1 done."),
            # Step 2 executor
            _tool_call_response("echo", {"text": "second"}),
            _text_response("Step 2 done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Two echoes")

        assert result.success is True
        assert result.stop_reason == "plan_complete"
        assert result.tool_call_count == 2
        assert len(result.attempts) == 2
        assert all(a["status"] == "success" for a in result.attempts)

    @pytest.mark.asyncio
    async def test_hybrid_tier2_pass_keeps_plan_success(self, tmp_path: Path):
        llm = MockAgentLLM([
            _plan_response("One step", [
                {"step_number": 1, "goal": "Echo", "budget": 5},
            ]),
            _tool_call_response("echo", {"text": "ok"}),
            _text_response("Step done."),
        ])
        config = AgentConfig(
            git_checkpoint=False,
            use_planning=True,
            tier2_test_command="pytest -q",
        )
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        with patch(
            "pare.agent.orchestrator.run_tier2_check",
            return_value=Tier2CheckResult(
                enabled=True,
                command="pytest -q",
                passed=True,
                return_code=0,
                output="ok",
            ),
        ) as mock_tier2:
            result = await agent.run("Echo")

        assert result.success is True
        assert result.stop_reason == "plan_complete"
        assert result.tier2_enabled is True
        assert result.tier2_pass is True
        mock_tier2.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_tier2_failure_stops_plan(self, tmp_path: Path):
        llm = MockAgentLLM([
            _plan_response("One step", [
                {"step_number": 1, "goal": "Echo", "budget": 5},
            ]),
            _tool_call_response("echo", {"text": "ok"}),
            _text_response("Step done."),
        ])
        config = AgentConfig(
            git_checkpoint=False,
            use_planning=True,
            tier2_test_command="pytest -q",
        )
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        with patch(
            "pare.agent.orchestrator.run_tier2_check",
            return_value=Tier2CheckResult(
                enabled=True,
                command="pytest -q",
                passed=False,
                return_code=1,
                output="failed",
            ),
        ):
            result = await agent.run("Echo")

        assert result.success is False
        assert result.stop_reason == "tier2_failed"
        assert result.tier2_enabled is True
        assert result.tier2_pass is False
        assert len(result.attempts) == 1
        assert result.attempts[0]["failure_reason"] == "tier2_failed"

    @pytest.mark.asyncio
    async def test_hybrid_stops_on_budget_exceeded_no_replan(self, tmp_path: Path):
        """Per-step guardrail ceiling is the only hard executor cap.
        Forcing `GuardrailConfig(max_tool_calls_per_step=1)` stops the step
        after one tool call with plan_failed (no replan)."""
        llm = MockAgentLLM([
            _plan_response("Tight guardrail", [
                {"step_number": 1, "goal": "Do task", "budget": 1},
                {"step_number": 2, "goal": "Verify", "budget": 5},
            ]),
            # Only one tool-call response; executor will exhaust after 1 iter.
            _tool_call_response("echo", {"text": "attempt"}),
        ])
        config = AgentConfig(
            git_checkpoint=False,
            use_planning=True,
            guardrail_config=GuardrailConfig(max_tool_calls_per_step=1),
        )
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Do something")

        assert result.success is False
        assert result.stop_reason == "plan_failed"

    @pytest.mark.asyncio
    async def test_hybrid_tight_planner_budget_does_not_hard_cap_step(self, tmp_path: Path):
        """Regression: `step.budget` is advisory — it must NOT be used as
        the executor's hard `max_iterations`. A tight planner estimate
        (budget=3) should not kill a step mid-execution when the guardrail
        ceiling allows more. Previously (Phase 3.4 floor-at-6 fix) we
        raised the effective cap to 6; now (Phase 3.6 fix) the cap is
        purely the guardrail ceiling and step.budget is a prompt signal
        only. This test verifies a step with budget=3 can use 10 tool
        calls when the guardrail allows 15.
        """
        llm = MockAgentLLM([
            _plan_response("Advisory only", [
                {"step_number": 1, "goal": "Do task", "budget": 3},
            ]),
            # 11 LLM turns (10 tool calls + final text) — would fail at any
            # cap < 11. Default per-step guardrail is 15.
            _tool_call_response("echo", {"text": "1"}),
            _tool_call_response("echo", {"text": "2"}),
            _tool_call_response("echo", {"text": "3"}),
            _tool_call_response("echo", {"text": "4"}),
            _tool_call_response("echo", {"text": "5"}),
            _tool_call_response("echo", {"text": "6"}),
            _tool_call_response("echo", {"text": "7"}),
            _tool_call_response("echo", {"text": "8"}),
            _tool_call_response("echo", {"text": "9"}),
            _tool_call_response("echo", {"text": "10"}),
            _text_response("Done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Do something")

        assert result.success is True
        assert result.stop_reason == "plan_complete"
        assert result.tool_call_count == 10

    @pytest.mark.asyncio
    async def test_step_budget_surfaces_in_prompt_only(self, tmp_path: Path):
        """`step.budget` flows into the user_message as `Expected effort:`
        text. It must NOT affect how many iterations the executor runs."""
        captured_user_messages: list[str] = []

        class CapturingLLM(MockAgentLLM):
            async def chat(self, messages, tools=None, **kwargs):
                # Capture the most recent user message content
                for m in messages:
                    if getattr(m, "role", None) == "user":
                        captured_user_messages.append(m.content)
                return await super().chat(messages, tools, **kwargs)

        llm = CapturingLLM([
            _plan_response("Signal test", [
                {"step_number": 1, "goal": "Do task", "budget": 7},
            ]),
            _text_response("Done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        await agent.run("Task")

        step_prompts = [m for m in captured_user_messages if "## Current Step" in m]
        assert step_prompts, "step prompt not captured"
        assert "Expected effort: ~7 tool calls" in step_prompts[0]

    @pytest.mark.asyncio
    async def test_hybrid_runs_tier2_on_plan_failed(self, tmp_path: Path):
        """Finalize-time tier2: when a step hits the guardrail ceiling and the
        plan fails, tier2 should still run once against the current code state
        so `verification.tier2_pass` / `tier2_command` reflect whether the
        agent's partial work passes the gold tests. This decouples subjective
        plan_complete from objective tier2_pass for research data.
        """
        llm = MockAgentLLM([
            _plan_response("Budget-exhausted step", [
                {"step_number": 1, "goal": "Do task", "budget": 5},
            ]),
            # Only 1 tool call response — guardrail cap of 1 will exhaust the step.
            _tool_call_response("echo", {"text": "attempt"}),
        ])
        config = AgentConfig(
            git_checkpoint=False,
            use_planning=True,
            tier2_test_command="pytest -q",
            guardrail_config=GuardrailConfig(max_tool_calls_per_step=1),
        )
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        with patch(
            "pare.agent.orchestrator.run_tier2_check",
            return_value=Tier2CheckResult(
                enabled=True,
                command="pytest -q",
                passed=True,
                return_code=0,
                output="ok",
            ),
        ) as mock_tier2:
            result = await agent.run("Task")

        # Step 1 budget-exhausted → plan_failed, but tier2 still ran at finalize.
        assert result.success is False
        assert result.stop_reason == "plan_failed"
        assert result.tier2_enabled is True
        # Code actually passes gold tests even though agent didn't declare done.
        assert result.tier2_pass is True
        assert result.tier2_command == "pytest -q"
        # Called exactly once — at finalize, not in the loop (step didn't succeed).
        assert mock_tier2.call_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_finalize_tier2_failure_keeps_plan_failed_reason(self, tmp_path: Path):
        """Finalize-time tier2 is a *measurement*, not a control-flow gate.
        If the step already failed with plan_failed, a failing finalize-tier2
        must NOT overwrite stop_reason to tier2_failed. The original failure
        reason is preserved; tier2_pass just records the objective state.
        """
        llm = MockAgentLLM([
            _plan_response("Budget-exhausted step", [
                {"step_number": 1, "goal": "Do task", "budget": 5},
            ]),
            _tool_call_response("echo", {"text": "attempt"}),
        ])
        config = AgentConfig(
            git_checkpoint=False,
            use_planning=True,
            tier2_test_command="pytest -q",
            guardrail_config=GuardrailConfig(max_tool_calls_per_step=1),
        )
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        with patch(
            "pare.agent.orchestrator.run_tier2_check",
            return_value=Tier2CheckResult(
                enabled=True,
                command="pytest -q",
                passed=False,
                return_code=1,
                output="failed",
            ),
        ):
            result = await agent.run("Task")

        assert result.success is False
        # Reason stays plan_failed — finalize-tier2 is measurement only.
        assert result.stop_reason == "plan_failed"
        assert result.tier2_enabled is True
        assert result.tier2_pass is False
        assert result.tier2_command == "pytest -q"

    @pytest.mark.asyncio
    async def test_hybrid_fallback_plan_on_invalid_json(self, tmp_path: Path):
        """If planner can't parse JSON, fallback single-step plan still works."""
        llm = MockAgentLLM([
            # Planner returns garbage → fallback single-step plan
            _text_response("I don't know how to plan this."),
            # Fallback step executor (budget=30)
            _tool_call_response("echo", {"text": "doing it"}),
            _text_response("Done via fallback."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Simple task")

        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: Executor max_iterations
# ---------------------------------------------------------------------------


class TestMaxIterations:
    @pytest.mark.asyncio
    async def test_max_iterations_enforced(self, tmp_path: Path):
        """ReActExecutor respects max_iterations limit."""
        from pare.agent.executor import ReActExecutor

        llm = MockAgentLLM([
            _tool_call_response("echo", {"text": "1"}),
            _tool_call_response("echo", {"text": "2"}),
            _tool_call_response("echo", {"text": "3"}),
        ])
        registry = _make_registry()
        executor = ReActExecutor(llm, registry, max_iterations=2)
        ctx = ToolContext(cwd=tmp_path, headless=True)

        result = await executor.run("system", "task", ctx)

        assert result.success is False
        assert result.stop_reason == "budget_exhausted"
        # 2 iterations means 2 LLM calls, each producing 1 tool call
        assert result.tool_call_count == 2

    @pytest.mark.asyncio
    async def test_no_max_iterations(self, tmp_path: Path):
        """Without max_iterations, uses guardrail budget only."""
        from pare.agent.executor import ReActExecutor

        llm = MockAgentLLM([
            _tool_call_response("echo", {"text": "1"}),
            _tool_call_response("echo", {"text": "2"}),
            _text_response("Done."),
        ])
        registry = _make_registry()
        executor = ReActExecutor(llm, registry, max_iterations=None)
        ctx = ToolContext(cwd=tmp_path, headless=True)

        result = await executor.run("system", "task", ctx)

        assert result.success is True
        assert result.tool_call_count == 2


# ---------------------------------------------------------------------------
# Tests: Token tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Verify that token usage is accumulated correctly across LLM calls."""

    @pytest.mark.asyncio
    async def test_flat_mode_tracks_tokens(self, tmp_path: Path):
        """Flat mode: total_usage reflects all LLM calls in the executor."""
        llm = MockAgentLLM([
            _tool_call_response("echo", {"text": "hello"}),
            _text_response("Done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=False)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Echo hello")

        # 2 LLM calls × _USAGE(100 input, 50 output) = 200 input, 100 output
        assert result.total_usage.input_tokens == 200
        assert result.total_usage.output_tokens == 100
        assert result.total_usage.total_tokens == 300

    @pytest.mark.asyncio
    async def test_flat_mode_single_call_tokens(self, tmp_path: Path):
        """Flat mode with no tool calls: single LLM call usage."""
        llm = MockAgentLLM([_text_response("Nothing to do.")])
        config = AgentConfig(git_checkpoint=False, use_planning=False)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Do nothing")

        assert result.total_usage.input_tokens == 100
        assert result.total_usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_hybrid_mode_includes_planner_tokens(self, tmp_path: Path):
        """Hybrid mode: total_usage includes planner + executor LLM calls."""
        llm = MockAgentLLM([
            # 1. Planner create_plan (1 LLM call)
            _plan_response("Simple task", [
                {"step_number": 1, "goal": "Do it", "budget": 5},
            ]),
            # 2. Executor step 1: tool call + finish (2 LLM calls)
            _tool_call_response("echo", {"text": "hello"}),
            _text_response("Done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Simple task")

        # 3 LLM calls total: 1 planner + 2 executor
        # Each = _USAGE(100 input, 50 output)
        assert result.total_usage.input_tokens == 300
        assert result.total_usage.output_tokens == 150

    @pytest.mark.asyncio
    async def test_hybrid_mode_multi_step_tokens(self, tmp_path: Path):
        """Hybrid mode with 2 steps: tokens accumulate across all steps + planner."""
        llm = MockAgentLLM([
            # Planner (1 call)
            _plan_response("Two steps", [
                {"step_number": 1, "goal": "First", "budget": 5},
                {"step_number": 2, "goal": "Second", "budget": 5},
            ]),
            # Step 1 executor (2 calls)
            _tool_call_response("echo", {"text": "1"}),
            _text_response("Step 1 done."),
            # Step 2 executor (1 call)
            _text_response("Step 2 done."),
        ])
        config = AgentConfig(git_checkpoint=False, use_planning=True)
        agent = Agent(llm=llm, cwd=tmp_path, registry=_make_registry(), config=config)

        result = await agent.run("Two step task")

        # 4 LLM calls: 1 planner + 2 step1 + 1 step2
        assert result.total_usage.input_tokens == 400
        assert result.total_usage.output_tokens == 200

    @pytest.mark.asyncio
    async def test_executor_tracks_tokens_directly(self, tmp_path: Path):
        """ReActExecutor.run() returns accumulated total_usage."""
        from pare.agent.executor import ReActExecutor

        llm = MockAgentLLM([
            _tool_call_response("echo", {"text": "1"}),
            _tool_call_response("echo", {"text": "2"}),
            _text_response("Done."),
        ])
        registry = _make_registry()
        executor = ReActExecutor(llm, registry)
        ctx = ToolContext(cwd=tmp_path, headless=True)

        result = await executor.run("system", "task", ctx)

        # 3 LLM calls
        assert result.total_usage.input_tokens == 300
        assert result.total_usage.output_tokens == 150
