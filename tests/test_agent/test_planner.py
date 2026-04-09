"""Tests for the Planner — LLM-based plan generation and replanning.

Uses mock LLM responses to test parsing logic without real API calls.
"""

from __future__ import annotations

import json

import pytest

from pare.agent.planner import Plan, PlanStep, Planner
from pare.llm.base import LLMAdapter, LLMResponse, Message, StopReason, TokenUsage, ToolSchema


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockPlannerLLM(LLMAdapter):
    """LLM that returns a predetermined JSON response."""

    def __init__(self, response_json: str) -> None:
        super().__init__(model="mock")
        self._response = response_json

    async def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(
            content=self._response,
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    async def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    def count_tokens(self, messages):
        return 100


# ---------------------------------------------------------------------------
# Plan data type tests
# ---------------------------------------------------------------------------


class TestPlan:
    def test_current_step_returns_first_pending(self):
        plan = Plan(summary="test", steps=[
            PlanStep(step_number=1, goal="a", status="completed"),
            PlanStep(step_number=2, goal="b", status="pending"),
            PlanStep(step_number=3, goal="c", status="pending"),
        ])
        assert plan.current_step.step_number == 2

    def test_current_step_returns_none_when_complete(self):
        plan = Plan(summary="test", steps=[
            PlanStep(step_number=1, goal="a", status="completed"),
        ])
        assert plan.current_step is None

    def test_is_complete(self):
        plan = Plan(summary="test", steps=[
            PlanStep(step_number=1, goal="a", status="completed"),
            PlanStep(step_number=2, goal="b", status="completed"),
        ])
        assert plan.is_complete is True

    def test_is_not_complete(self):
        plan = Plan(summary="test", steps=[
            PlanStep(step_number=1, goal="a", status="completed"),
            PlanStep(step_number=2, goal="b", status="pending"),
        ])
        assert plan.is_complete is False

    def test_completed_count(self):
        plan = Plan(summary="test", steps=[
            PlanStep(step_number=1, goal="a", status="completed"),
            PlanStep(step_number=2, goal="b", status="failed"),
            PlanStep(step_number=3, goal="c", status="completed"),
        ])
        assert plan.completed_count == 2

    def test_to_markdown(self):
        plan = Plan(summary="Fix the bug", steps=[
            PlanStep(step_number=1, goal="Read the file", status="completed", summary="Done"),
            PlanStep(step_number=2, goal="Edit the code", status="in_progress"),
            PlanStep(step_number=3, goal="Run tests", status="pending"),
        ])
        md = plan.to_markdown()
        assert "Fix the bug" in md
        assert "✓" in md
        assert "◆" in md
        assert "○" in md


# ---------------------------------------------------------------------------
# Planner create_plan tests
# ---------------------------------------------------------------------------


class TestCreatePlan:
    @pytest.mark.asyncio
    async def test_creates_plan_from_json(self):
        response = json.dumps({
            "summary": "Add login feature",
            "estimated_complexity": "medium",
            "steps": [
                {
                    "step_number": 1,
                    "goal": "Read auth module",
                    "target_files": ["src/auth.py"],
                    "expected_tools": ["file_read"],
                    "budget": 5,
                    "success_criteria": "Understand current auth flow",
                },
                {
                    "step_number": 2,
                    "goal": "Add login endpoint",
                    "target_files": ["src/auth.py", "src/routes.py"],
                    "expected_tools": ["file_read", "file_edit"],
                    "budget": 15,
                    "success_criteria": "Login endpoint exists and handles POST",
                },
            ],
        })
        planner = Planner(MockPlannerLLM(response))
        plan = await planner.create_plan("Add login feature")

        assert plan.summary == "Add login feature"
        assert plan.estimated_complexity == "medium"
        assert len(plan.steps) == 2
        assert plan.steps[0].goal == "Read auth module"
        assert plan.steps[0].budget == 5
        assert plan.steps[1].target_files == ["src/auth.py", "src/routes.py"]

    @pytest.mark.asyncio
    async def test_handles_markdown_fenced_json(self):
        response = '```json\n{"summary": "Fix bug", "steps": [{"step_number": 1, "goal": "Fix it"}]}\n```'
        planner = Planner(MockPlannerLLM(response))
        plan = await planner.create_plan("Fix bug")

        assert plan.summary == "Fix bug"
        assert len(plan.steps) == 1

    @pytest.mark.asyncio
    async def test_handles_extra_text_around_json(self):
        response = 'Here is the plan:\n{"summary": "Refactor", "steps": [{"step_number": 1, "goal": "Clean up"}]}\nDone!'
        planner = Planner(MockPlannerLLM(response))
        plan = await planner.create_plan("Refactor")

        assert plan.summary == "Refactor"

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        response = "I can't create a plan because the task is unclear."
        planner = Planner(MockPlannerLLM(response))
        plan = await planner.create_plan("Do something")

        # Should fall back to single-step plan
        assert len(plan.steps) == 1
        assert "Do something" in plan.steps[0].goal

    @pytest.mark.asyncio
    async def test_defensive_defaults(self):
        response = json.dumps({
            "summary": "Minimal",
            "steps": [
                {"step_number": 1, "goal": "Do it"},  # Minimal fields
            ],
        })
        planner = Planner(MockPlannerLLM(response))
        plan = await planner.create_plan("Minimal task")

        step = plan.steps[0]
        assert step.budget == 15  # Default
        assert step.target_files == []
        assert step.expected_tools == []
        assert step.status == "pending"

    @pytest.mark.asyncio
    async def test_passes_memory_index(self):
        """Verify the memory index is included in the prompt."""
        calls: list[list[Message]] = []

        class CaptureLLM(MockPlannerLLM):
            async def chat(self, messages, tools=None, **kwargs):
                calls.append(messages)
                return await super().chat(messages, tools, **kwargs)

        response = json.dumps({"summary": "test", "steps": [{"step_number": 1, "goal": "a"}]})
        planner = Planner(CaptureLLM(response))
        await planner.create_plan("task", memory_index="## Structure\nsrc/ (3 files)")

        assert len(calls) == 1
        system_content = calls[0][0].content
        assert "src/ (3 files)" in system_content


# ---------------------------------------------------------------------------
# Replan tests
# ---------------------------------------------------------------------------


class TestReplan:
    @pytest.mark.asyncio
    async def test_replan_creates_revised_plan(self):
        response = json.dumps({
            "summary": "Revised approach",
            "steps": [
                {"step_number": 2, "goal": "Try different approach"},
                {"step_number": 3, "goal": "Run tests"},
            ],
        })
        planner = Planner(MockPlannerLLM(response))

        failed = PlanStep(step_number=2, goal="Edit auth", status="failed", failure_reason="File not found")
        remaining = [PlanStep(step_number=3, goal="Run tests")]

        plan = await planner.replan(failed, remaining, diff_summary="no changes")
        assert len(plan.steps) == 2
        assert "different approach" in plan.steps[0].goal

    @pytest.mark.asyncio
    async def test_replan_fallback_on_failure(self):
        planner = Planner(MockPlannerLLM("not valid json at all"))

        failed = PlanStep(step_number=1, goal="Step 1", status="failed")
        remaining = [PlanStep(step_number=2, goal="Step 2")]

        plan = await planner.replan(failed, remaining)
        # Should fall back to remaining steps
        assert len(plan.steps) == 1
        assert plan.steps[0].goal == "Step 2"
