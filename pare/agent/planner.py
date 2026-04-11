"""Planner — LLM-based structured plan generation.

The Plan phase takes the repository context and user task, and asks the
LLM to produce a structured JSON plan with coarse-grained steps.  Each
step has a goal, target files, expected tools, budget, and success criteria.

Plan granularity is deliberately coarse — "modify the auth module" not
"change line 47".  Fine-grained decisions happen in the Execute phase.

"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pare.llm.base import LLMAdapter, Message, TokenUsage
from pare.llm.output_parser import parse_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step in an agent plan."""

    step_number: int
    goal: str
    target_files: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    budget: int = 15
    success_criteria: str = ""

    # Runtime state (not from LLM)
    status: str = "pending"  # pending | in_progress | completed | failed | budget_exceeded
    summary: str = ""  # Filled after execution
    failure_reason: str = ""


@dataclass
class Plan:
    """A structured plan for accomplishing a task."""

    summary: str
    steps: list[PlanStep] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low | medium | high

    @property
    def current_step(self) -> PlanStep | None:
        """Return the next pending or in-progress step."""
        for step in self.steps:
            if step.status in ("pending", "in_progress"):
                return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(s.status == "completed" for s in self.steps)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    def to_markdown(self) -> str:
        """Render plan as Markdown for display or context injection."""
        lines = [f"**Plan:** {self.summary}", ""]
        for step in self.steps:
            status_icon = {
                "pending": "○",
                "in_progress": "◆",
                "completed": "✓",
                "failed": "✗",
                "budget_exceeded": "⚠",
            }.get(step.status, "?")
            lines.append(
                f"  {status_icon} Step {step.step_number}: {step.goal}"
            )
            if step.summary:
                lines.append(f"    └─ {step.summary}")
            elif step.failure_reason:
                lines.append(f"    └─ Failed: {step.failure_reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON Schema for the LLM
# ---------------------------------------------------------------------------

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "1-sentence summary of the task",
        },
        "estimated_complexity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "goal": {"type": "string"},
                    "target_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expected_tools": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "budget": {"type": "integer", "default": 15},
                    "success_criteria": {"type": "string"},
                },
                "required": ["step_number", "goal"],
            },
        },
    },
    "required": ["summary", "steps"],
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT = """\
You are Pare, an expert coding agent. You are given a repository context \
and a user task.

Your job is to create a structured plan to accomplish the task.

## Repository Context
{memory_index}

## Rules
- Output ONLY a JSON object matching the schema below. No markdown, no \
explanation, no text outside the JSON.
- Each step should be a coarse-grained goal ("modify the auth module"), \
not a fine-grained action ("change line 47").
- Steps should be ordered logically: understand first, then modify, then test.
- Estimate which files each step will need to read or modify.
- Set a realistic budget (max tool calls) for each step. Simple reads: \
3-5. Complex edits: 10-15.
- For simple tasks (single file edit, quick fix), use 1-2 steps max.
- Do NOT over-plan. If the task is simple, the plan should be simple.

## Output Schema
{plan_schema}
"""

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Generates structured plans via LLM.

    Usage:
        planner = Planner(llm)
        plan = await planner.create_plan(task, memory_index)
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self.llm = llm
        self._total_usage = TokenUsage(input_tokens=0, output_tokens=0)

    @property
    def total_usage(self) -> TokenUsage:
        """Accumulated token usage across all planner LLM calls."""
        return self._total_usage

    async def create_plan(
        self,
        task: str,
        memory_index: str = "",
    ) -> Plan:
        """Ask the LLM to create a structured plan for the task.

        Args:
            task: The user's task description.
            memory_index: Current memory index content (repo context).

        Returns:
            A Plan object. Falls back to a single-step plan on parse failure.
        """
        system = _PLAN_SYSTEM_PROMPT.format(
            memory_index=memory_index or "(no repository context available)",
            plan_schema=json.dumps(_PLAN_SCHEMA, indent=2),
        )

        try:
            response = await self.llm.chat([
                Message(role="system", content=system),
                Message(role="user", content=f"Task: {task}"),
            ])
            self._total_usage = self._total_usage + response.usage

            plan = self._parse_plan(response.content)
            logger.info(
                "Plan created: %d steps, complexity=%s",
                len(plan.steps), plan.estimated_complexity,
            )
            return plan

        except Exception as e:
            logger.warning("Plan generation failed: %s — using fallback", e)
            return self._fallback_plan(task)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_plan(self, raw: str) -> Plan:
        """Parse LLM JSON output into a Plan object.

        Uses the progressive JSON parser to handle markdown fences,
        trailing commas, and other common LLM output quirks.
        """
        data = parse_json_response(raw)
        return self._dict_to_plan(data)

    @staticmethod
    def _dict_to_plan(data: dict) -> Plan:
        """Convert a parsed dict to a Plan, with defensive defaults."""
        steps: list[PlanStep] = []
        for i, step_data in enumerate(data.get("steps", []), start=1):
            steps.append(PlanStep(
                step_number=step_data.get("step_number", i),
                goal=step_data.get("goal", f"Step {i}"),
                target_files=step_data.get("target_files", []),
                expected_tools=step_data.get("expected_tools", []),
                budget=step_data.get("budget", 15),
                success_criteria=step_data.get("success_criteria", ""),
            ))

        return Plan(
            summary=data.get("summary", ""),
            steps=steps,
            estimated_complexity=data.get("estimated_complexity", "medium"),
        )

    @staticmethod
    def _fallback_plan(task: str) -> Plan:
        """Create a simple single-step plan when LLM planning fails."""
        return Plan(
            summary=task[:200],
            steps=[
                PlanStep(
                    step_number=1,
                    goal=task[:200],
                    budget=30,
                    success_criteria="Task is complete",
                ),
            ],
            estimated_complexity="medium",
        )
