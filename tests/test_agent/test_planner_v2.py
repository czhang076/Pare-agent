"""Unit tests for :mod:`pare.agent.planner_v2`.

Covers three contracts the pre-pass must honour:

1. ``run_planner`` returns the LLM's ``response.content`` stripped, with
   instance_id / repo_context properly threaded into the user message.
2. LLM failures are swallowed — ``run_planner`` returns ``""`` and never
   raises (flat loop must survive pre-pass errors).
3. ``format_plan_for_system_prompt`` returns ``""`` on empty / whitespace
   input so callers can unconditionally concatenate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pare.agent.planner_v2 import (
    format_plan_for_system_prompt,
    run_planner,
)
from pare.llm.base import (
    LLMAdapter,
    LLMResponse,
    ModelProfile,
    StopReason as LLMStopReason,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Fake adapter — records each call's messages so tests can assert on prompt
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Call:
    messages: list
    tools: Any
    temperature: float | None
    max_tokens: int | None


class ScriptedLLM(LLMAdapter):
    """LLM that returns a scripted string on each ``chat`` call."""

    def __init__(self, reply: str = "", raises: Exception | None = None) -> None:
        super().__init__(model="fake", profile=ModelProfile(), temperature=0.0)
        self._reply = reply
        self._raises = raises
        self.calls: list[_Call] = []

    async def chat(
        self,
        messages,
        tools=None,
        *,
        temperature=None,
        max_tokens=None,
    ) -> LLMResponse:
        self.calls.append(
            _Call(
                messages=list(messages),
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if self._raises is not None:
            raise self._raises
        return LLMResponse(
            content=self._reply,
            tool_calls=[],
            stop_reason=LLMStopReason.END_TURN,
            usage=TokenUsage(input_tokens=5, output_tokens=50),
        )

    async def chat_stream(self, messages, tools=None, *, temperature=None, max_tokens=None):  # pragma: no cover
        raise NotImplementedError

    def count_tokens(self, messages):  # pragma: no cover
        return 0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_planner_returns_stripped_reply() -> None:
    llm = ScriptedLLM(reply="  ## Plan\n\n1. Do thing.  \n")
    out = await run_planner(llm=llm, task="fix the bug in foo.py")
    assert out == "## Plan\n\n1. Do thing."


@pytest.mark.asyncio
async def test_run_planner_threads_repo_context_into_user_message() -> None:
    """Repo context must reach the user message or the planner hallucinates paths."""
    llm = ScriptedLLM(reply="plan")
    await run_planner(
        llm=llm,
        task="T",
        repo_context="### Repo map\n- foo.py\n- bar.py",
        instance_id="swe-42",
    )
    assert len(llm.calls) == 1
    user_msg = llm.calls[0].messages[1]
    assert user_msg.role == "user"
    assert "Task: T" in user_msg.content
    assert "Repository context" in user_msg.content
    assert "foo.py" in user_msg.content


@pytest.mark.asyncio
async def test_run_planner_passes_temperature_and_max_tokens() -> None:
    llm = ScriptedLLM(reply="plan")
    await run_planner(llm=llm, task="T", temperature=0.5, max_tokens=400)
    assert llm.calls[0].temperature == 0.5
    assert llm.calls[0].max_tokens == 400


# ---------------------------------------------------------------------------
# Fail-open behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_planner_returns_empty_on_llm_exception() -> None:
    """Plan errors must not break ``run_agent``'s pre-pass."""
    llm = ScriptedLLM(raises=RuntimeError("429 rate limited"))
    out = await run_planner(llm=llm, task="T")
    assert out == ""


@pytest.mark.asyncio
async def test_run_planner_returns_empty_on_whitespace_reply() -> None:
    llm = ScriptedLLM(reply="   \n\n  ")
    out = await run_planner(llm=llm, task="T")
    assert out == ""


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def test_format_plan_for_empty_returns_empty() -> None:
    assert format_plan_for_system_prompt("") == ""
    assert format_plan_for_system_prompt("   \n  ") == ""


def test_format_plan_for_non_empty_wraps_with_section_header() -> None:
    out = format_plan_for_system_prompt("1. Check foo.py\n2. Run pytest")
    assert "Suggested Approach" in out
    assert "1. Check foo.py" in out
    # One-line disclaimer so the agent treats the plan as advisory.
    assert "revise it" in out.lower()
