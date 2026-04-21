"""R3 regression guards for ``pare.agent.loop.run_agent``.

These three tests are the minimum contract the refactor plan calls out
(`plan-curried-nygaard.md` §Validation / R3 gate):

- ``test_declared_done_exits``  — fake LLM eventually calls
  ``declare_done(status="fixed")``; assert ``stop_reason == "declared_done"``
  and ``success == True`` (no tier2 configured, so the invariant reduces to
  ``declared_status == "fixed"``).
- ``test_budget_exhausted``     — fake LLM spams ``file_read``; assert
  ``stop_reason == "budget_exhausted"`` once ``max_steps`` is hit.
- ``test_end_turn``             — fake LLM returns no ``tool_calls``;
  assert ``stop_reason == "end_turn"``.

Uses hand-rolled ``FakeLLM`` / ``FakeContainer`` rather than
``MagicMock(spec=...)`` so the tests double as documentation for the
contracts ``run_agent`` relies on (no Docker required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pytest

from pare.agent.loop import LoopConfig, LoopResult, run_agent
from pare.llm.base import (
    LLMAdapter,
    LLMResponse,
    Message,
    ModelProfile,
    StopReason as LLMStopReason,
    TokenUsage,
    ToolCallRequest,
    ToolSchema,
)
from pare.sandbox.instance_container import ExecResult
from pare.tools.base import ToolRegistry, create_default_registry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FakeContainer:
    """Minimum surface of :class:`InstanceContainer` ``run_agent`` touches.

    Covers: workdir attr, ``git_init_checkpoint``, ``git_commit``,
    ``git_diff``, ``read_file`` / ``write_file`` / ``exec`` (for the
    file_read path the budget-exhaustion test exercises).
    """

    instance_id: str = "fake"
    workdir: str = "/testbed"
    files: dict[str, str] = field(default_factory=dict)
    base_commit: str = "deadbeef"
    diff_text: str = ""

    async def git_init_checkpoint(self) -> str:
        return self.base_commit

    async def git_commit(self, message: str = "pare: agent session") -> str:
        return self.base_commit

    async def git_diff(self, base: str | None = None) -> str:
        return self.diff_text

    async def exec(self, cmd, *, timeout=60.0, cwd=None, env=None):  # pragma: no cover - not exercised
        return ExecResult(stdout="", stderr="", exit_code=0, timed_out=False)

    async def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        if path not in self.files:
            raise RuntimeError(f"file not found: {path}")
        return self.files[path]

    async def write_file(self, path: str, content: str) -> None:  # pragma: no cover
        self.files[path] = content


class FakeLLM(LLMAdapter):
    """Scripted LLMAdapter — yields a pre-built LLMResponse per call.

    When ``responses`` is exhausted it keeps returning the last scripted
    response; that's what makes the budget-exhaustion test clean — we only
    need to script one file_read reply and the loop will keep "asking" the
    LLM until ``max_steps``.
    """

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        super().__init__(model="fake", profile=ModelProfile(), temperature=0.0)
        self._queue = list(responses)
        self.call_count = 0
        self._last: LLMResponse | None = None

    async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None):
        self.call_count += 1
        if self._queue:
            self._last = self._queue.pop(0)
        assert self._last is not None, "FakeLLM has no response scripted"
        return self._last

    async def chat_stream(self, messages, tools=None, *, temperature=None, max_tokens=None):  # pragma: no cover
        raise NotImplementedError

    def count_tokens(self, messages):  # pragma: no cover
        return 0


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def _resp(
    *,
    tool_calls: list[ToolCallRequest] | None = None,
    content: str = "",
    stop_reason: LLMStopReason = LLMStopReason.TOOL_USE,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )


def _tc(tool_id: str, name: str, args: dict) -> ToolCallRequest:
    return ToolCallRequest(id=tool_id, name=name, arguments=args)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_done_exits() -> None:
    """declare_done(status='fixed') → stop_reason=declared_done, success=True."""
    llm = FakeLLM([
        _resp(
            content="looking around",
            tool_calls=[_tc("c1", "file_read", {"file_path": "a.py"})],
        ),
        _resp(
            content="all set",
            tool_calls=[
                _tc(
                    "c2",
                    "declare_done",
                    {"status": "fixed", "summary": "patched foo()"},
                )
            ],
        ),
    ])
    container = FakeContainer(files={"/testbed/a.py": "x = 1\n"})
    registry = create_default_registry()

    result = await run_agent(
        llm=llm,
        task="fix the bug in a.py",
        container=container,
        registry=registry,
        config=LoopConfig(max_steps=5),
    )

    assert isinstance(result, LoopResult)
    assert result.stop_reason == "declared_done"
    assert result.declared_status == "fixed"
    assert result.declared_summary == "patched foo()"
    # tier2 not configured → success collapses to declared_status check
    assert result.tier2_enabled is False
    assert result.success is True
    # trajectory should contain both events
    names = [e.tool_name for e in result.tool_call_events]
    assert names == ["file_read", "declare_done"]
    # declare_done's metadata lives in the *tool* layer; the loop mirrors it
    # onto LoopResult. Check that both legacy and new keys survived.
    assert result.tool_call_count == 2


@pytest.mark.asyncio
async def test_budget_exhausted() -> None:
    """max_steps reached with active tool_calls → budget_exhausted, success=False."""
    file_read_resp = _resp(
        content="reading again",
        tool_calls=[_tc("c1", "file_read", {"file_path": "a.py"})],
    )
    llm = FakeLLM([file_read_resp] * 10)  # way more than max_steps
    container = FakeContainer(files={"/testbed/a.py": "hello\n"})
    registry = create_default_registry()

    result = await run_agent(
        llm=llm,
        task="keep reading",
        container=container,
        registry=registry,
        config=LoopConfig(max_steps=3),
    )

    assert result.stop_reason == "budget_exhausted"
    assert result.declared_status == ""
    assert result.success is False
    # exactly max_steps LLM calls before we stop
    assert llm.call_count == 3
    assert result.tool_call_count == 3
    # every event should be a file_read
    assert {e.tool_name for e in result.tool_call_events} == {"file_read"}


@pytest.mark.asyncio
async def test_end_turn() -> None:
    """LLM returns no tool_calls → stop_reason=end_turn, success=False."""
    llm = FakeLLM([
        _resp(
            content="I have no edits to make.",
            tool_calls=[],
            stop_reason=LLMStopReason.END_TURN,
        ),
    ])
    container = FakeContainer()
    registry = create_default_registry()

    result = await run_agent(
        llm=llm,
        task="look but don't touch",
        container=container,
        registry=registry,
        config=LoopConfig(max_steps=5),
    )

    assert result.stop_reason == "end_turn"
    assert result.declared_status == ""
    # no declare_done → success is False regardless of tier2
    assert result.success is False
    assert result.tool_call_count == 0
    assert result.tool_call_events == []


@pytest.mark.asyncio
async def test_pre_passes_inject_into_system_prompt() -> None:
    """``use_orient`` / ``use_planner`` must augment the system prompt.

    We run the loop with both switches on and a FakeLLM that ends on turn
    1 (no tool calls). The loop's first LLM call should receive a system
    message whose content contains both the orient blob marker
    ("Repository Context") and the planner marker ("Suggested Approach").
    """

    orient_llm = FakeLLM([
        _resp(
            content="nothing to do",
            tool_calls=[],
            stop_reason=LLMStopReason.END_TURN,
        ),
    ])

    # Intercept planner_v2.run_planner by monkey-patching FakeLLM to
    # serve both calls: first is the planner's one-shot (returns a plan),
    # second is the main-loop's agent call (returns end_turn).
    planner_and_main_llm = FakeLLM([
        _resp(
            content="## Plan\n1. Check foo.py.",
            tool_calls=[],
            stop_reason=LLMStopReason.END_TURN,
        ),
        _resp(
            content="nothing to do",
            tool_calls=[],
            stop_reason=LLMStopReason.END_TURN,
        ),
    ])

    container = FakeContainer(
        files={"/testbed/README.md": "# Hello\n"},
    )

    # Monkey-patch container.exec so orient's ls / git ls-files calls
    # succeed with minimal output. FakeContainer's default exec returns
    # exit_code=0 and empty stdout — enough for the top-listing to be
    # skipped and the repo_map to produce nothing, leaving just the
    # README section.
    registry = create_default_registry()

    result = await run_agent(
        llm=planner_and_main_llm,
        task="fix it",
        container=container,
        registry=registry,
        config=LoopConfig(
            system_prompt="BASE PROMPT",
            max_steps=3,
            use_orient=True,
            use_planner=True,
        ),
    )

    # Two LLM calls total: planner pre-pass + one main-loop turn.
    assert planner_and_main_llm.call_count == 2
    # stop_reason unchanged by pre-passes.
    assert result.stop_reason == "end_turn"

    # Inspect the main-loop call's system message (messages[0] of the
    # stored conversation in the result).
    system_msg = result.messages[0]
    assert system_msg.role == "system"
    system_text = (
        system_msg.content
        if isinstance(system_msg.content, str)
        else "".join(b.text for b in system_msg.content)
    )
    assert "BASE PROMPT" in system_text
    assert "Repository Context" in system_text
    assert "Suggested Approach" in system_text
    # The actual plan text ended up in there, not a stub.
    assert "Check foo.py" in system_text


@pytest.mark.asyncio
async def test_pre_pass_failure_does_not_break_loop() -> None:
    """If orient_v2.run_orient raises, the loop must still run cleanly."""

    llm = FakeLLM([
        _resp(
            content="nothing",
            tool_calls=[],
            stop_reason=LLMStopReason.END_TURN,
        ),
    ])

    class ExplodingContainer(FakeContainer):
        async def read_file(self, path, *, max_bytes=1_000_000):
            raise RuntimeError("disk on fire")

        async def exec(self, cmd, *, timeout=60.0, cwd=None, env=None):
            raise RuntimeError("docker on fire")

    container = ExplodingContainer()
    registry = create_default_registry()

    # use_orient=True but container blows up on every read/exec. run_orient
    # catches at top level and returns "". Loop should proceed normally.
    result = await run_agent(
        llm=llm,
        task="go",
        container=container,
        registry=registry,
        config=LoopConfig(
            system_prompt="BASE",
            max_steps=2,
            use_orient=True,
        ),
    )
    assert result.stop_reason == "end_turn"
    # No crash, no pre-pass marker in system prompt — just the BASE we set.
    system_msg = result.messages[0]
    assert system_msg.role == "system"
    system_text = (
        system_msg.content
        if isinstance(system_msg.content, str)
        else "".join(b.text for b in system_msg.content)
    )
    assert "BASE" in system_text
    assert "Repository Context" not in system_text


@pytest.mark.asyncio
async def test_llm_exception_yields_error_exit() -> None:
    """Any exception from llm.chat → stop_reason='error', no tier2."""

    class BrokenLLM(FakeLLM):
        async def chat(self, messages, tools=None, *, temperature=None, max_tokens=None):
            raise RuntimeError("network down")

    llm = BrokenLLM([])
    container = FakeContainer()
    registry = create_default_registry()

    result = await run_agent(
        llm=llm,
        task="go",
        container=container,
        registry=registry,
        config=LoopConfig(max_steps=3, verify_instance_id="sympy__sympy-11618"),
    )

    assert result.stop_reason == "error"
    assert result.error is not None and "network down" in result.error
    # tier2 must be skipped on error exits (plan §R3 Risks)
    assert result.tier2_enabled is False
    assert result.success is False
