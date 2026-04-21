"""Flat ReAct loop — single authoritative writer of ``stop_reason``.

Replaces orchestrator/executor/orient/planner. The function :func:`run_agent`
owns the whole lifecycle of one instance's trajectory:

1. Start :class:`InstanceContainer` (caller responsibility — passed in).
2. Loop:
   a. Render prompt (history + guardrails advisory).
   b. Call LLM; parse tool_calls or end_turn.
   c. If no tool_calls → end_turn exit.
   d. For each tool_call: execute via :class:`ToolRegistry`, append
      :class:`ToolCallEvent` to trajectory, check declare_done signal.
   e. If declare_done fired → declared_done exit.
   f. If step_count >= max_steps → budget_exhausted exit.
3. Return :class:`LoopResult` with ONE ``stop_reason`` — no post-hoc rewriting.

All four exits funnel through ``return await _finalize(container, config, state)``.
``_finalize`` is the ONLY writer of :class:`LoopResult` — if you find yourself
constructing ``LoopResult(...)`` in the loop body, you are wrong.

Priority (highest first): ``declared_done`` > ``budget_exhausted`` > ``end_turn`` > ``error``.

Research-critical invariant, enforced in :func:`_finalize`::

    success = (declared_status == "fixed")
              AND (not tier2_enabled OR tier2_pass)

Decomposing ``success`` into ``declared_status`` and ``tier2_pass`` is what
makes Liu et al.'s C2 "Premature Success" observable: a trajectory with
``declared_status=="fixed"`` but ``tier2_pass==False`` is C2, not a clean
failure and not a success — the old orchestrator collapsed both into
``success=False`` and hid the signal.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional

from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    LLMAdapter,
    LLMResponse,
    Message,
    StopReason as LLMStopReason,
    TokenUsage,
)
from pare.sandbox.instance_container import InstanceContainer
from pare.tools.base import ToolContext, ToolRegistry, ToolResult
from pare.trajectory.error_signal_extractor import extract_error_signal
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent

logger = logging.getLogger(__name__)


StopReason = Literal["declared_done", "budget_exhausted", "end_turn", "error"]


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoopConfig:
    """Tunables passed into :func:`run_agent`.

    All fields have defaults so callers only override what they need. The
    ``use_orient`` / ``use_planner`` switches control the optional pre-passes
    that replace the deleted 3-layer architecture — keep them off by default
    to get a clean flat-ReAct baseline before ablating them back on.
    """

    system_prompt: str = ""
    max_steps: int = 50
    max_tool_result_lines: int = 200
    # Advisory nudge when the last N steps contain no file_edit/file_create.
    no_edit_nudge_after: int = 6
    # Advisory nudge when the last N steps contain file_edit(s) but zero
    # bash — i.e. the "edit blindly without testing" B2.1 Wrong-Fix signature
    # observed on sympy__sympy-12489 (50 steps, 15 edits, 0 bash, tier2=False).
    # Gated on ``use_test_nudge`` so v7/v8 baselines remain byte-identical
    # when the switch is off.
    no_test_nudge_after: int = 5
    # --- tier2 finalize hook ------------------------------------------------
    verify_instance_id: Optional[str] = None  # set → tier2 runs after loop
    checkpoint_enabled: bool = True
    # --- ablation switches (plan.md §5.4.3) ---------------------------------
    use_orient: bool = False
    use_planner: bool = False
    use_test_nudge: bool = False


@dataclass(slots=True)
class LoopResult:
    """Authoritative result of a single instance run.

    All fields are set by :func:`_finalize` — there is no post-hoc rewriting.
    See the docstring on :func:`_finalize` for the ``success`` invariant and
    why ``declared_status`` / ``tier2_pass`` are kept decomposed (C2 Premature
    Success observability).
    """

    # --- authoritative fields (single writer: _finalize) -------------------
    success: bool
    stop_reason: StopReason
    declared_status: str = ""       # "fixed" | "cannot_fix" | "need_info" | ""
    declared_summary: str = ""
    # --- counters / raw data ------------------------------------------------
    tool_call_count: int = 0
    tool_call_events: list[ToolCallEvent] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    total_usage: TokenUsage = field(
        default_factory=lambda: TokenUsage(input_tokens=0, output_tokens=0)
    )
    # --- verification signals (independent of declared_status) --------------
    tier1_pass: bool = False
    tier2_enabled: bool = False
    tier2_pass: bool = False
    tier2_output: str = ""
    final_diff: str = ""
    # --- error detail -------------------------------------------------------
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal state bag
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LoopState:
    """Mutable bag collected during the loop. Never returned directly.

    Exists ONLY so :func:`run_agent` body has one shape: mutate state, return
    via :func:`_finalize`. If you find yourself constructing ``LoopResult``
    inside ``run_agent``, go back and set state fields instead.
    """

    base_commit: str
    messages: list[Message]
    events: list[ToolCallEvent]
    global_index: int
    total_usage: TokenUsage
    tool_call_count: int = 0
    stop_reason: Optional[StopReason] = None
    declared_status: str = ""
    declared_summary: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Advisory nudge (ported from pare.agent.guardrails semantics)
# ---------------------------------------------------------------------------


_NO_EDIT_NUDGE = (
    "[advisory] You have taken several tool-using steps without editing any "
    "file. If you have gathered enough context, either propose an edit via "
    "file_edit / file_create, or call declare_done with status='cannot_fix' "
    "or 'need_info'."
)


def _maybe_nudge(events: list[ToolCallEvent], threshold: int) -> Optional[str]:
    """Return advisory text if the last ``threshold`` events had no edits."""
    if threshold <= 0 or len(events) < threshold:
        return None
    recent = events[-threshold:]
    if any(e.tool_name in ("file_edit", "file_create") for e in recent):
        return None
    return _NO_EDIT_NUDGE


_NO_TEST_NUDGE = (
    "[advisory] You have edited files recently but not run a test or script. "
    "Before editing further, invoke bash to execute the failing test "
    "(e.g. `python -m pytest <test_path> -x`) or a minimal reproducer, so "
    "you can tell whether your edits actually work. Blind editing without "
    "runtime feedback is the most common cause of wrong fixes."
)


def _maybe_test_nudge(
    events: list[ToolCallEvent], threshold: int,
) -> Optional[str]:
    """Return advisory text when the agent edits without testing.

    Fires when the last ``threshold`` events contain at least one
    ``file_edit``/``file_create`` and zero ``bash``. This is the B2.1
    "Wrong Fix" signature from sympy__sympy-12489 — the agent revises a
    file repeatedly without ever running the test to check the revision.

    Ablation-gated via ``LoopConfig.use_test_nudge`` so turning the switch
    off reproduces the v7/v8 behaviour exactly; turning it on lets us
    measure whether earlier test invocations shift trajectories off the
    B2.1 label into verified_with_recovery.
    """
    if threshold <= 0 or len(events) < threshold:
        return None
    recent = events[-threshold:]
    has_edit = any(
        e.tool_name in ("file_edit", "file_create") for e in recent
    )
    has_bash = any(e.tool_name == "bash" for e in recent)
    if has_edit and not has_bash:
        return _NO_TEST_NUDGE
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_agent(
    *,
    llm: LLMAdapter,
    task: str,
    container: InstanceContainer,
    registry: ToolRegistry,
    config: LoopConfig,
) -> LoopResult:
    """Run one instance's flat ReAct loop.

    Caller owns container lifecycle (``async with InstanceContainer.build(...)``);
    this function does not start or stop the container. Every exit path in
    the loop body sets ``state.stop_reason`` and returns via
    ``await _finalize(container, config, state)`` — this keeps ``LoopResult``
    construction and the ``success`` formula in a single place.
    """
    # ---- pre-loop: checkpoint, context, initial messages -------------------
    try:
        base_commit = (
            await container.git_init_checkpoint()
            if config.checkpoint_enabled
            else ""
        )
    except Exception as e:
        # Fall through as error — finalize will skip tier2 for stop_reason=error.
        state = _LoopState(
            base_commit="",
            messages=_build_initial_messages(config.system_prompt, task),
            events=[],
            global_index=0,
            total_usage=TokenUsage(input_tokens=0, output_tokens=0),
            stop_reason="error",
            error=f"git_init_checkpoint failed: {e}",
        )
        return await _finalize(container, config, state)

    ctx = ToolContext(
        cwd=_container_cwd(container),
        env={},
        confirmed_tools=set(),
        headless=True,
        container=container,
        exec_target="container",
    )

    # Optional pre-passes (orient + planner). They run before the main
    # loop and only mutate the system prompt — never the control flow.
    # Both fail open: if they raise or return empty, we fall through to
    # the unaugmented baseline. `use_orient` runs first so its blob can
    # feed into the planner's user message as repo_context.
    effective_system_prompt = await _run_prepasses(
        llm=llm, task=task, container=container, config=config,
    )

    state = _LoopState(
        base_commit=base_commit,
        messages=_build_initial_messages(effective_system_prompt, task),
        events=[],
        global_index=0,
        total_usage=TokenUsage(input_tokens=0, output_tokens=0),
    )

    # ---- main loop ---------------------------------------------------------
    for step in range(config.max_steps):
        # 1. optional advisory nudge (inject as a user-role system-style hint).
        # The test-nudge is gated on the ablation switch; the edit-nudge
        # fires unconditionally as it did before. Both route through the
        # same user-role injection so the LLM sees a single advisory.
        nudge = _maybe_nudge(state.events, config.no_edit_nudge_after)
        if nudge is not None:
            state.messages.append(Message(role="user", content=nudge))
        if config.use_test_nudge:
            test_nudge = _maybe_test_nudge(
                state.events, config.no_test_nudge_after,
            )
            if test_nudge is not None:
                state.messages.append(
                    Message(role="user", content=test_nudge),
                )

        # 2. LLM call — any exception → unified error exit
        try:
            response = await llm.chat(
                messages=state.messages,
                tools=registry.get_all_schemas(),
            )
        except Exception as e:
            state.stop_reason = "error"
            state.error = f"llm.chat: {e}"
            return await _finalize(container, config, state)

        state.total_usage = state.total_usage + response.usage
        state.messages.append(
            Message(role="assistant", content=_build_assistant_blocks(response))
        )

        # 3. no tool_calls → end_turn exit
        if not response.tool_calls:
            state.stop_reason = "end_turn"
            return await _finalize(container, config, state)

        # 4. execute tool calls; record events; detect declare_done
        result_blocks: list[ContentBlock] = []
        call_index_in_turn = 0

        for tc in response.tool_calls:
            # Unknown tool → record failure event, feed error back to LLM.
            if tc.name not in registry:
                error_msg = (
                    f"Unknown tool: '{tc.name}'. "
                    f"Available: {registry.tool_names}"
                )
                _append_event(
                    state,
                    turn_id=step,
                    call_index_in_turn=call_index_in_turn,
                    tool_name=tc.name,
                    params=tc.arguments,
                    success=False,
                    content=error_msg,
                    signal=ErrorSignal.OTHER,
                )
                result_blocks.append(
                    ContentBlock(
                        type=ContentBlockType.TOOL_RESULT,
                        tool_call_id=tc.id,
                        text=error_msg,
                    )
                )
                call_index_in_turn += 1
                continue

            try:
                tool = registry.get(tc.name)
                result = await registry._execute_one(tool, tc.arguments, ctx)
            except Exception as e:
                # registry._execute_one is supposed to catch inside, but
                # belt-and-suspenders: any escape becomes a unified error.
                state.stop_reason = "error"
                state.error = f"registry._execute_one({tc.name}): {e}"
                return await _finalize(container, config, state)

            state.tool_call_count += 1

            # Truncate output for transport back to the LLM
            result_text = _format_result_text(result)
            truncated = ToolResult(
                success=result.success, output=result_text
            ).truncate(max_lines=config.max_tool_result_lines)
            wire_text = truncated.output

            _append_event(
                state,
                turn_id=step,
                call_index_in_turn=call_index_in_turn,
                tool_name=tc.name,
                params=tc.arguments,
                success=result.success,
                content=wire_text[:2000],
            )

            result_blocks.append(
                ContentBlock(
                    type=ContentBlockType.TOOL_RESULT,
                    tool_call_id=tc.id,
                    text=wire_text,
                )
            )
            call_index_in_turn += 1

            # Capture declare_done signal — don't break mid-turn; finish the
            # batch so the trajectory stays consistent, then exit after the
            # tool_result message is appended below.
            if tc.name == "declare_done" and result.success:
                state.declared_status = str(
                    result.metadata.get("declared_status")
                    or result.metadata.get("status")
                    or ""
                )
                state.declared_summary = str(
                    result.metadata.get("declared_summary")
                    or result.metadata.get("summary")
                    or ""
                )

        # 5. fold tool_results back into the conversation as a tool_result msg
        state.messages.append(
            Message(role="tool_result", content=result_blocks)
        )

        # 6. declared_done → highest priority exit
        if state.declared_status:
            state.stop_reason = "declared_done"
            return await _finalize(container, config, state)

    # 7. loop exhausted → budget_exhausted
    state.stop_reason = "budget_exhausted"
    return await _finalize(container, config, state)


# ---------------------------------------------------------------------------
# Helpers (loop body only sets state; _finalize builds LoopResult)
# ---------------------------------------------------------------------------


async def _run_prepasses(
    *,
    llm: LLMAdapter,
    task: str,
    container: InstanceContainer,
    config: LoopConfig,
) -> str:
    """Run orient / planner pre-passes and return the augmented system prompt.

    Neither pre-pass is allowed to break ``run_agent``. Each is wrapped
    in a ``try/except`` that logs a warning and falls back to the
    unaugmented baseline — this preserves the "flat ReAct works on its
    own" invariant even if the upstream services (docker exec, LLM
    endpoint) misbehave. The orient blob is passed as ``repo_context``
    to the planner so the planner can cite real files instead of
    hallucinating paths.
    """
    prompt = config.system_prompt

    orient_blob = ""
    if config.use_orient:
        try:
            from pare.agent.orient_v2 import (
                format_orient_for_system_prompt,
                run_orient,
            )

            orient_blob = await run_orient(container)
            prompt += format_orient_for_system_prompt(orient_blob)
        except Exception as e:
            logger.warning("use_orient pre-pass raised, continuing: %s", e)
            orient_blob = ""

    if config.use_planner:
        try:
            from pare.agent.planner_v2 import (
                format_plan_for_system_prompt,
                run_planner,
            )

            plan = await run_planner(
                llm=llm,
                task=task,
                # Cap the context we feed the planner so it doesn't
                # dominate the prompt — the repo map can be long.
                repo_context=orient_blob[:2000] if orient_blob else "",
                instance_id=getattr(container, "instance_id", ""),
            )
            prompt += format_plan_for_system_prompt(plan)
        except Exception as e:
            logger.warning("use_planner pre-pass raised, continuing: %s", e)

    return prompt


def _build_initial_messages(system_prompt: str, task: str) -> list[Message]:
    msgs: list[Message] = []
    if system_prompt:
        msgs.append(Message(role="system", content=system_prompt))
    msgs.append(Message(role="user", content=task))
    return msgs


def _build_assistant_blocks(response: LLMResponse) -> list[ContentBlock]:
    """Convert LLMResponse to assistant message content blocks."""
    blocks: list[ContentBlock] = []
    if response.content:
        blocks.append(
            ContentBlock(type=ContentBlockType.TEXT, text=response.content)
        )
    for tc in response.tool_calls:
        blocks.append(
            ContentBlock(type=ContentBlockType.TOOL_USE, tool_call=tc)
        )
    if not blocks:
        blocks.append(ContentBlock(type=ContentBlockType.TEXT, text=""))
    return blocks


def _format_result_text(result: ToolResult) -> str:
    """Merge output + error into the text sent back to the LLM."""
    if result.success:
        return result.output or "(no output)"
    if result.output:
        return f"ERROR: {result.error}\n{result.output}"
    return f"ERROR: {result.error}"


def _container_cwd(container: InstanceContainer) -> Any:
    """Return a Path-like object for ToolContext.cwd inside the container.

    ``ToolContext.cwd`` is typed as ``pathlib.Path`` in the host code path,
    but container tools only treat it as "something we can str() and prefix".
    Use ``PurePosixPath`` so forward slashes survive on Windows hosts.
    """
    return PurePosixPath(container.workdir)


def _append_event(
    state: _LoopState,
    *,
    turn_id: int,
    call_index_in_turn: int,
    tool_name: str,
    params: dict,
    success: bool,
    content: str,
    signal: ErrorSignal = ErrorSignal.NONE,
) -> None:
    """Construct a ToolCallEvent via the canonical factory and append it.

    This is the **only** call site for ``ToolCallEvent.create`` in the new
    loop; classifier_liu / recovery_detector_v2 / sft_exporter all read
    ``TrajectoryRecord.tool_call_events`` and any drift in how events are
    built would silently miscategorise trajectories downstream. Keep the
    factory invocation local to this helper.
    """
    event = ToolCallEvent.create(
        turn_id=turn_id,
        call_index_in_turn=call_index_in_turn,
        global_index=state.global_index,
        tool_name=tool_name,
        params=params,
        result_success=success,
        result_content=content,
        timestamp=time.time(),
        error_signal=signal,
    )
    # Post-hoc classify non-BLOCKED events (BLOCKED is sticky by contract).
    if signal == ErrorSignal.NONE:
        classified = extract_error_signal(event)
        if classified != ErrorSignal.NONE:
            event = dataclasses.replace(event, error_signal=classified)
    state.events.append(event)
    state.global_index += 1


# ---------------------------------------------------------------------------
# Single writer of LoopResult
# ---------------------------------------------------------------------------


async def _finalize(
    container: InstanceContainer,
    config: LoopConfig,
    state: _LoopState,
) -> LoopResult:
    """ONLY writer of :class:`LoopResult`. ONLY place ``success`` is computed.

    Invariants this function enforces by being the single exit:

    - ``success = (declared_status == "fixed") AND
                 (not tier2_enabled OR tier2_pass)`` — computed once, for
      every ``stop_reason``.
    - Tier 2 runs whenever ``verify_instance_id`` is set AND
      ``stop_reason != "error"``, regardless of ``declared_status``. This is
      how Liu et al. C2 "Premature Success" (``declared_status=="fixed"`` AND
      ``tier2_pass==False``) stays observable.
    - ``final_diff`` is always computed (empty if no events), so downstream
      JSONL consumers don't have to handle a null field.
    """
    assert state.stop_reason is not None, "_finalize called without stop_reason"

    # Commit incremental agent work and capture the full diff against the
    # pre-loop checkpoint. Failures here shouldn't crash the result — the
    # trajectory data is still useful even if git cooperation wobbles.
    final_diff = ""
    if config.checkpoint_enabled and state.events and state.base_commit:
        try:
            await container.git_commit("pare: agent session")
        except Exception as e:
            logger.warning("git_commit failed during finalize: %s", e)
        try:
            final_diff = await container.git_diff(base=state.base_commit)
        except Exception as e:
            logger.warning("git_diff failed during finalize: %s", e)

    tier2_enabled = (
        config.verify_instance_id is not None
        and state.stop_reason != "error"
    )
    tier2_pass = False
    tier2_output = ""
    if tier2_enabled:
        try:
            from pare.sandbox.docker_eval import run_tier2_in_container

            tier2 = await run_tier2_in_container(container, final_diff)
            tier2_pass = tier2.passed
            tier2_output = tier2.output or tier2.error
        except Exception as e:
            tier2_output = f"tier2 error: {e}"

    success = (state.declared_status == "fixed") and (
        not tier2_enabled or tier2_pass
    )

    return LoopResult(
        success=success,
        stop_reason=state.stop_reason,
        declared_status=state.declared_status,
        declared_summary=state.declared_summary,
        tool_call_count=state.tool_call_count,
        tool_call_events=list(state.events),
        messages=list(state.messages),
        total_usage=state.total_usage,
        tier1_pass=bool(final_diff and final_diff.strip()),
        tier2_enabled=tier2_enabled,
        tier2_pass=tier2_pass,
        tier2_output=tier2_output,
        final_diff=final_diff,
        error=state.error,
    )
