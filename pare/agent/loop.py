"""Flat ReAct loop — single authoritative writer of ``stop_reason``.

R0 scaffold — signatures only. Real implementation lands in R3.

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pare.llm.base import LLMAdapter
from pare.sandbox.instance_container import InstanceContainer
from pare.tools.base import ToolRegistry
from pare.trajectory.schema_v2 import ToolCallEvent


StopReason = Literal["declared_done", "budget_exhausted", "end_turn", "error"]


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
    # --- tier2 finalize hook ------------------------------------------------
    verify_instance_id: Optional[str] = None  # set → tier2 runs after loop
    checkpoint_enabled: bool = True
    # --- ablation switches (plan.md §5.4.3) ---------------------------------
    use_orient: bool = False
    use_planner: bool = False


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
    messages: list[Any] = field(default_factory=list)  # list[Message]
    # --- verification signals (independent of declared_status) --------------
    tier1_pass: bool = False
    tier2_enabled: bool = False
    tier2_pass: bool = False
    tier2_output: str = ""
    final_diff: str = ""
    # --- error detail -------------------------------------------------------
    error: Optional[str] = None

    # Research-critical invariant (computed ONCE in _finalize):
    #   success = (declared_status == "fixed") AND (not tier2_enabled OR tier2_pass)
    # Decomposition makes the Liu et al. C2 "Premature Success" signal observable:
    #   declared_status=="fixed" AND tier2_pass==False  → C2 Premature Success
    #   declared_status=="fixed" AND tier2_pass==True   → genuine success
    #   declared_status in ("cannot_fix","need_info")   → honest give-up


async def run_agent(
    *,
    llm: LLMAdapter,
    task: str,
    container: InstanceContainer,
    registry: ToolRegistry,
    config: LoopConfig,
) -> LoopResult:
    """Run one instance's flat ReAct loop.

    See module docstring for control flow. Caller owns container lifecycle
    (``async with InstanceContainer.build(...)``); this function does not
    start or stop the container.

    R3 implementer: the loop body mutates a ``_LoopState`` dataclass and
    returns via ``await _finalize(container, config, state)`` at every exit.
    No ``LoopResult(...)`` construction in this function's body.
    """
    raise NotImplementedError("R3: flat ReAct loop; all exits via _finalize")
