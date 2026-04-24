"""Optional LLM pre-pass: generate a markdown plan for the agent.

``run_agent`` calls :func:`run_planner` once before the main ReAct loop
(when ``LoopConfig.use_planner=True``) and injects the output into the
system prompt as a "Suggested Approach" section via
:func:`format_plan_for_system_prompt`. The flat loop itself is
unchanged — the plan is purely advisory guidance that the agent can
ignore if the task demands.

Design decisions:

- **Not in control flow.** The deleted ``pare.agent.planner`` (3-layer
  orchestrator) wrapped the plan in its own replan loop, which was the
  source of the "planner and executor disagree" bug. v2 keeps the plan
  as a frozen string with no re-entry point.
- **Fails open.** Any exception returns an empty string so the caller
  can skip the pre-pass without crashing the run. The caller must
  format with :func:`format_plan_for_system_prompt` to get a no-op
  when the plan is empty.
- **Low temperature, short cap.** The plan is a hint, not a spec; we
  want it stable and short so the agent doesn't treat it as gospel.
"""

from __future__ import annotations

import logging

from pare.llm.base import LLMAdapter, Message

logger = logging.getLogger(__name__)


PLANNER_SYSTEM_PROMPT = """\
You are a senior engineer about to hand a bug-fix task to a junior agent.
Produce a concise markdown plan (≤ 200 words) that:

1. Identifies where in the codebase the likely fix lives (file + rough area).
   If unsure, say "search for X first" — do NOT invent file names.
2. Lists 2–5 concrete steps the agent should try, in order.
3. Flags one "if this fails, try X" fallback.

The agent has these tools: file_read, file_edit, file_create, bash
(pytest, grep, git), search, declare_done. Do NOT write the fix yourself
— just the plan. Keep lines short; the plan will be appended to an
already-long system prompt.
"""


async def run_planner(
    *,
    llm: LLMAdapter,
    task: str,
    repo_context: str = "",
    instance_id: str = "",
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> str:
    """Run a one-shot LLM call to produce a markdown plan.

    Returns the raw markdown string (stripped). On any LLM error,
    returns an empty string and logs a warning — the caller should
    proceed without a plan rather than failing the whole run.

    ``repo_context``, when provided, is the markdown blob produced by
    :func:`pare.agent.orient_v2.run_orient`. It is injected into the
    user message so the planner can cite real files instead of
    hallucinating paths.
    """
    user_parts = [f"Task: {task}"]
    if repo_context.strip():
        user_parts.append(f"\nRepository context:\n{repo_context}")
    user_content = "\n".join(user_parts)

    messages = [
        Message(role="system", content=PLANNER_SYSTEM_PROMPT),
        Message(role="user", content=user_content),
    ]

    try:
        response = await llm.chat(
            messages=messages,
            tools=None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.warning(
            "planner_v2: LLM call failed for %s: %s",
            instance_id or "unknown",
            e,
        )
        return ""

    plan = (response.content or "").strip()
    if not plan:
        logger.info(
            "planner_v2: empty plan for %s", instance_id or "unknown"
        )
        return ""
    logger.info(
        "planner_v2: generated plan (%d chars) for %s",
        len(plan),
        instance_id or "unknown",
    )
    return plan


def format_plan_for_system_prompt(plan: str) -> str:
    """Wrap ``plan`` in a labelled markdown section for the system prompt.

    Empty plans return an empty string so the caller can safely
    concatenate without leaving a stub section header behind. Non-empty
    plans get a header + one-line disclaimer reminding the agent the
    plan is advisory.
    """
    if not plan.strip():
        return ""
    return (
        "\n\n## Suggested Approach (planner_v2 pre-pass)\n\n"
        "Use this as orientation only — revise it as you learn more.\n\n"
        + plan.strip()
        + "\n"
    )
