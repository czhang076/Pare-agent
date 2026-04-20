"""DeclareDoneTool — the agent's explicit termination signal.

By default the ReAct loop ends when the LLM stops calling tools. That's a
*soft* signal — "LLM had nothing more to say" and "LLM gave up" look the
same from the outside, which is exactly the failure mode we saw on
sympy20: 11618 and 12419 burned 20 tool calls on reads and then "quietly
succeeded" without ever editing a file.

This tool, modelled on SWE-agent's ``submit`` / ``forfeit`` pair, forces
the agent to make an explicit structured claim:

- ``fixed`` — "I edited the code, I believe it resolves the task"
- ``cannot_fix`` — "this is not addressable by a code change, or I can't
  localize it within the budget; here is my analysis"
- ``need_info`` — "I need clarification from the user to proceed"
  (treated as ``cannot_fix`` in headless mode, but preserved verbatim
  in the trajectory so Module A can measure how often the agent *knows*
  it's blocked)

The executor recognises this tool by name and terminates the loop cleanly
with ``stop_reason="declared_done"``, recording the status + summary on
:class:`ExecutionResult`. This gives Module B a ground-truth self-claim
signal instead of inferring from "did the LLM stop calling tools?".
"""

from __future__ import annotations

from pare.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolResult,
)


_VALID_STATUSES: tuple[str, ...] = ("fixed", "cannot_fix", "need_info")


class DeclareDoneTool(Tool):
    name = "declare_done"
    description = (
        "Call this exactly once when you are finished with the task. "
        "Use status='fixed' if you believe your edits resolve the issue, "
        "'cannot_fix' if the task cannot be addressed with a code change "
        "or you were unable to localize it within the budget, or "
        "'need_info' if you need clarification from the user. This is the "
        "only correct way to end a session — do not simply stop calling "
        "tools, as that is indistinguishable from having given up silently."
    )
    parameters = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": list(_VALID_STATUSES),
                "description": (
                    "'fixed' = task solved by the edits you made; "
                    "'cannot_fix' = this task cannot be addressed by a "
                    "code change, or you failed to localize within budget; "
                    "'need_info' = user clarification required to proceed."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "A 1-3 sentence summary of what you did (for 'fixed') "
                    "or why you stopped (for 'cannot_fix' / 'need_info'). "
                    "Written for a human reviewer, not for further LLM use."
                ),
            },
        },
        "required": ["status", "summary"],
    }
    # Marking as READ means this tool can run alongside reads in a single
    # turn, but the executor treats it specially — it terminates the loop.
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        status = params.get("status", "")
        summary = params.get("summary", "")

        if status not in _VALID_STATUSES:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Invalid status {status!r}. Must be one of "
                    f"{list(_VALID_STATUSES)}."
                ),
            )
        if not summary or not summary.strip():
            return ToolResult(
                success=False,
                output="",
                error="summary is required and must be non-empty.",
            )

        clean_summary = summary.strip()
        return ToolResult(
            success=True,
            output=f"Session ended with status={status}. Summary: {clean_summary}",
            # Two naming conventions kept in parallel:
            # - status/summary: what the legacy executor.py reads (R5 delete);
            # - declared_status/declared_summary: what the flat ReAct loop
            #   reads in pare.agent.loop (R3). Shipping both avoids coupling
            #   this tool's lifespan to R4's default-switch timing.
            metadata={
                "status": status,
                "summary": clean_summary,
                "declared_status": status,
                "declared_summary": clean_summary,
            },
        )
