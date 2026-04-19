"""Emit ToolCallEvent instances as Langfuse spans.

R0 scaffold. Real implementation in W3 Day 1-2.

Every ``ToolCallEvent`` becomes one Langfuse observation (span) under a
trace keyed by ``trajectory_id``. The emitter is fire-and-forget from the
agent's perspective — buffering and retry happen inside the Langfuse SDK.

Plan §2.3 RFC contract: each span carries ``metadata.agent_status``
in ``{"success", "dead_end", "retrying", "error"}`` (computed by
:mod:`pare.telemetry_langfuse.agent_status`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pare_research.trajectory.schema_v2 import ToolCallEvent  # type: ignore[import-not-found]

    from pare.telemetry_langfuse.config import LangfuseConfig


class LangfuseEmitter:
    """Stateful emitter scoped to a single trajectory_id.

    Usage (from a future LoopConfig hook):

        emitter = LangfuseEmitter.for_trajectory(trajectory_id="...", config=...)
        # then in LoopConfig: telemetry_emit=emitter.emit_event
        ...
        emitter.flush()  # at end of run_agent
    """

    @classmethod
    def for_trajectory(
        cls, trajectory_id: str, config: "LangfuseConfig"
    ) -> "LangfuseEmitter":
        raise NotImplementedError("W3 Day 1")

    def emit_event(self, event: "ToolCallEvent") -> None:
        """Convert a ToolCallEvent to a Langfuse span and queue it."""
        raise NotImplementedError("W3 Day 1")

    def flush(self) -> None:
        """Block until the SDK has drained its background queue."""
        raise NotImplementedError("W3 Day 1")
