"""Reverse direction: pull a Langfuse trace and rebuild a TrajectoryRecord.

R0 scaffold. Real implementation in W3 Day 5 — feeds
``pare.inspector.loader.load_langfuse_trace``.

This must be the inverse of :class:`pare.telemetry_langfuse.emitter.LangfuseEmitter`.
Any field added to the emitter side must be reflected here, or the
Inspector will produce different output for the same agent run depending
on whether it reads from JSONL or from Langfuse — that would silently
break product credibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pare.trajectory.schema import TrajectoryRecord

    from pare.telemetry_langfuse.config import LangfuseConfig


def fetch_trajectory(trace_id: str, config: "LangfuseConfig") -> "TrajectoryRecord":
    """Pull all spans for ``trace_id`` and reassemble a TrajectoryRecord."""
    raise NotImplementedError("W3 Day 5")
