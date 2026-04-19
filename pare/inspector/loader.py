"""Trajectory loading — JSONL and Langfuse sources.

R0 scaffold — function signatures only.

The JSONL path consumes the research branch's
``pare.trajectory.schema_v2.TrajectoryRecord.from_json_line`` directly;
do not re-parse here. Adding a parallel parser would silently desync from
the research schema and break ``annotator`` downstream.

The Langfuse path is W3 — implemented after the emitter is wired so we
can round-trip our own traces during development.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported lazily / for typing only. Installed via the `[research]` extra.
    from pare_research.trajectory.schema_v2 import TrajectoryRecord  # type: ignore[import-not-found]


def load_jsonl(path: Path) -> list["TrajectoryRecord"]:
    """Load a JSONL file of TrajectoryRecord rows. One record per line.

    Empty lines and lines starting with ``#`` are skipped (handy for
    hand-curated fixture files). All other lines must be valid records;
    parse failure raises rather than silently dropping.
    """
    raise NotImplementedError("W1 Day 1")


def load_langfuse_trace(trace_id: str) -> "TrajectoryRecord":
    """Reconstruct a TrajectoryRecord from a Langfuse trace.

    Round-trip of :func:`pare.telemetry_langfuse.emitter.emit_event`. Pulls
    spans via the Langfuse SDK, sorts by ``observation.start_time``, maps
    each span back into a ``ToolCallEvent``. The mapping must agree
    bit-for-bit with the emitter or the inspector will produce wrong
    divergence points on Langfuse-sourced data.
    """
    raise NotImplementedError("W3 Day 1")
