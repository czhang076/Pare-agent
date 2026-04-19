"""Tests for the Langfuse emitter — W3 deliverable.

R0: skipped placeholders, real tests land W3 Day 1-2 against a Langfuse
mock server. The roundtrip test (emit → loader.fetch_trajectory) is the
critical one — it protects the contract that Inspector reports look the
same regardless of source (JSONL vs Langfuse).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="emitter implementation lands W3 Day 1-2")


def test_emit_event_creates_one_span_per_tool_call() -> None:
    """One ToolCallEvent → one Langfuse observation."""


def test_agent_status_metadata_attached() -> None:
    """Each span carries metadata.agent_status in {success, dead_end, retrying, error}."""


def test_emit_then_fetch_roundtrip_preserves_event_order() -> None:
    """fetch_trajectory(trace_id) returns a TrajectoryRecord whose
    tool_call_events match the emitted ones in the same global_index order."""
