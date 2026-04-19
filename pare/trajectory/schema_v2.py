# Vendored from research branch claude/great-carson-333acd @ 8571fb1f.
# Do not edit here — make changes on the research side and re-vendor.
"""Tool-call-centric trajectory schema (v2).

Extends the v1 schema with per-tool-call recording. The key addition is
``ToolCallEvent`` — a frozen dataclass capturing every tool invocation
during a ReAct execution loop, including turn-level ordering, target file,
and an error signal slot for downstream classification.

This module is for JSONL serialization and analysis, distinct from the
ephemeral ``ToolCallEvent`` in ``executor.py`` (which is a UI callback
object).

Schema changelog:
    v1.0 — Step-level ``StepAttempt`` only.
    v2.0 — Add ``ToolCallEvent`` sequence alongside ``StepAttempt``.
           ``TrajectoryRecord.tool_call_events`` field added.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Error signal enum — populated by error_signal_extractor (Phase 2.2).
# At recording time, the executor sets NONE or BLOCKED; the full
# classification happens in a post-processing pass.
# ---------------------------------------------------------------------------


class ErrorSignal(enum.Enum):
    """Error signal extracted from a tool call result.

    Values map to the taxonomy in plan.md §3.4.
    """

    NONE = "NONE"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    EMPTY_DIFF = "EMPTY_DIFF"
    TIMEOUT = "TIMEOUT"
    BLOCKED = "BLOCKED"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# ToolCallEvent
# ---------------------------------------------------------------------------

_TOOL_CALL_EVENT_REQUIRED = {
    "turn_id",
    "call_index_in_turn",
    "global_index",
    "tool_name",
    "result_success",
    "timestamp",
}

_TOOL_CALL_EVENT_OPTIONAL = {
    "params",
    "params_hash",
    "target_file",
    "result_content",
    "error_signal",
}


def _compute_params_hash(params: dict[str, Any]) -> str:
    """Deterministic SHA-256 prefix of JSON-serialized params."""
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_target_file(tool_name: str, params: dict[str, Any]) -> str:
    """Best-effort extraction of the primary target file from tool params.

    Returns empty string if no file can be determined.
    """
    # file_read, file_edit, file_create all use "file_path"
    if tool_name in ("file_read", "file_edit", "file_create"):
        return params.get("file_path", "")

    # search tool uses "path" (may be a directory, still useful)
    if tool_name == "search":
        return params.get("path", "")

    # bash — attempt to extract from command string (best-effort)
    if tool_name == "bash":
        cmd = params.get("command", "")
        # Common patterns: redirect targets, explicit file args
        # This is intentionally conservative — Phase 2.2 can refine
        return ""

    return ""


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A single tool invocation within a ReAct execution loop.

    Attributes:
        turn_id: Which LLM response turn this call belongs to (0-indexed).
                 All tool calls from a single LLM response share the same turn_id.
        call_index_in_turn: Position within the turn (0-indexed).
        global_index: Monotonically increasing index across the entire
                      execution (0-indexed). Supports temporal comparison
                      across turn boundaries.
        tool_name: Name of the tool invoked (e.g. "file_edit", "bash").
        params: Full tool call parameters dict.
        params_hash: SHA-256 prefix of deterministically serialized params.
                     Used for quick equality checks in recovery detection.
        target_file: Primary file targeted by this call (best-effort extraction).
        result_success: Whether the tool execution succeeded.
        result_content: Truncated tool result text (for error signal extraction).
        error_signal: Error classification. Set to NONE at recording time;
                      populated by error_signal_extractor in Phase 2.2.
        timestamp: Unix timestamp of when the tool call completed.
    """

    turn_id: int
    call_index_in_turn: int
    global_index: int
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    params_hash: str = ""
    target_file: str = ""
    result_success: bool = True
    result_content: str = ""
    error_signal: ErrorSignal = ErrorSignal.NONE
    timestamp: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        turn_id: int,
        call_index_in_turn: int,
        global_index: int,
        tool_name: str,
        params: dict[str, Any],
        result_success: bool,
        result_content: str,
        timestamp: float,
        error_signal: ErrorSignal = ErrorSignal.NONE,
    ) -> ToolCallEvent:
        """Factory that auto-computes params_hash and target_file."""
        return cls(
            turn_id=turn_id,
            call_index_in_turn=call_index_in_turn,
            global_index=global_index,
            tool_name=tool_name,
            params=dict(params),
            params_hash=_compute_params_hash(params),
            target_file=_extract_target_file(tool_name, params),
            result_success=result_success,
            result_content=result_content,
            error_signal=error_signal,
            timestamp=timestamp,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallEvent:
        """Deserialize from a JSON-compatible dict with strict validation."""
        from pare.trajectory.schema import (
            SchemaValidationError,
            _expect_keys,
        )

        _expect_keys(
            data,
            required=_TOOL_CALL_EVENT_REQUIRED,
            optional=_TOOL_CALL_EVENT_OPTIONAL,
            context="tool_call_event",
        )

        for int_field in ("turn_id", "call_index_in_turn", "global_index"):
            if not isinstance(data[int_field], int):
                raise SchemaValidationError(
                    f"tool_call_event.{int_field}: expected int"
                )

        if not isinstance(data["tool_name"], str):
            raise SchemaValidationError("tool_call_event.tool_name: expected str")
        if not isinstance(data["result_success"], bool):
            raise SchemaValidationError(
                "tool_call_event.result_success: expected bool"
            )
        if not isinstance(data["timestamp"], (int, float)):
            raise SchemaValidationError(
                "tool_call_event.timestamp: expected float"
            )

        # Parse error_signal enum
        raw_signal = data.get("error_signal", "NONE")
        if not isinstance(raw_signal, str):
            raise SchemaValidationError(
                "tool_call_event.error_signal: expected str"
            )
        try:
            error_signal = ErrorSignal(raw_signal)
        except ValueError:
            raise SchemaValidationError(
                f"tool_call_event.error_signal: unknown value '{raw_signal}'"
            )

        params = data.get("params", {})
        if not isinstance(params, dict):
            raise SchemaValidationError("tool_call_event.params: expected dict")

        return cls(
            turn_id=data["turn_id"],
            call_index_in_turn=data["call_index_in_turn"],
            global_index=data["global_index"],
            tool_name=data["tool_name"],
            params=params,
            params_hash=data.get("params_hash", _compute_params_hash(params)),
            target_file=data.get("target_file", ""),
            result_success=data["result_success"],
            result_content=data.get("result_content", ""),
            error_signal=error_signal,
            timestamp=float(data["timestamp"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "turn_id": self.turn_id,
            "call_index_in_turn": self.call_index_in_turn,
            "global_index": self.global_index,
            "tool_name": self.tool_name,
            "params": dict(self.params),
            "params_hash": self.params_hash,
            "target_file": self.target_file,
            "result_success": self.result_success,
            "result_content": self.result_content,
            "error_signal": self.error_signal.value,
            "timestamp": self.timestamp,
        }

    def temporal_key(self) -> tuple[int, int]:
        """Return (turn_id, call_index_in_turn) for temporal ordering."""
        return (self.turn_id, self.call_index_in_turn)
