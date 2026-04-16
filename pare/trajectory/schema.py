"""Strict trajectory schema for JSONL storage and exchange.

This module defines a deterministic, no-LLM data format used by the
research pipeline. Each JSONL line is one trajectory record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from pare.trajectory.schema_v2 import ToolCallEvent

SCHEMA_VERSION = "2.0"

_COMPATIBLE_VERSIONS = {"1.0", "2.0"}

_ATTEMPT_STATUSES = {"success", "failed", "budget_exhausted", "error"}


class SchemaValidationError(ValueError):
    """Raised when a trajectory payload fails strict schema validation."""


def _expect_keys(
    data: dict[str, Any],
    *,
    required: set[str],
    optional: set[str],
    context: str,
) -> None:
    unknown = set(data.keys()) - required - optional
    missing = required - set(data.keys())
    if unknown:
        raise SchemaValidationError(f"{context}: unknown keys: {sorted(unknown)}")
    if missing:
        raise SchemaValidationError(f"{context}: missing required keys: {sorted(missing)}")


def _expect_type(value: Any, expected: type, context: str) -> None:
    if not isinstance(value, expected):
        raise SchemaValidationError(
            f"{context}: expected {expected.__name__}, got {type(value).__name__}"
        )


def _expect_str_list(value: Any, context: str) -> list[str]:
    _expect_type(value, list, context)
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise SchemaValidationError(f"{context}[{i}]: expected str")
        out.append(item)
    return out


@dataclass(frozen=True, slots=True)
class TokenUsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsageSummary":
        _expect_keys(
            data,
            required={"input_tokens", "output_tokens"},
            optional={"cache_read_tokens", "cache_create_tokens", "total_tokens"},
            context="token_usage",
        )
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_create_tokens"):
            if key in data and not isinstance(data[key], int):
                raise SchemaValidationError(f"token_usage.{key}: expected int")

        if "total_tokens" in data and not isinstance(data["total_tokens"], int):
            raise SchemaValidationError("token_usage.total_tokens: expected int")

        expected_total = data["input_tokens"] + data["output_tokens"]
        if "total_tokens" in data and data["total_tokens"] != expected_total:
            raise SchemaValidationError(
                "token_usage.total_tokens: must equal input_tokens + output_tokens"
            )

        return cls(
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_create_tokens=data.get("cache_create_tokens", 0),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_create_tokens": self.cache_create_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    final_passed: bool
    tier1_pass: bool
    tier2_pass: bool
    tier2_command: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        _expect_keys(
            data,
            required={"final_passed", "tier1_pass", "tier2_pass"},
            optional={"tier2_command"},
            context="verification",
        )
        for key in ("final_passed", "tier1_pass", "tier2_pass"):
            if not isinstance(data[key], bool):
                raise SchemaValidationError(f"verification.{key}: expected bool")
        if "tier2_command" in data and not isinstance(data["tier2_command"], str):
            raise SchemaValidationError("verification.tier2_command: expected str")
        return cls(
            final_passed=data["final_passed"],
            tier1_pass=data["tier1_pass"],
            tier2_pass=data["tier2_pass"],
            tier2_command=data.get("tier2_command", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_passed": self.final_passed,
            "tier1_pass": self.tier1_pass,
            "tier2_pass": self.tier2_pass,
            "tier2_command": self.tier2_command,
        }


@dataclass(frozen=True, slots=True)
class StepAttempt:
    step_number: int
    attempt_number: int
    goal: str
    status: str
    target_files: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    failure_reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepAttempt":
        _expect_keys(
            data,
            required={"step_number", "attempt_number", "goal", "status"},
            # `rolled_back` accepted for backward compat with v1 JSONL
            # records but dropped on load — field has been removed.
            optional={"rolled_back", "target_files", "tool_names", "failure_reason"},
            context="attempt",
        )
        if not isinstance(data["step_number"], int):
            raise SchemaValidationError("attempt.step_number: expected int")
        if not isinstance(data["attempt_number"], int):
            raise SchemaValidationError("attempt.attempt_number: expected int")
        if not isinstance(data["goal"], str):
            raise SchemaValidationError("attempt.goal: expected str")
        if not isinstance(data["status"], str):
            raise SchemaValidationError("attempt.status: expected str")
        if data["status"] not in _ATTEMPT_STATUSES:
            raise SchemaValidationError(
                "attempt.status: expected one of "
                f"{sorted(_ATTEMPT_STATUSES)}, got {data['status']}"
            )
        if "failure_reason" in data and not isinstance(data["failure_reason"], str):
            raise SchemaValidationError("attempt.failure_reason: expected str")

        return cls(
            step_number=data["step_number"],
            attempt_number=data["attempt_number"],
            goal=data["goal"],
            status=data["status"],
            target_files=_expect_str_list(data.get("target_files", []), "attempt.target_files"),
            tool_names=_expect_str_list(data.get("tool_names", []), "attempt.tool_names"),
            failure_reason=data.get("failure_reason", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "attempt_number": self.attempt_number,
            "goal": self.goal,
            "status": self.status,
            "target_files": list(self.target_files),
            "tool_names": list(self.tool_names),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryRecord:
    schema_version: str
    trajectory_id: str
    instance_id: str
    task: str
    model: str
    seed: int
    created_at: float
    llm_claimed_success: bool
    verification: VerificationResult
    attempts: list[StepAttempt] = field(default_factory=list)
    tool_call_events: list[ToolCallEvent] = field(default_factory=list)
    token_usage: TokenUsageSummary = field(default_factory=TokenUsageSummary)
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryRecord":
        _expect_keys(
            data,
            required={
                "schema_version",
                "trajectory_id",
                "instance_id",
                "task",
                "model",
                "seed",
                "created_at",
                "llm_claimed_success",
                "verification",
            },
            optional={"attempts", "tool_call_events", "token_usage", "metadata"},
            context="trajectory",
        )

        if data["schema_version"] not in _COMPATIBLE_VERSIONS:
            raise SchemaValidationError(
                f"trajectory.schema_version: expected one of {sorted(_COMPATIBLE_VERSIONS)}, "
                f"got {data['schema_version']}"
            )

        for key in ("trajectory_id", "instance_id", "task", "model"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise SchemaValidationError(f"trajectory.{key}: expected non-empty str")

        if not isinstance(data["seed"], int):
            raise SchemaValidationError("trajectory.seed: expected int")
        if not isinstance(data["created_at"], (int, float)):
            raise SchemaValidationError("trajectory.created_at: expected float")
        if not isinstance(data["llm_claimed_success"], bool):
            raise SchemaValidationError("trajectory.llm_claimed_success: expected bool")

        verification = VerificationResult.from_dict(data["verification"])

        attempts_raw = data.get("attempts", [])
        _expect_type(attempts_raw, list, "trajectory.attempts")
        attempts = [StepAttempt.from_dict(item) for item in attempts_raw]

        # Tool-call-level events (v2, optional for backward compat with v1)
        from pare.trajectory.schema_v2 import ToolCallEvent as TCEvent

        tc_events_raw = data.get("tool_call_events", [])
        _expect_type(tc_events_raw, list, "trajectory.tool_call_events")
        tool_call_events = [TCEvent.from_dict(item) for item in tc_events_raw]

        token_usage_raw = data.get("token_usage")
        token_usage = (
            TokenUsageSummary.from_dict(token_usage_raw)
            if token_usage_raw is not None
            else TokenUsageSummary()
        )

        metadata_raw = data.get("metadata", {})
        _expect_type(metadata_raw, dict, "trajectory.metadata")
        metadata: dict[str, str] = {}
        for key, value in metadata_raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SchemaValidationError("trajectory.metadata: expected str->str map")
            metadata[key] = value

        return cls(
            schema_version=data["schema_version"],
            trajectory_id=data["trajectory_id"],
            instance_id=data["instance_id"],
            task=data["task"],
            model=data["model"],
            seed=data["seed"],
            created_at=float(data["created_at"]),
            llm_claimed_success=data["llm_claimed_success"],
            verification=verification,
            attempts=attempts,
            tool_call_events=tool_call_events,
            token_usage=token_usage,
            metadata=metadata,
        )

    @classmethod
    def from_json_line(cls, line: str) -> "TrajectoryRecord":
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"trajectory json decode error: {e}") from e
        if not isinstance(payload, dict):
            raise SchemaValidationError("trajectory root: expected object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "instance_id": self.instance_id,
            "task": self.task,
            "model": self.model,
            "seed": self.seed,
            "created_at": self.created_at,
            "llm_claimed_success": self.llm_claimed_success,
            "verification": self.verification.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "tool_call_events": [evt.to_dict() for evt in self.tool_call_events],
            "token_usage": self.token_usage.to_dict(),
            "metadata": dict(self.metadata),
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def load_trajectory_jsonl(path: Path) -> list[TrajectoryRecord]:
    """Load and validate all trajectory lines from a JSONL file."""
    records: list[TrajectoryRecord] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                records.append(TrajectoryRecord.from_json_line(raw))
            except SchemaValidationError as e:
                raise SchemaValidationError(f"{path}:{i}: {e}") from e
    return records


def write_trajectory_jsonl(path: Path, records: Iterable[TrajectoryRecord]) -> None:
    """Write trajectory records to JSONL with one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json_line())
            f.write("\n")


def append_trajectory_jsonl(path: Path, record: TrajectoryRecord) -> None:
    """Append a single trajectory record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.to_json_line())
        f.write("\n")
