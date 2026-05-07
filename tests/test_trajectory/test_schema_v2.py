"""Tests for ToolCallEvent schema (v2) and TrajectoryRecord integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pare.trajectory.schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)
from pare.trajectory.schema_v2 import (
    ErrorSignal,
    ToolCallEvent,
    _compute_params_hash,
    _extract_target_file,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _sample_tool_call_event(**overrides) -> dict:
    base = {
        "turn_id": 0,
        "call_index_in_turn": 0,
        "global_index": 0,
        "tool_name": "file_read",
        "params": {"file_path": "main.py"},
        "params_hash": _compute_params_hash({"file_path": "main.py"}),
        "target_file": "main.py",
        "result_success": True,
        "result_content": "def hello(): ...",
        "error_signal": "NONE",
        "timestamp": 1710000001.0,
    }
    base.update(overrides)
    return base


def _sample_trajectory_with_events() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": "traj-v2-001",
        "instance_id": "swe-456",
        "task": "Fix the bug",
        "model": "deepseek/deepseek-chat",
        "seed": 0,
        "created_at": 1710000000.0,
        "llm_claimed_success": True,
        "verification": {
            "final_passed": True,
            "has_diff": True,
            "tier2_pass": True,
        },
        "attempts": [
            {
                "step_number": 1,
                "attempt_number": 1,
                "goal": "Fix bug",
                "status": "success",
            }
        ],
        "tool_call_events": [
            _sample_tool_call_event(
                turn_id=0,
                call_index_in_turn=0,
                global_index=0,
                tool_name="file_read",
                params={"file_path": "main.py"},
            ),
            _sample_tool_call_event(
                turn_id=0,
                call_index_in_turn=1,
                global_index=1,
                tool_name="file_edit",
                params={"file_path": "main.py", "content": "fixed"},
                target_file="main.py",
                result_content="File edited successfully",
            ),
            _sample_tool_call_event(
                turn_id=1,
                call_index_in_turn=0,
                global_index=2,
                tool_name="bash",
                params={"command": "pytest tests/"},
                target_file="",
                result_content="1 passed",
            ),
        ],
        "token_usage": {
            "input_tokens": 2000,
            "output_tokens": 500,
        },
    }


# ---------------------------------------------------------------------------
# ToolCallEvent unit tests
# ---------------------------------------------------------------------------


class TestToolCallEvent:
    def test_create_factory(self):
        evt = ToolCallEvent.create(
            turn_id=0,
            call_index_in_turn=0,
            global_index=0,
            tool_name="file_read",
            params={"file_path": "main.py"},
            result_success=True,
            result_content="contents...",
            timestamp=1710000001.0,
        )

        assert evt.turn_id == 0
        assert evt.tool_name == "file_read"
        assert evt.target_file == "main.py"
        assert evt.params_hash == _compute_params_hash({"file_path": "main.py"})
        assert evt.error_signal == ErrorSignal.NONE
        assert evt.result_success is True

    def test_create_auto_target_file_edit(self):
        evt = ToolCallEvent.create(
            turn_id=0,
            call_index_in_turn=0,
            global_index=0,
            tool_name="file_edit",
            params={"file_path": "src/utils.py", "content": "new"},
            result_success=True,
            result_content="ok",
            timestamp=1.0,
        )
        assert evt.target_file == "src/utils.py"

    def test_create_auto_target_search(self):
        evt = ToolCallEvent.create(
            turn_id=0,
            call_index_in_turn=0,
            global_index=0,
            tool_name="search",
            params={"path": "src/", "query": "def foo"},
            result_success=True,
            result_content="found",
            timestamp=1.0,
        )
        assert evt.target_file == "src/"

    def test_create_bash_no_target(self):
        evt = ToolCallEvent.create(
            turn_id=0,
            call_index_in_turn=0,
            global_index=0,
            tool_name="bash",
            params={"command": "ls -la"},
            result_success=True,
            result_content="...",
            timestamp=1.0,
        )
        assert evt.target_file == ""

    def test_temporal_key(self):
        evt = ToolCallEvent.create(
            turn_id=2,
            call_index_in_turn=3,
            global_index=7,
            tool_name="bash",
            params={"command": "echo hi"},
            result_success=True,
            result_content="hi",
            timestamp=1.0,
        )
        assert evt.temporal_key() == (2, 3)

    def test_roundtrip_to_dict_from_dict(self):
        evt = ToolCallEvent.create(
            turn_id=1,
            call_index_in_turn=2,
            global_index=5,
            tool_name="file_edit",
            params={"file_path": "a.py", "content": "x"},
            result_success=False,
            result_content="ERROR: file not found",
            timestamp=1710000005.0,
            error_signal=ErrorSignal.RUNTIME_ERROR,
        )

        d = evt.to_dict()
        assert d["error_signal"] == "RUNTIME_ERROR"
        assert d["turn_id"] == 1
        assert d["params_hash"] == evt.params_hash

        restored = ToolCallEvent.from_dict(d)
        assert restored == evt

    def test_from_dict_minimal(self):
        """Only required fields — optional fields get defaults."""
        d = {
            "turn_id": 0,
            "call_index_in_turn": 0,
            "global_index": 0,
            "tool_name": "bash",
            "result_success": True,
            "timestamp": 1.0,
        }
        evt = ToolCallEvent.from_dict(d)
        assert evt.params == {}
        assert evt.target_file == ""
        assert evt.result_content == ""
        assert evt.error_signal == ErrorSignal.NONE

    def test_from_dict_blocked_signal(self):
        d = _sample_tool_call_event(
            error_signal="BLOCKED",
            result_success=False,
            result_content="[BLOCKED] budget exhausted",
        )
        evt = ToolCallEvent.from_dict(d)
        assert evt.error_signal == ErrorSignal.BLOCKED

    def test_frozen(self):
        evt = ToolCallEvent.create(
            turn_id=0,
            call_index_in_turn=0,
            global_index=0,
            tool_name="bash",
            params={},
            result_success=True,
            result_content="",
            timestamp=1.0,
        )
        with pytest.raises(AttributeError):
            evt.turn_id = 99  # type: ignore[misc]


class TestToolCallEventValidation:
    def test_missing_required_field(self):
        d = _sample_tool_call_event()
        del d["tool_name"]
        with pytest.raises(SchemaValidationError):
            ToolCallEvent.from_dict(d)

    def test_invalid_turn_id_type(self):
        d = _sample_tool_call_event(turn_id="zero")
        with pytest.raises(SchemaValidationError):
            ToolCallEvent.from_dict(d)

    def test_invalid_error_signal_value(self):
        d = _sample_tool_call_event(error_signal="INVALID_SIGNAL")
        with pytest.raises(SchemaValidationError):
            ToolCallEvent.from_dict(d)

    def test_invalid_params_type(self):
        d = _sample_tool_call_event(params="not a dict")
        with pytest.raises(SchemaValidationError):
            ToolCallEvent.from_dict(d)

    def test_unknown_key_raises(self):
        d = _sample_tool_call_event(extra_field="bad")
        with pytest.raises(SchemaValidationError):
            ToolCallEvent.from_dict(d)


# ---------------------------------------------------------------------------
# ErrorSignal tests
# ---------------------------------------------------------------------------


class TestErrorSignal:
    def test_all_values(self):
        expected = {
            "NONE", "SYNTAX_ERROR", "TEST_FAILURE", "RUNTIME_ERROR",
            "COMMAND_NOT_FOUND", "EMPTY_DIFF", "TIMEOUT", "BLOCKED", "OTHER",
        }
        assert {s.value for s in ErrorSignal} == expected

    def test_roundtrip(self):
        for signal in ErrorSignal:
            assert ErrorSignal(signal.value) is signal


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_params_hash_deterministic(self):
        params = {"file_path": "main.py", "content": "hello"}
        h1 = _compute_params_hash(params)
        h2 = _compute_params_hash(params)
        assert h1 == h2
        assert len(h1) == 16  # SHA-256 prefix

    def test_params_hash_different_for_different_params(self):
        h1 = _compute_params_hash({"file_path": "a.py"})
        h2 = _compute_params_hash({"file_path": "b.py"})
        assert h1 != h2

    def test_params_hash_order_independent(self):
        h1 = _compute_params_hash({"a": 1, "b": 2})
        h2 = _compute_params_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_extract_target_file_read(self):
        assert _extract_target_file("file_read", {"file_path": "x.py"}) == "x.py"

    def test_extract_target_file_edit(self):
        assert _extract_target_file("file_edit", {"file_path": "y.py", "content": "..."}) == "y.py"

    def test_extract_target_file_create(self):
        assert _extract_target_file("file_create", {"file_path": "z.py"}) == "z.py"

    def test_extract_target_search(self):
        assert _extract_target_file("search", {"path": "src/"}) == "src/"

    def test_extract_target_bash(self):
        assert _extract_target_file("bash", {"command": "ls"}) == ""

    def test_extract_target_unknown_tool(self):
        assert _extract_target_file("custom_tool", {"x": 1}) == ""


# ---------------------------------------------------------------------------
# TrajectoryRecord + ToolCallEvent integration
# ---------------------------------------------------------------------------


class TestTrajectoryRecordWithEvents:
    def test_roundtrip_with_tool_call_events(self):
        payload = _sample_trajectory_with_events()
        record = TrajectoryRecord.from_dict(payload)

        assert len(record.tool_call_events) == 3
        assert record.tool_call_events[0].tool_name == "file_read"
        assert record.tool_call_events[1].tool_name == "file_edit"
        assert record.tool_call_events[2].tool_name == "bash"
        assert record.tool_call_events[2].turn_id == 1

        # Round-trip
        d = record.to_dict()
        assert len(d["tool_call_events"]) == 3
        assert d["tool_call_events"][0]["tool_name"] == "file_read"

        restored = TrajectoryRecord.from_dict(d)
        assert len(restored.tool_call_events) == 3
        assert restored.tool_call_events[2].global_index == 2

    def test_backward_compat_no_events(self):
        """v1 payloads without tool_call_events still load fine."""
        payload = {
            "schema_version": "1.0",
            "trajectory_id": "traj-legacy",
            "instance_id": "swe-old",
            "task": "Old task",
            "model": "gpt-4o",
            "seed": 0,
            "created_at": 1700000000.0,
            "llm_claimed_success": True,
            "verification": {
                "final_passed": True,
                "has_diff": True,
                "tier2_pass": False,
            },
        }
        record = TrajectoryRecord.from_dict(payload)
        assert record.tool_call_events == []
        assert record.schema_version == "1.0"

    def test_jsonl_roundtrip_with_events(self, tmp_path: Path):
        path = tmp_path / "traj_v2.jsonl"
        record = TrajectoryRecord.from_dict(_sample_trajectory_with_events())

        write_trajectory_jsonl(path, [record])
        loaded = load_trajectory_jsonl(path)

        assert len(loaded) == 1
        assert len(loaded[0].tool_call_events) == 3
        assert loaded[0].tool_call_events[1].target_file == "main.py"

    def test_to_dict_includes_events(self):
        payload = _sample_trajectory_with_events()
        record = TrajectoryRecord.from_dict(payload)
        d = record.to_dict()

        assert "tool_call_events" in d
        assert isinstance(d["tool_call_events"], list)
        assert d["tool_call_events"][0]["error_signal"] == "NONE"

    def test_temporal_ordering_preserved(self):
        payload = _sample_trajectory_with_events()
        record = TrajectoryRecord.from_dict(payload)

        keys = [evt.temporal_key() for evt in record.tool_call_events]
        assert keys == sorted(keys), "Events should be in temporal order"

        global_indices = [evt.global_index for evt in record.tool_call_events]
        assert global_indices == [0, 1, 2], "Global indices should be monotonic"
