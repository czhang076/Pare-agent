"""Tests for strict trajectory schema parsing and serialization."""

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


def _sample_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": "traj-001",
        "instance_id": "swe-123",
        "task": "Fix parser bug",
        "model": "deepseek/deepseek-chat",
        "seed": 0,
        "created_at": 1710000000.0,
        "llm_claimed_success": True,
        "verification": {
            "final_passed": True,
            "tier1_pass": True,
            "tier2_pass": True,
            "tier2_command": "pytest tests/",
        },
        "attempts": [
            {
                "step_number": 1,
                "attempt_number": 1,
                "goal": "Update parser",
                "status": "success",
                "rolled_back": False,
                "target_files": ["parser.py"],
                "tool_names": ["file_read", "file_edit"],
            }
        ],
        "token_usage": {
            "input_tokens": 1200,
            "output_tokens": 300,
            "cache_read_tokens": 0,
            "cache_create_tokens": 0,
        },
        "metadata": {
            "provider": "openrouter",
        },
    }


class TestSchemaRoundTrip:
    def test_from_dict_and_to_dict(self):
        payload = _sample_payload()
        record = TrajectoryRecord.from_dict(payload)

        assert record.trajectory_id == "traj-001"
        assert record.verification.tier2_pass is True
        assert record.attempts[0].status == "success"

        out = record.to_dict()
        assert out["schema_version"] == SCHEMA_VERSION
        assert out["token_usage"]["total_tokens"] == 1500

    def test_json_line_round_trip(self):
        payload = _sample_payload()
        line = json.dumps(payload)

        record = TrajectoryRecord.from_json_line(line)
        reloaded = TrajectoryRecord.from_json_line(record.to_json_line())

        assert reloaded.instance_id == record.instance_id
        assert reloaded.verification.final_passed == record.verification.final_passed

    def test_write_and_load_jsonl(self, tmp_path: Path):
        path = tmp_path / "trajectory.jsonl"
        record = TrajectoryRecord.from_dict(_sample_payload())

        write_trajectory_jsonl(path, [record])
        loaded = load_trajectory_jsonl(path)

        assert len(loaded) == 1
        assert loaded[0].trajectory_id == "traj-001"


class TestSchemaValidation:
    def test_missing_required_key_raises(self):
        payload = _sample_payload()
        del payload["task"]

        with pytest.raises(SchemaValidationError):
            TrajectoryRecord.from_dict(payload)

    def test_unknown_key_raises(self):
        payload = _sample_payload()
        payload["unknown"] = 1

        with pytest.raises(SchemaValidationError):
            TrajectoryRecord.from_dict(payload)

    def test_invalid_attempt_status_raises(self):
        payload = _sample_payload()
        payload["attempts"][0]["status"] = "retrying"

        with pytest.raises(SchemaValidationError):
            TrajectoryRecord.from_dict(payload)

    def test_invalid_metadata_map_raises(self):
        payload = _sample_payload()
        payload["metadata"] = {"provider": 123}

        with pytest.raises(SchemaValidationError):
            TrajectoryRecord.from_dict(payload)

    def test_version_mismatch_raises(self):
        payload = _sample_payload()
        payload["schema_version"] = "0.9"

        with pytest.raises(SchemaValidationError):
            TrajectoryRecord.from_dict(payload)
