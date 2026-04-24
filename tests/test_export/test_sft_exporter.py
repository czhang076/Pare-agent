"""Tests for SFT exporter (trajectory -> OpenAI messages)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pare.export.sft_exporter import (
    SFTExportError,
    SFTExporter,
    SFTExporterConfig,
    export_trajectory_jsonl_to_sft,
)
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    write_trajectory_jsonl,
)


def _sample_record(*, metadata: dict[str, str] | None = None) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id="traj-1",
        instance_id="swe-1",
        task="Fix parser bug",
        model="deepseek/deepseek-chat",
        seed=1,
        created_at=1710000000.0,
        llm_claimed_success=True,
        verification=VerificationResult(
            final_passed=True,
            has_diff=True,
            tier2_pass=False,
            tier2_command="",
        ),
        attempts=[
            StepAttempt(
                step_number=1,
                attempt_number=1,
                goal="Update parser",
                status="failed",
                target_files=["parser.py"],
                tool_names=["file_read", "file_edit"],
                failure_reason="syntax error",
            ),
            StepAttempt(
                step_number=1,
                attempt_number=2,
                goal="Update parser",
                status="success",
                target_files=["parser.py"],
                tool_names=["file_edit"],
                failure_reason="",
            ),
        ],
        token_usage=TokenUsageSummary(input_tokens=120, output_tokens=30),
        metadata=metadata or {},
    )


class TestSFTExporter:
    def test_uses_raw_messages_from_metadata(self):
        raw_messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix bug."},
            {"role": "assistant", "content": "Done."},
        ]
        record = _sample_record(
            metadata={
                "openai_messages_json": json.dumps(raw_messages),
            }
        )

        exporter = SFTExporter()
        sample = exporter.export_record(record)

        assert sample["messages"] == raw_messages
        assert sample["metadata"]["export_source"] == "metadata_raw_messages"

    def test_reconstructs_when_no_raw_messages(self):
        record = _sample_record()
        exporter = SFTExporter()

        sample = exporter.export_record(record)
        messages = sample["messages"]

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Fix parser bug"

        # At least one assistant tool-call turn and one tool turn exist.
        assert any(msg.get("tool_calls") for msg in messages if msg["role"] == "assistant")
        assert any(msg["role"] == "tool" for msg in messages)

        # Final summary must include verification fields.
        assert messages[-1]["role"] == "assistant"
        assert "final_passed=True" in messages[-1]["content"]
        assert sample["metadata"]["export_source"] == "reconstructed_attempts"

    def test_require_raw_messages_raises(self):
        record = _sample_record()
        exporter = SFTExporter(SFTExporterConfig(require_raw_messages=True))

        with pytest.raises(SFTExportError, match="Raw messages required"):
            exporter.export_record(record)

    def test_invalid_raw_messages_json_raises(self):
        record = _sample_record(metadata={"openai_messages_json": "{broken"})
        exporter = SFTExporter()

        with pytest.raises(SFTExportError, match="Invalid JSON"):
            exporter.export_record(record)

    def test_invalid_raw_messages_shape_raises(self):
        record = _sample_record(metadata={"openai_messages_json": json.dumps({"role": "user"})})
        exporter = SFTExporter()

        with pytest.raises(SFTExportError, match="must be a JSON list"):
            exporter.export_record(record)

    def test_export_many(self):
        exporter = SFTExporter()
        records = [_sample_record(), _sample_record(metadata={"system_prompt": "custom"})]

        out = exporter.export_many(records)
        assert len(out) == 2


class TestExportTrajectoryJsonlToSFT:
    def test_end_to_end_jsonl_export(self, tmp_path: Path):
        input_path = tmp_path / "trajectories.jsonl"
        output_path = tmp_path / "sft.jsonl"

        records = [_sample_record(), _sample_record()]
        write_trajectory_jsonl(input_path, records)

        exported_count = export_trajectory_jsonl_to_sft(input_path, output_path)
        assert exported_count == 2

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert "messages" in first
        assert isinstance(first["messages"], list)
        assert first["messages"][0]["role"] == "system"
