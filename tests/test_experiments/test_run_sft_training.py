"""Tests for SFT LoRA smoke script."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.run_sft_training import (
    SFTSmokeError,
    main,
    load_and_validate_sft_jsonl,
    run_lora_smoke,
)
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    write_trajectory_jsonl,
)


def _record(trajectory_id: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=trajectory_id,
        instance_id=f"inst-{trajectory_id}",
        task="Fix parser bug",
        model="deepseek/deepseek-chat",
        seed=0,
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
                status="success",
                target_files=["parser.py"],
                tool_names=["file_read", "file_edit"],
            )
        ],
        token_usage=TokenUsageSummary(input_tokens=120, output_tokens=30),
        metadata={},
    )


class TestRunSFTTrainingSmoke:
    def test_run_lora_smoke_success(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        sft_path = tmp_path / "out.sft.jsonl"

        write_trajectory_jsonl(trajectory_path, [_record("t1"), _record("t2")])

        report = run_lora_smoke(
            trajectory_path,
            sft_jsonl=sft_path,
            batch_size=2,
            min_samples=1,
        )

        assert report.exported_samples == 2
        assert report.loaded_samples == 2
        assert report.batch_count == 1
        assert report.output_path == sft_path
        assert sft_path.exists()

    def test_load_and_validate_rejects_bad_format(self, tmp_path: Path):
        bad = tmp_path / "bad.sft.jsonl"
        bad.write_text(
            json.dumps({"messages": [{"role": "assistant", "content": 123}]}) + "\n",
            encoding="utf-8",
        )

        try:
            load_and_validate_sft_jsonl(bad)
            assert False, "expected SFTSmokeError"
        except SFTSmokeError:
            pass

    def test_main_success(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        sft_path = tmp_path / "out.sft.jsonl"
        write_trajectory_jsonl(trajectory_path, [_record("t1")])

        code = main([
            "--trajectory-jsonl",
            str(trajectory_path),
            "--sft-jsonl",
            str(sft_path),
            "--batch-size",
            "1",
        ])

        assert code == 0
        assert sft_path.exists()

    def test_main_failure_on_missing_input(self, tmp_path: Path):
        code = main([
            "--trajectory-jsonl",
            str(tmp_path / "missing.jsonl"),
        ])
        assert code == 1
