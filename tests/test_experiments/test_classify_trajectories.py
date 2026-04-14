"""Tests for trajectory classification script."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.classify_trajectories import classify_trajectories, main
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)


def _record(
    trajectory_id: str,
    *,
    llm_claimed_success: bool,
    final_passed: bool,
    tier1_pass: bool,
    tier2_pass: bool,
    attempts: list[StepAttempt],
) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=trajectory_id,
        instance_id=f"inst-{trajectory_id}",
        task="Fix bug",
        model="deepseek/deepseek-chat",
        seed=0,
        created_at=1710000000.0,
        llm_claimed_success=llm_claimed_success,
        verification=VerificationResult(
            final_passed=final_passed,
            tier1_pass=tier1_pass,
            tier2_pass=tier2_pass,
            tier2_command="",
        ),
        attempts=attempts,
        token_usage=TokenUsageSummary(input_tokens=100, output_tokens=20),
        metadata={},
    )


def _attempt(
    step: int,
    attempt_number: int,
    status: str,
    *,
    rolled_back: bool = False,
) -> StepAttempt:
    return StepAttempt(
        step_number=step,
        attempt_number=attempt_number,
        goal="Fix parser",
        status=status,
        rolled_back=rolled_back,
        target_files=["a.py"],
        tool_names=["file_edit"],
        failure_reason="" if status == "success" else "error",
    )


class TestClassifyTrajectoriesScript:
    def test_classify_and_write_outputs(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        non_toxic_path = tmp_path / "non_toxic.jsonl"

        records = [
            _record(
                "toxic",
                llm_claimed_success=True,
                final_passed=False,
                tier1_pass=False,
                tier2_pass=False,
                attempts=[_attempt(1, 1, "failed")],
            ),
            _record(
                "weak",
                llm_claimed_success=True,
                final_passed=True,
                tier1_pass=True,
                tier2_pass=False,
                attempts=[_attempt(1, 1, "success")],
            ),
            _record(
                "one",
                llm_claimed_success=True,
                final_passed=True,
                tier1_pass=True,
                tier2_pass=True,
                attempts=[_attempt(1, 1, "success")],
            ),
            _record(
                "recovery",
                llm_claimed_success=True,
                final_passed=True,
                tier1_pass=True,
                tier2_pass=True,
                attempts=[
                    _attempt(1, 1, "failed", rolled_back=True),
                    _attempt(1, 2, "success"),
                ],
            ),
        ]
        write_trajectory_jsonl(trajectory_path, records)

        summary = classify_trajectories(
            trajectory_path,
            labels_jsonl=labels_path,
            non_toxic_jsonl=non_toxic_path,
        )

        assert summary.total == 4
        assert summary.toxic_count == 1
        assert summary.non_toxic_count == 3
        assert labels_path.exists()
        assert non_toxic_path.exists()

        loaded_non_toxic = load_trajectory_jsonl(non_toxic_path)
        assert len(loaded_non_toxic) == 3

        lines = [line for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 4
        payload = json.loads(lines[0])
        assert "primary_label" in payload

    def test_main_writes_default_outputs(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        write_trajectory_jsonl(
            trajectory_path,
            [
                _record(
                    "one",
                    llm_claimed_success=True,
                    final_passed=True,
                    tier1_pass=True,
                    tier2_pass=True,
                    attempts=[_attempt(1, 1, "success")],
                )
            ],
        )

        code = main([
            "--trajectory-jsonl",
            str(trajectory_path),
        ])
        assert code == 0

        assert trajectory_path.with_suffix(".labels.jsonl").exists()
        assert trajectory_path.with_suffix(".summary.json").exists()
        assert trajectory_path.with_suffix(".non_toxic.jsonl").exists()
