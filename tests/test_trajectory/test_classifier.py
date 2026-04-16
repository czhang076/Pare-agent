"""Tests for deterministic trajectory classifier (v1 — deprecated).

Kept as a regression harness for the legacy classifier. New code should
use ``classifier_liu`` (plus ``recovery_detector_v2``) via
``experiments/classify_trajectories.py``.
"""

from __future__ import annotations

import pytest

from pare.trajectory.classifier import TrajectoryClassifier, TrajectoryLabel
from pare.trajectory.recovery_detector import RecoveryLevel
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::DeprecationWarning:pare.trajectory.classifier"
    ),
    pytest.mark.filterwarnings(
        "ignore::DeprecationWarning:pare.trajectory.recovery_detector"
    ),
]


def _record(
    *,
    llm_claimed_success: bool,
    verification: VerificationResult,
    attempts: list[StepAttempt],
) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id="traj-1",
        instance_id="swe-1",
        task="Fix bug",
        model="deepseek/deepseek-chat",
        seed=0,
        created_at=1710000000.0,
        llm_claimed_success=llm_claimed_success,
        verification=verification,
        attempts=attempts,
        token_usage=TokenUsageSummary(input_tokens=100, output_tokens=20),
        metadata={},
    )


def _attempt(
    step: int,
    attempt: int,
    status: str,
    *,
    goal: str = "Fix parser",
    files: list[str] | None = None,
    tools: list[str] | None = None,
) -> StepAttempt:
    return StepAttempt(
        step_number=step,
        attempt_number=attempt,
        goal=goal,
        status=status,
        target_files=files or [],
        tool_names=tools or [],
    )


class TestTrajectoryClassifier:
    def test_toxic(self):
        classifier = TrajectoryClassifier()
        record = _record(
            llm_claimed_success=True,
            verification=VerificationResult(final_passed=False, tier1_pass=False, tier2_pass=False),
            attempts=[],
        )

        result = classifier.classify(record)
        assert result.primary_label == TrajectoryLabel.TOXIC

    def test_failed(self):
        classifier = TrajectoryClassifier()
        record = _record(
            llm_claimed_success=False,
            verification=VerificationResult(final_passed=False, tier1_pass=True, tier2_pass=False),
            attempts=[],
        )

        result = classifier.classify(record)
        assert result.primary_label == TrajectoryLabel.FAILED

    def test_weakly_verified(self):
        classifier = TrajectoryClassifier()
        record = _record(
            llm_claimed_success=True,
            verification=VerificationResult(final_passed=True, tier1_pass=True, tier2_pass=False),
            attempts=[],
        )

        result = classifier.classify(record)
        assert result.primary_label == TrajectoryLabel.WEAKLY_VERIFIED

    def test_one_shot_success(self):
        classifier = TrajectoryClassifier()
        record = _record(
            llm_claimed_success=True,
            verification=VerificationResult(final_passed=True, tier1_pass=True, tier2_pass=True),
            attempts=[
                _attempt(1, 1, "success", files=["a.py"], tools=["file_edit"]),
                _attempt(2, 1, "success", files=["b.py"], tools=["bash"]),
            ],
        )

        result = classifier.classify(record)
        assert result.primary_label == TrajectoryLabel.ONE_SHOT_SUCCESS

    def test_failure_recovery_with_l1(self):
        classifier = TrajectoryClassifier()
        record = _record(
            llm_claimed_success=True,
            verification=VerificationResult(final_passed=True, tier1_pass=True, tier2_pass=True),
            attempts=[
                _attempt(1, 1, "failed", files=["a.py"], tools=["file_edit"]),
                _attempt(1, 2, "success", files=["a.py"], tools=["file_edit"]),
            ],
        )

        result = classifier.classify(record)
        assert result.primary_label == TrajectoryLabel.FAILURE_RECOVERY
        assert result.recovery_level == RecoveryLevel.L1_RETRY

    def test_fully_verified_fallback_without_attempts(self):
        classifier = TrajectoryClassifier()
        record = _record(
            llm_claimed_success=True,
            verification=VerificationResult(final_passed=True, tier1_pass=True, tier2_pass=True),
            attempts=[],
        )

        result = classifier.classify(record)
        assert result.primary_label == TrajectoryLabel.FULLY_VERIFIED

    def test_classify_many(self):
        classifier = TrajectoryClassifier()
        records = [
            _record(
                llm_claimed_success=False,
                verification=VerificationResult(final_passed=False, tier1_pass=False, tier2_pass=False),
                attempts=[],
            ),
            _record(
                llm_claimed_success=True,
                verification=VerificationResult(final_passed=True, tier1_pass=True, tier2_pass=False),
                attempts=[],
            ),
        ]

        results = classifier.classify_many(records)
        assert len(results) == 2
        assert results[0].primary_label == TrajectoryLabel.FAILED
        assert results[1].primary_label == TrajectoryLabel.WEAKLY_VERIFIED
