"""Deterministic trajectory classifier (v1 — DEPRECATED).

DEPRECATED: Step-based classifier using a bespoke 6-label taxonomy.
Superseded by ``pare.trajectory.classifier_liu`` (Liu et al. 2025
taxonomy) paired with ``pare.trajectory.recovery_detector_v2``. The v2
pipeline is exposed via ``experiments/classify_trajectories.py``. Kept
for backward compatibility with ``pare.curation.sampler``; do not
build new code against this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from pare.trajectory.recovery_detector import (
    RecoveryEvent,
    RecoveryLevel,
    detect_recovery_events,
    highest_recovery_level,
)
from pare.trajectory.schema import TrajectoryRecord

_DEPRECATION_MSG = (
    "pare.trajectory.classifier is deprecated; "
    "use pare.trajectory.classifier_liu (with recovery_detector_v2) instead."
)


class TrajectoryLabel(str, Enum):
    TOXIC = "toxic"
    FAILED = "failed"
    WEAKLY_VERIFIED = "weakly_verified"
    FULLY_VERIFIED = "fully_verified"
    ONE_SHOT_SUCCESS = "one_shot_success"
    FAILURE_RECOVERY = "failure_recovery"


@dataclass(slots=True)
class ClassificationResult:
    primary_label: TrajectoryLabel
    verification_label: TrajectoryLabel
    recovery_level: RecoveryLevel | None = None
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class TrajectoryClassifier:
    """Classify trajectories according to deterministic plan rules.

    DEPRECATED — see module docstring.
    """

    def __init__(self) -> None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

    def classify(self, trajectory: TrajectoryRecord) -> ClassificationResult:
        reasons: list[str] = []
        verification_label = self._verification_label(trajectory)

        if (
            trajectory.llm_claimed_success
            and not trajectory.verification.tier1_pass
        ):
            reasons.append("llm claimed success while tier1 verification failed")
            return ClassificationResult(
                primary_label=TrajectoryLabel.TOXIC,
                verification_label=verification_label,
                reasons=reasons,
            )

        if not trajectory.verification.final_passed:
            reasons.append("final verification did not pass")
            return ClassificationResult(
                primary_label=TrajectoryLabel.FAILED,
                verification_label=verification_label,
                reasons=reasons,
            )

        if verification_label == TrajectoryLabel.WEAKLY_VERIFIED:
            reasons.append("tier1 passed without tier2 confirmation")
            return ClassificationResult(
                primary_label=TrajectoryLabel.WEAKLY_VERIFIED,
                verification_label=verification_label,
                reasons=reasons,
            )

        # final_passed + tier1 + tier2
        recovery_events = detect_recovery_events(trajectory.attempts)
        if recovery_events:
            level = highest_recovery_level(recovery_events)
            reasons.append("detected failure -> rollback -> success pattern")
            return ClassificationResult(
                primary_label=TrajectoryLabel.FAILURE_RECOVERY,
                verification_label=verification_label,
                recovery_level=level,
                recovery_events=recovery_events,
                reasons=reasons,
            )

        if not trajectory.attempts:
            reasons.append("fully verified with no attempt-level traces")
            return ClassificationResult(
                primary_label=TrajectoryLabel.FULLY_VERIFIED,
                verification_label=verification_label,
                reasons=reasons,
            )

        if all(attempt.status == "success" for attempt in trajectory.attempts):
            reasons.append("all attempts succeeded with no recovery events")
            return ClassificationResult(
                primary_label=TrajectoryLabel.ONE_SHOT_SUCCESS,
                verification_label=verification_label,
                reasons=reasons,
            )

        reasons.append("fully verified but no explicit rollback-recovery chain")
        return ClassificationResult(
            primary_label=TrajectoryLabel.FULLY_VERIFIED,
            verification_label=verification_label,
            reasons=reasons,
        )

    def classify_many(self, trajectories: Sequence[TrajectoryRecord]) -> list[ClassificationResult]:
        return [self.classify(trajectory) for trajectory in trajectories]

    @staticmethod
    def _verification_label(trajectory: TrajectoryRecord) -> TrajectoryLabel:
        if trajectory.verification.tier1_pass and trajectory.verification.tier2_pass:
            return TrajectoryLabel.FULLY_VERIFIED
        if trajectory.verification.tier1_pass:
            return TrajectoryLabel.WEAKLY_VERIFIED
        return TrajectoryLabel.FAILED
