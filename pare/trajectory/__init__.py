"""Trajectory data model and deterministic labeling utilities."""

from pare.trajectory.classifier import (
    ClassificationResult,
    TrajectoryClassifier,
    TrajectoryLabel,
)
from pare.trajectory.recovery_detector import (
    RecoveryEvent,
    RecoveryLevel,
    detect_recovery_events,
    highest_recovery_level,
)
from pare.trajectory.schema import (
    append_trajectory_jsonl,
    SCHEMA_VERSION,
    SchemaValidationError,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)

__all__ = [
    "ClassificationResult",
    "TrajectoryClassifier",
    "TrajectoryLabel",
    "append_trajectory_jsonl",
    "RecoveryEvent",
    "RecoveryLevel",
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "StepAttempt",
    "TokenUsageSummary",
    "TrajectoryRecord",
    "VerificationResult",
    "detect_recovery_events",
    "highest_recovery_level",
    "load_trajectory_jsonl",
    "write_trajectory_jsonl",
]
