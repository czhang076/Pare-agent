# Vendored from research branch claude/great-carson-333acd @ 8571fb1f.
# Do not edit here — make changes on the research side and re-vendor.
"""Trajectory data model and deterministic labeling utilities."""

from pare.trajectory.classifier_liu import (
    LiuClassification,
    OutcomeLabel,
    assign_outcome_label,
    classify_liu_from_record,
)
from pare.trajectory.error_signal_extractor import (
    classify_trajectory_signals,
    extract_error_signal,
)
from pare.trajectory.recovery_detector_v2 import (
    RecoveryDetectionResult,
    RecoveryEvent,
    RecoveryLevel,
    detect_recovery_events,
)
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    append_trajectory_jsonl,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent

__all__ = [
    "ErrorSignal",
    "LiuClassification",
    "OutcomeLabel",
    "RecoveryDetectionResult",
    "RecoveryEvent",
    "RecoveryLevel",
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "StepAttempt",
    "TokenUsageSummary",
    "ToolCallEvent",
    "TrajectoryRecord",
    "VerificationResult",
    "append_trajectory_jsonl",
    "assign_outcome_label",
    "classify_liu_from_record",
    "classify_trajectory_signals",
    "detect_recovery_events",
    "extract_error_signal",
    "load_trajectory_jsonl",
    "write_trajectory_jsonl",
]
