"""Trajectory data model and deterministic labeling utilities.

Public surface after R5:

- ``schema``        — v1 JSONL master + v2 ``tool_call_events`` extension.
- ``schema_v2``     — ``ToolCallEvent``, ``ErrorSignal``, recovery/classification
                      field definitions. Separate module because it can be
                      consumed without the pre-v2 attempt-centric schema.
- ``classifier_liu`` — Liu et al. (2025) 8-category classifier (core + extended).
- ``recovery_detector_v2`` — ``RecoveryEvent`` / ``RecoveryLevel`` + detector.
- ``error_signal_extractor`` — stdout/stderr → ``ErrorSignal`` classifier.

The v1 ``classifier`` and ``recovery_detector`` modules were removed when
the flat-ReAct refactor landed. The sampler owns its own label vocabulary
in ``pare.curation.sampler`` now.
"""

from pare.trajectory.recovery_detector_v2 import (
    RecoveryDetectionResult,
    RecoveryEvent,
    RecoveryLevel,
    detect_recovery_events,
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
from pare.trajectory.sft_export import (
    ExportReport,
    SFTRow,
    export_dataset,
    export_trajectory_to_sft,
)

__all__ = [
    "append_trajectory_jsonl",
    "ExportReport",
    "RecoveryDetectionResult",
    "RecoveryEvent",
    "RecoveryLevel",
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "SFTRow",
    "StepAttempt",
    "TokenUsageSummary",
    "TrajectoryRecord",
    "VerificationResult",
    "detect_recovery_events",
    "export_dataset",
    "export_trajectory_to_sft",
    "load_trajectory_jsonl",
    "write_trajectory_jsonl",
]
