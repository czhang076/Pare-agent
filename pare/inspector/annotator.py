"""Apply the vendored failure classifier to a trajectory.

All classification logic lives in ``pare.trajectory.*``. This module is a
thin wrapper that fans those calls out and bundles the results into one
``AnnotatedTrajectory`` for the renderer to consume.

Do not reimplement classification here. The whole point of the
research/product split is that the classifiers stay one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from pare.trajectory.classifier_liu import (
    LiuClassification,
    assign_outcome_label,
    classify_liu_from_record,
)
from pare.trajectory.error_signal_extractor import classify_trajectory_signals
from pare.trajectory.recovery_detector_v2 import (
    RecoveryDetectionResult,
    detect_recovery_events,
)
from pare.trajectory.schema import TrajectoryRecord
from pare.trajectory.schema_v2 import ToolCallEvent


@dataclass(frozen=True, slots=True)
class StepAnnotation:
    """Per-step annotations stitched onto a ToolCallEvent."""

    event: ToolCallEvent
    error_signal_inferred: str        # ErrorSignal name (post-extractor pass)
    recovery_label: str               # "L1" | "L2" | "L3" | "none"


@dataclass(frozen=True, slots=True)
class AnnotatedTrajectory:
    """A TrajectoryRecord plus per-step and trajectory-level annotations."""

    record: TrajectoryRecord
    steps: list[StepAnnotation]
    liu_classification: LiuClassification
    outcome_label: str                # OutcomeLabel name


def _recovery_label_for_event(
    recovery: RecoveryDetectionResult, global_index: int
) -> str:
    """Return the recovery level string for an event, or ``"none"``.

    An event is labeled with a recovery level if it appears as either the
    error or the correction side of a ``RecoveryEvent``.
    """
    for ev in recovery.recovery_events:
        if ev.error_index == global_index or ev.correction_index == global_index:
            return ev.level.value
    return "none"


def annotate(record: TrajectoryRecord) -> AnnotatedTrajectory:
    """Run the full classifier pipeline against one record.

    Order matters:
        1. error_signal_extractor over each ToolCallEvent
        2. recovery_detector_v2 over the (events, signals) pair
        3. classifier_liu.classify_liu_from_record over record + signals
        4. assign_outcome_label from liu + verification + recovery
    """
    events = record.tool_call_events
    signals = classify_trajectory_signals(events)
    recovery = detect_recovery_events(events, signals)
    liu = classify_liu_from_record(record, signals)
    outcome = assign_outcome_label(
        liu, record.verification, recovery.contains_recovery
    )

    steps = [
        StepAnnotation(
            event=ev,
            error_signal_inferred=sig.name,
            recovery_label=_recovery_label_for_event(recovery, ev.global_index),
        )
        for ev, sig in zip(events, signals, strict=True)
    ]
    return AnnotatedTrajectory(
        record=record,
        steps=steps,
        liu_classification=liu,
        outcome_label=outcome.name,
    )
