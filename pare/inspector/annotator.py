"""Apply the research-branch failure classifier to a trajectory.

R0 scaffold — function signatures only.

All classification logic lives in the research package
(``pare_research.trajectory.{classifier_liu, error_signal_extractor,
recovery_detector_v2}``). This module is a thin wrapper that fans those
calls out and bundles the results into one ``AnnotatedTrajectory`` for
the renderer to consume.

Do not reimplement classification here. The whole point of the
research/product split is that the classifiers stay one source of
truth on the research side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pare_research.trajectory.schema_v2 import (  # type: ignore[import-not-found]
        ToolCallEvent,
        TrajectoryRecord,
    )


@dataclass(frozen=True, slots=True)
class StepAnnotation:
    """Per-step annotations stitched onto a ToolCallEvent."""

    event: "ToolCallEvent"
    error_signal_inferred: str        # ErrorSignal name (post-extractor pass)
    recovery_label: str               # "L1" | "L2" | "L3" | "none"


@dataclass(frozen=True, slots=True)
class AnnotatedTrajectory:
    """A TrajectoryRecord plus per-step and trajectory-level annotations."""

    record: "TrajectoryRecord"
    steps: list[StepAnnotation]
    liu_classification: Any           # research.classifier_liu.LiuClassification
    outcome_label: str                # OutcomeLabel name


def annotate(record: "TrajectoryRecord") -> AnnotatedTrajectory:
    """Run the full classifier pipeline against one record.

    Order matters:
        1. error_signal_extractor over each ToolCallEvent (re-extract
           even if record.tool_call_events[*].error_signal is already
           populated — old JSONLs may have NONE for everything)
        2. recovery_detector_v2 over the (events, signals) pair
        3. classifier_liu.classify_liu_from_record over the record + signals
        4. assign_outcome_label from the Liu result + verification + recovery

    Steps 1-2 yield per-step annotations, steps 3-4 yield trajectory-level.
    """
    raise NotImplementedError("W1 Day 1-2")
