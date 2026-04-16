"""Deterministic failure-recovery pattern detection (v1 — DEPRECATED).

DEPRECATED: Step-based recovery detection. Superseded by
``pare.trajectory.recovery_detector_v2`` which operates on ToolCallEvent
sequences and is driven by error signals rather than the dead
``StepAttempt.rolled_back`` field. Kept for backward compatibility with
``pare.trajectory.classifier`` (also deprecated). Do not build new code
against this module.

This module identifies failure -> rollback -> success transitions and
classifies them into recovery levels:
- L1: Retry (same approach)
- L2: Strategy switch
- L3: Goal decomposition
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from pare.trajectory.schema import StepAttempt

_DEPRECATION_MSG = (
    "pare.trajectory.recovery_detector is deprecated; "
    "use pare.trajectory.recovery_detector_v2 instead."
)


class RecoveryLevel(str, Enum):
    L1_RETRY = "L1"
    L2_STRATEGY_SWITCH = "L2"
    L3_GOAL_DECOMPOSITION = "L3"


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    failure_index: int
    success_index: int
    failure_step: int
    success_step: int
    level: RecoveryLevel
    reason: str


def detect_recovery_events(attempts: Sequence[StepAttempt]) -> list[RecoveryEvent]:
    """Detect deterministic failure-recovery events from ordered attempts.

    DEPRECATED — see module docstring.
    """
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    events: list[RecoveryEvent] = []

    for i, failed in enumerate(attempts):
        if failed.status == "success":
            continue

        success_index = _next_success_index(attempts, i + 1)
        if success_index is None:
            continue

        recovered = attempts[success_index]
        window = list(attempts[i + 1 : success_index + 1])
        level, reason = _classify_level(failed, recovered, window)

        events.append(
            RecoveryEvent(
                failure_index=i,
                success_index=success_index,
                failure_step=failed.step_number,
                success_step=recovered.step_number,
                level=level,
                reason=reason,
            )
        )

    return events


def highest_recovery_level(events: Sequence[RecoveryEvent]) -> RecoveryLevel | None:
    """Return the strongest observed recovery level among events."""
    if not events:
        return None
    rank = {
        RecoveryLevel.L1_RETRY: 1,
        RecoveryLevel.L2_STRATEGY_SWITCH: 2,
        RecoveryLevel.L3_GOAL_DECOMPOSITION: 3,
    }
    return max((event.level for event in events), key=lambda level: rank[level])


def _next_success_index(attempts: Sequence[StepAttempt], start: int) -> int | None:
    for idx in range(start, len(attempts)):
        if attempts[idx].status == "success":
            return idx
    return None


def _classify_level(
    failed: StepAttempt,
    recovered: StepAttempt,
    window: Sequence[StepAttempt],
) -> tuple[RecoveryLevel, str]:
    """Classify recovery level based on deterministic heuristics."""
    if _is_goal_decomposition(failed, recovered, window):
        return RecoveryLevel.L3_GOAL_DECOMPOSITION, "success moved to later/decomposed step(s)"

    if _is_same_approach(failed, recovered):
        return RecoveryLevel.L1_RETRY, "same step and similar approach after rollback"

    return RecoveryLevel.L2_STRATEGY_SWITCH, "same step but changed file/tool strategy"


def _is_goal_decomposition(
    failed: StepAttempt,
    recovered: StepAttempt,
    window: Sequence[StepAttempt],
) -> bool:
    if recovered.step_number > failed.step_number:
        return True

    step_numbers = {attempt.step_number for attempt in window}
    if len(step_numbers) > 1 and max(step_numbers) > failed.step_number:
        return True

    return False


def _is_same_approach(failed: StepAttempt, recovered: StepAttempt) -> bool:
    if recovered.step_number != failed.step_number:
        return False

    if failed.goal.strip().lower() != recovered.goal.strip().lower():
        return False

    if _jaccard_similarity(failed.target_files, recovered.target_files) < 0.6:
        return False

    failed_tool = _dominant_tool(failed.tool_names)
    recovered_tool = _dominant_tool(recovered.tool_names)
    if failed_tool and recovered_tool and failed_tool != recovered_tool:
        return False

    return True


def _jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {value for value in left if value}
    right_set = {value for value in right if value}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _dominant_tool(tool_names: Sequence[str]) -> str:
    if not tool_names:
        return ""
    counts = Counter(name for name in tool_names if name)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]
