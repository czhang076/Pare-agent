"""Tool-call-level recovery detection (v2).

Finds error-correction pairs in a ``ToolCallEvent`` sequence and assigns
recovery levels per plan.md §3.2:

    L1: Local correction — same tool_name + same target_file, different params
    L2: Tactical switch  — different tool_name or different target_file
    L3: Exploratory recovery — investigation sequence (file_read/search) before correction

Each ``RecoveryEvent`` links an error event to its correction event.
A trajectory has ``contains_recovery = True`` iff at least one
``RecoveryEvent`` exists AND the trajectory passes Tier 1 + Tier 2
verification (§3.1.3).

All detection is deterministic. No LLM calls.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Sequence

from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


# ---------------------------------------------------------------------------
# Recovery level enum
# ---------------------------------------------------------------------------


class RecoveryLevel(enum.Enum):
    """Recovery level assigned to each (error, correction) pair."""

    L1 = "L1"  # Local correction: same tool + same target, different params
    L2 = "L2"  # Tactical switch: different tool or different target
    L3 = "L3"  # Exploratory recovery: investigation before correction


# ---------------------------------------------------------------------------
# RecoveryEvent — one error-correction pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    """An observed error → correction pair in a tool-call sequence.

    Attributes:
        error_index: global_index of the error event.
        correction_index: global_index of the correction event.
        error_signal: The error signal type of the error event.
        level: Recovery level (L1/L2/L3).
        investigation_count: Number of read/search calls between error and
                             correction (only meaningful for L3).
    """

    error_index: int
    correction_index: int
    error_signal: ErrorSignal
    level: RecoveryLevel
    investigation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "error_index": self.error_index,
            "correction_index": self.correction_index,
            "error_signal": self.error_signal.value,
            "level": self.level.value,
            "investigation_count": self.investigation_count,
        }


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecoveryDetectionResult:
    """Result of running recovery detection on a trajectory.

    Attributes:
        recovery_events: All detected (error, correction) pairs.
        contains_recovery: True iff at least one recovery event exists.
                           NOTE: caller must also check verification pass
                           per §3.1.3 before using this for trajectory labeling.
        highest_level: The highest recovery level found, or None.
    """

    recovery_events: list[RecoveryEvent]

    @property
    def contains_recovery(self) -> bool:
        return len(self.recovery_events) > 0

    @property
    def highest_level(self) -> RecoveryLevel | None:
        if not self.recovery_events:
            return None
        order = {RecoveryLevel.L1: 1, RecoveryLevel.L2: 2, RecoveryLevel.L3: 3}
        return max(self.recovery_events, key=lambda r: order[r.level]).level


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_INVESTIGATION_TOOLS = frozenset({"file_read", "search"})
_WRITE_TOOLS = frozenset({"file_edit", "file_create", "bash"})


def _is_investigation(event: ToolCallEvent) -> bool:
    """Is this event an investigation action (reading/searching)?"""
    return event.tool_name in _INVESTIGATION_TOOLS and event.result_success


def _is_correction_candidate(event: ToolCallEvent) -> bool:
    """Could this event be a correction action?"""
    return event.result_success and event.tool_name in (_WRITE_TOOLS | _INVESTIGATION_TOOLS)


def _params_materially_differ(a: ToolCallEvent, b: ToolCallEvent) -> bool:
    """Check that params are materially different (not identical retry)."""
    return a.params_hash != b.params_hash


def _classify_level(
    error_evt: ToolCallEvent,
    correction_evt: ToolCallEvent,
    events_between: Sequence[ToolCallEvent],
) -> RecoveryLevel:
    """Assign L1/L2/L3 to an (error, correction) pair.

    Rules (plan.md §3.2):
        L1: same tool_name AND same target_file, params differ
        L2: different tool_name OR different target_file
        L3: L1 or L2 conditions met, BUT preceded by ≥2 investigation calls
    """
    investigation_count = sum(1 for e in events_between if _is_investigation(e))

    # L3 check first: investigation sequence before correction
    if investigation_count >= 2:
        return RecoveryLevel.L3

    same_tool = error_evt.tool_name == correction_evt.tool_name
    same_target = (
        error_evt.target_file == correction_evt.target_file
        and error_evt.target_file != ""
    )

    if same_tool and same_target:
        return RecoveryLevel.L1

    # L1 also applies: same tool, both have no target (bash commands)
    if same_tool and error_evt.target_file == "" and correction_evt.target_file == "":
        return RecoveryLevel.L1

    return RecoveryLevel.L2


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

# Maximum gap (in global_index) between error and correction to consider
# them a pair. Prevents linking distant events that aren't related.
_MAX_CORRECTION_DISTANCE = 8


def detect_recovery_events(
    events: list[ToolCallEvent],
    signals: list[ErrorSignal],
) -> RecoveryDetectionResult:
    """Find error-correction pairs in a tool-call sequence.

    For each error event (signal != NONE and signal != BLOCKED), scans
    forward for the nearest successful event that qualifies as a
    correction (same tool or target, with different params; or a
    tactical switch to a different tool/target).

    Args:
        events: Tool call events in temporal order (by global_index).
        signals: Parallel list of classified error signals (from extractor).

    Returns:
        RecoveryDetectionResult with all detected pairs.
    """
    if len(events) != len(signals):
        raise ValueError(
            f"events and signals must have same length: {len(events)} != {len(signals)}"
        )

    recovery_events: list[RecoveryEvent] = []
    # Track which error events have already been matched to avoid
    # double-counting the same error in multiple pairs
    matched_errors: set[int] = set()

    for i, (evt, sig) in enumerate(zip(events, signals)):
        # Skip non-error events and already matched errors
        if sig in (ErrorSignal.NONE, ErrorSignal.BLOCKED):
            continue
        if i in matched_errors:
            continue

        # Scan forward for a correction
        best_correction = _find_correction(events, signals, i)
        if best_correction is not None:
            corr_idx, level, inv_count = best_correction
            recovery_events.append(RecoveryEvent(
                error_index=evt.global_index,
                correction_index=events[corr_idx].global_index,
                error_signal=sig,
                level=level,
                investigation_count=inv_count,
            ))
            matched_errors.add(i)

    return RecoveryDetectionResult(recovery_events=recovery_events)


def _find_correction(
    events: list[ToolCallEvent],
    signals: list[ErrorSignal],
    error_idx: int,
) -> tuple[int, RecoveryLevel, int] | None:
    """Find the best correction event for error at position error_idx.

    Returns (correction_list_index, level, investigation_count) or None.
    """
    error_evt = events[error_idx]
    error_key = error_evt.temporal_key()

    for j in range(error_idx + 1, min(error_idx + 1 + _MAX_CORRECTION_DISTANCE, len(events))):
        candidate = events[j]

        # Must be strictly later in temporal order
        if candidate.temporal_key() <= error_key:
            continue

        # Skip other errors (they're not corrections)
        if signals[j] not in (ErrorSignal.NONE,):
            continue

        # Must be a successful action
        if not candidate.result_success:
            continue

        # Check if this qualifies as a correction
        if not _is_correction_for(error_evt, candidate):
            continue

        # Must have materially different params if same tool
        if (error_evt.tool_name == candidate.tool_name
                and not _params_materially_differ(error_evt, candidate)):
            continue

        # Found a correction — classify level
        events_between = events[error_idx + 1: j]
        level = _classify_level(error_evt, candidate, events_between)
        inv_count = sum(1 for e in events_between if _is_investigation(e))

        return (j, level, inv_count)

    return None


def _is_correction_for(error_evt: ToolCallEvent, candidate: ToolCallEvent) -> bool:
    """Check if candidate could be a correction for error_evt.

    A correction is one of:
    1. Same tool_name with different params (retry/fix)
    2. Different tool_name but plausibly addressing the same goal
       (e.g., bash → search, file_edit on different file)
    3. Same target_file (working on the same file)
    """
    # Same tool — likely a retry with different params
    if error_evt.tool_name == candidate.tool_name:
        return True

    # Same target file — different tool on same file
    if (error_evt.target_file and candidate.target_file
            and error_evt.target_file == candidate.target_file):
        return True

    # Tactical switch: bash error → search (or vice versa)
    tool_pair = frozenset({error_evt.tool_name, candidate.tool_name})
    tactical_pairs = [
        frozenset({"bash", "search"}),
        frozenset({"bash", "file_read"}),
        frozenset({"file_edit", "file_create"}),
    ]
    if tool_pair in tactical_pairs:
        return True

    # Write tool after error (bash error → file_edit fix)
    if candidate.tool_name in ("file_edit", "file_create"):
        return True

    return False
