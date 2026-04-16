"""Tests for deterministic recovery event detection (v1 — deprecated).

v1 recovery_detector operates on StepAttempt sequences. After Phase 2.6
cleanup, it no longer gates detection on the removed ``rolled_back``
field — any failed→success attempt pair is treated as a recovery event.
Kept as a regression harness for the legacy module; new code should use
``recovery_detector_v2``.
"""

from __future__ import annotations

import warnings

import pytest

from pare.trajectory.schema import StepAttempt

# v1 module emits a DeprecationWarning on use; we exercise it intentionally.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:pare.trajectory.recovery_detector"
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from pare.trajectory.recovery_detector import (
        RecoveryLevel,
        detect_recovery_events,
        highest_recovery_level,
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


class TestDetectRecoveryEvents:
    def test_l1_retry_same_approach(self):
        attempts = [
            _attempt(1, 1, "failed", files=["a.py"], tools=["file_edit", "file_edit"]),
            _attempt(1, 2, "success", files=["a.py"], tools=["file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 1
        assert events[0].level == RecoveryLevel.L1_RETRY

    def test_l2_strategy_switch(self):
        attempts = [
            _attempt(1, 1, "failed", files=["a.py"], tools=["search", "file_edit"]),
            _attempt(1, 2, "success", files=["b.py"], tools=["bash", "file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 1
        assert events[0].level == RecoveryLevel.L2_STRATEGY_SWITCH

    def test_l3_goal_decomposition_on_later_step(self):
        attempts = [
            _attempt(1, 1, "failed", files=["a.py"], tools=["file_edit"]),
            _attempt(2, 1, "success", goal="Split into sub-step", files=["a.py", "b.py"], tools=["file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 1
        assert events[0].level == RecoveryLevel.L3_GOAL_DECOMPOSITION

    def test_multiple_events_and_highest_level(self):
        attempts = [
            _attempt(1, 1, "failed", files=["a.py"], tools=["file_edit"]),
            _attempt(1, 2, "success", files=["a.py"], tools=["file_edit"]),
            _attempt(2, 1, "failed", files=["b.py"], tools=["search"]),
            _attempt(3, 1, "success", goal="Sub-goal", files=["b.py"], tools=["file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 2
        assert highest_recovery_level(events) == RecoveryLevel.L3_GOAL_DECOMPOSITION
