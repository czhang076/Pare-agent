"""Tests for deterministic recovery event detection."""

from __future__ import annotations

from pare.trajectory.recovery_detector import (
    RecoveryLevel,
    detect_recovery_events,
    highest_recovery_level,
)
from pare.trajectory.schema import StepAttempt


def _attempt(
    step: int,
    attempt: int,
    status: str,
    *,
    goal: str = "Fix parser",
    rolled_back: bool = False,
    files: list[str] | None = None,
    tools: list[str] | None = None,
) -> StepAttempt:
    return StepAttempt(
        step_number=step,
        attempt_number=attempt,
        goal=goal,
        status=status,
        rolled_back=rolled_back,
        target_files=files or [],
        tool_names=tools or [],
    )


class TestDetectRecoveryEvents:
    def test_l1_retry_same_approach(self):
        attempts = [
            _attempt(1, 1, "failed", rolled_back=True, files=["a.py"], tools=["file_edit", "file_edit"]),
            _attempt(1, 2, "success", files=["a.py"], tools=["file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 1
        assert events[0].level == RecoveryLevel.L1_RETRY

    def test_l2_strategy_switch(self):
        attempts = [
            _attempt(1, 1, "failed", rolled_back=True, files=["a.py"], tools=["search", "file_edit"]),
            _attempt(1, 2, "success", files=["b.py"], tools=["bash", "file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 1
        assert events[0].level == RecoveryLevel.L2_STRATEGY_SWITCH

    def test_l3_goal_decomposition_on_later_step(self):
        attempts = [
            _attempt(1, 1, "failed", rolled_back=True, files=["a.py"], tools=["file_edit"]),
            _attempt(2, 1, "success", goal="Split into sub-step", files=["a.py", "b.py"], tools=["file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 1
        assert events[0].level == RecoveryLevel.L3_GOAL_DECOMPOSITION

    def test_requires_rollback_marker(self):
        attempts = [
            _attempt(1, 1, "failed", rolled_back=False),
            _attempt(1, 2, "success"),
        ]

        events = detect_recovery_events(attempts)
        assert events == []

    def test_multiple_events_and_highest_level(self):
        attempts = [
            _attempt(1, 1, "failed", rolled_back=True, files=["a.py"], tools=["file_edit"]),
            _attempt(1, 2, "success", files=["a.py"], tools=["file_edit"]),
            _attempt(2, 1, "failed", rolled_back=True, files=["b.py"], tools=["search"]),
            _attempt(3, 1, "success", goal="Sub-goal", files=["b.py"], tools=["file_edit"]),
        ]

        events = detect_recovery_events(attempts)
        assert len(events) == 2
        assert highest_recovery_level(events) == RecoveryLevel.L3_GOAL_DECOMPOSITION
