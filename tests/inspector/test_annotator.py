"""Tests for ``pare.inspector.annotator.annotate``.

These tests pin the contract between ``annotate`` and the vendored
classifier chain (error_signal_extractor → recovery_detector_v2 →
classifier_liu → assign_outcome_label). If anything in that pipeline
silently reorders or drops a step, these assertions fail first.
"""

from __future__ import annotations

from pathlib import Path

from pare.inspector.annotator import AnnotatedTrajectory, annotate
from pare.inspector.loader import load_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.jsonl"


def _records() -> list:
    return load_jsonl(FIXTURE)


def test_annotate_verified_one_shot() -> None:
    record_a = _records()[0]
    ann = annotate(record_a)
    assert isinstance(ann, AnnotatedTrajectory)
    assert ann.outcome_label == "VERIFIED_ONE_SHOT"
    assert ann.liu_classification.categories == []


def test_annotate_toxic_flags_b22() -> None:
    record_b = _records()[1]
    ann = annotate(record_b)
    assert "B2.2" in ann.liu_classification.categories
    assert ann.outcome_label == "TOXIC"


def test_annotate_per_step_signals_match_events() -> None:
    record_b = _records()[1]
    ann = annotate(record_b)
    assert len(ann.steps) == len(record_b.tool_call_events)
    for step, event in zip(ann.steps, record_b.tool_call_events, strict=True):
        assert step.event.global_index == event.global_index
        assert step.recovery_label in {"L1", "L2", "L3", "none"}
