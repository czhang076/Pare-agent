"""Map classifier outputs to human-readable report labels.

Single source of truth for label strings; the renderer must not invent
its own. Category keys are the strings emitted by
``pare.trajectory.classifier_liu.LiuClassification.categories`` and the
``name`` of ``OutcomeLabel`` members.
"""

from __future__ import annotations

LIU_LABELS: dict[str, str] = {
    "A1":   "Missing Context",
    "A2":   "Mislocalization",
    "B1.1": "Incomplete Fix",
    "B1.2": "Insufficient Testing",
    "B2.1": "Logic Error",
    "B2.2": "Syntax Error After Edit",
    "C1":   "False Negative (test was wrong)",
    "C2":   "Premature Success (reward hacking)",
}

OUTCOME_LABELS: dict[str, str] = {
    "VERIFIED_ONE_SHOT":      "Verified — one shot",
    "VERIFIED_WITH_RECOVERY": "Verified — with recovery",
    "WEAKLY_VERIFIED":        "Weakly verified",
    "FAILED":                 "Failed",
    "TOXIC":                  "Toxic (succeeded by cheating)",
}


def liu_label(code: str) -> str:
    """Human-readable label for a Liu classification code."""
    return LIU_LABELS.get(code, code)


def outcome_label(code: str) -> str:
    """Human-readable label for an outcome enum value."""
    return OUTCOME_LABELS.get(code, code)
