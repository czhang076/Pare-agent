"""Map research-side classifier outputs to human-readable report labels.

R0 scaffold — table is empty, real values land in W1 Day 4 once
``annotator.py`` is wired. Single source of truth for label strings;
the renderer must not invent its own.
"""

from __future__ import annotations

# Liu et al. 8-category code → display label.
# Values to be filled in W1 once the research-side category enum names
# are confirmed against pare_research.trajectory.classifier_liu.
LIU_LABELS: dict[str, str] = {
    # "A1_MISSING_CONTEXT": "Missing Context",
    # "A2_MISLOCALIZATION": "Mislocalization",
    # "B1_1_INCOMPLETE_FIX": "Incomplete Fix",
    # "B1_2_INSUFFICIENT_TESTING": "Insufficient Testing",
    # "B2_1_LOGIC_ERROR": "Logic Error",
    # "B2_2_SYNTAX_ERROR_AFTER_EDIT": "Syntax Error After Edit",
    # "C1_FALSE_NEGATIVE": "False Negative (test was wrong)",
    # "C2_PREMATURE_SUCCESS": "Premature Success (reward hacking)",
}

# OutcomeLabel → display label.
OUTCOME_LABELS: dict[str, str] = {
    # "VERIFIED_ONE_SHOT": "Verified — one shot",
    # "VERIFIED_WITH_RECOVERY": "Verified — with recovery",
    # "WEAKLY_VERIFIED": "Weakly verified",
    # "FAILED": "Failed",
    # "TOXIC": "Toxic (succeeded by cheating)",
}


def liu_label(code: str) -> str:
    """Human-readable label for a Liu classification code."""
    return LIU_LABELS.get(code, code)


def outcome_label(code: str) -> str:
    """Human-readable label for an outcome enum value."""
    return OUTCOME_LABELS.get(code, code)
