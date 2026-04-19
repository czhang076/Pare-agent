"""Tests for divergence detection — the algorithmic heart of the Inspector.

These tests are the regression gate for the strict-positional alignment
choice (see ``pare/inspector/differ.py`` module docstring). If anyone
later swaps in LCS / DP, the IDENTICAL-coincidence test below should
fail loudly.

R0: tests are skipped placeholders. W1 Day 3-5 fills them in.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="differ implementation lands W1 Day 3-5")


def test_identical_trajectories_have_no_divergence() -> None:
    """Two bit-identical trajectories return DivergencePoint=None."""


def test_first_tool_diff_is_divergence_point() -> None:
    """Step 0: a uses bash, b uses file_read → DivergenceKind.TOOL_DIFF at index 0."""


def test_intent_match_alone_is_not_divergence() -> None:
    """If every step matches (tool, target_file) but params differ, the
    first step is reported as a 'parametric' divergence — but mid-trajectory
    INTENT_MATCH is not, as long as some later TARGET_DRIFT / TOOL_DIFF exists."""


def test_strict_positional_alignment_does_not_resync() -> None:
    """Regression gate against LCS-style realignment.

    a: [bash, file_read("a.py"), file_edit("a.py"), file_read("README")]
    b: [bash, file_edit("a.py"), file_read("a.py"), file_read("README")]

    Steps 1 and 2 are swapped. LCS would re-align step 3 of both as
    matching file_read("README") and report divergence at step 1 only.
    Strict positional alignment must report TARGET_DRIFT at step 1 AND
    keep walking — divergence point is the FIRST one (step 1)."""


def test_length_diff_kind_for_unequal_trajectories() -> None:
    """Trailing tail of the longer trajectory is tagged LENGTH_DIFF."""


def test_pilot4_instance_12489_golden() -> None:
    """Golden test: the failed instance from v6_pilot4 (two declare_done
    in a row) diverges from the gold trajectory at the first file_edit
    that targets a different file. This protects the divergence-detection
    semantics from silent drift."""
