"""Tests for Liu et al. failure taxonomy classifier (core 4 categories).

Tests B2.1, B2.2, C1, C2 detection plus trajectory-level outcome labels.
Uses synthetic scenarios and validates on pilot trajectory data.
"""

from __future__ import annotations

import pytest

from pare.trajectory.classifier_liu import (
    LiuClassification,
    OutcomeLabel,
    assign_outcome_label,
    classify_liu,
    detect_a1_missing_context,
    detect_a2_mislocalization,
    detect_b11_incomplete_fix,
    detect_b12_insufficient_testing,
    detect_b21_logic_error,
    detect_b22_syntax_error,
    detect_c1_false_negative,
    detect_c2_premature_success,
)
from pare.trajectory.schema import VerificationResult
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    *,
    tool_name: str = "bash",
    params: dict | None = None,
    target_file: str = "",
    result_success: bool = True,
    result_content: str = "",
    error_signal: ErrorSignal = ErrorSignal.NONE,
) -> ToolCallEvent:
    return ToolCallEvent(
        turn_id=0,
        call_index_in_turn=0,
        global_index=0,
        tool_name=tool_name,
        params=params or {},
        params_hash="",
        target_file=target_file,
        result_success=result_success,
        result_content=result_content,
        error_signal=error_signal,
    )


def _ver(
    *,
    tier1: bool = True,
    tier2: bool = True,
    tier2_cmd: str = "pytest tests/",
) -> VerificationResult:
    return VerificationResult(
        final_passed=tier1 and tier2,
        tier1_pass=tier1,
        tier2_pass=tier2,
        tier2_command=tier2_cmd,
    )


# ======================================================================
# C2: Premature Success
# ======================================================================


class TestC2PrematureSuccess:
    def test_agent_claims_success_tier1_fails(self):
        assert detect_c2_premature_success(True, _ver(tier1=False)) is True

    def test_agent_claims_success_tier1_passes(self):
        assert detect_c2_premature_success(True, _ver(tier1=True)) is False

    def test_agent_claims_failure(self):
        """Agent doesn't claim success → never C2."""
        assert detect_c2_premature_success(False, _ver(tier1=False)) is False

    def test_both_tiers_fail_but_agent_claims_success(self):
        assert detect_c2_premature_success(
            True, _ver(tier1=False, tier2=False)
        ) is True

    def test_tier2_not_configured(self):
        """C2 only depends on tier1, not tier2."""
        assert detect_c2_premature_success(
            True, _ver(tier1=False, tier2_cmd="")
        ) is True


# ======================================================================
# C1: False Negative
# ======================================================================


class TestC1FalseNegative:
    def test_agent_claims_failure_tier2_passes(self):
        assert detect_c1_false_negative(False, _ver(tier2=True)) is True

    def test_agent_claims_success(self):
        """Agent claims success → never C1."""
        assert detect_c1_false_negative(True, _ver(tier2=True)) is False

    def test_agent_claims_failure_tier2_fails(self):
        """Agent correctly reports failure → not C1."""
        assert detect_c1_false_negative(False, _ver(tier2=False)) is False

    def test_tier2_not_configured(self):
        """No tier2 → can't confirm false negative."""
        assert detect_c1_false_negative(
            False, _ver(tier2=True, tier2_cmd="")
        ) is False


# ======================================================================
# B2.2: Syntax Error (in final state)
# ======================================================================


class TestB22SyntaxError:
    def test_unresolved_syntax_error(self):
        """file_edit with SYNTAX_ERROR, never fixed → B2.2."""
        events = [
            _evt(
                tool_name="file_edit",
                target_file="main.py",
                result_success=True,
                result_content="⚠ SYNTAX ERROR: line 10",
            ),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR]
        assert detect_b22_syntax_error(events, signals) is True

    def test_resolved_syntax_error(self):
        """Syntax error followed by successful fix → NOT B2.2."""
        events = [
            _evt(
                tool_name="file_edit",
                target_file="main.py",
                result_success=True,
                result_content="⚠ SYNTAX ERROR: line 10",
            ),
            _evt(
                tool_name="file_edit",
                target_file="main.py",
                result_success=True,
                result_content="File edited successfully.",
            ),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR, ErrorSignal.NONE]
        assert detect_b22_syntax_error(events, signals) is False

    def test_multiple_files_one_broken(self):
        """Two files edited, one has unresolved syntax error → B2.2."""
        events = [
            _evt(tool_name="file_edit", target_file="a.py",
                 result_success=True, result_content="OK"),
            _evt(tool_name="file_edit", target_file="b.py",
                 result_success=True, result_content="⚠ SYNTAX ERROR"),
        ]
        signals = [ErrorSignal.NONE, ErrorSignal.SYNTAX_ERROR]
        assert detect_b22_syntax_error(events, signals) is True

    def test_non_python_file_ignored(self):
        """Syntax error on non-.py file → NOT B2.2."""
        events = [
            _evt(tool_name="file_edit", target_file="config.yaml",
                 result_success=True, result_content="⚠ SYNTAX ERROR"),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR]
        assert detect_b22_syntax_error(events, signals) is False

    def test_non_edit_tool_ignored(self):
        """SYNTAX_ERROR from bash (not file_edit) → NOT B2.2."""
        events = [
            _evt(tool_name="bash",
                 result_content="SyntaxError: invalid syntax"),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR]
        assert detect_b22_syntax_error(events, signals) is False

    def test_file_create_counted(self):
        """file_create with syntax error → B2.2."""
        events = [
            _evt(tool_name="file_create", target_file="new.py",
                 result_success=True, result_content="⚠ SYNTAX ERROR"),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR]
        assert detect_b22_syntax_error(events, signals) is True

    def test_no_edits(self):
        """No file_edit/file_create → NOT B2.2."""
        events = [
            _evt(tool_name="bash", result_content="ok"),
            _evt(tool_name="file_read", target_file="main.py"),
        ]
        signals = [ErrorSignal.NONE, ErrorSignal.NONE]
        assert detect_b22_syntax_error(events, signals) is False

    def test_empty_events(self):
        assert detect_b22_syntax_error([], []) is False

    def test_error_then_fix_then_error(self):
        """Error → fix → error again → B2.2 (final state has error)."""
        events = [
            _evt(tool_name="file_edit", target_file="x.py",
                 result_success=True, result_content="⚠ SYNTAX ERROR"),
            _evt(tool_name="file_edit", target_file="x.py",
                 result_success=True, result_content="OK"),
            _evt(tool_name="file_edit", target_file="x.py",
                 result_success=True, result_content="⚠ SYNTAX ERROR"),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR, ErrorSignal.NONE, ErrorSignal.SYNTAX_ERROR]
        assert detect_b22_syntax_error(events, signals) is True


# ======================================================================
# B2.1: Logic Error
# ======================================================================


class TestB21LogicError:
    def test_tier2_fails_with_assertion(self):
        """TEST_FAILURE with AssertionError → B2.1."""
        events = [
            _evt(
                tool_name="bash",
                params={"command": "python -m pytest tests/ -v"},
                result_success=False,
                result_content=(
                    "FAILED tests/test_foo.py::test_bar\n"
                    "AssertionError: Expected 1, got 2\n"
                    "===== 1 failed =====\n"
                ),
            ),
        ]
        signals = [ErrorSignal.TEST_FAILURE]
        ver = _ver(tier2=False)
        assert detect_b21_logic_error(events, signals, ver) is True

    def test_tier2_passes(self):
        """Tier 2 passes → NOT B2.1."""
        events = [
            _evt(tool_name="bash",
                 params={"command": "pytest tests/"},
                 result_content="1 passed"),
        ]
        signals = [ErrorSignal.NONE]
        assert detect_b21_logic_error(events, signals, _ver(tier2=True)) is False

    def test_tier2_not_configured(self):
        """Tier 2 not configured → NOT B2.1."""
        events = []
        signals = []
        assert detect_b21_logic_error(
            events, signals, _ver(tier2=False, tier2_cmd="")
        ) is False

    def test_test_failure_is_syntax(self):
        """TEST_FAILURE with SyntaxError → NOT B2.1 (it's B2.2)."""
        events = [
            _evt(
                tool_name="bash",
                params={"command": "python -m pytest tests/"},
                result_success=False,
                result_content=(
                    "FAILED tests/test_foo.py\n"
                    "SyntaxError: invalid syntax\n"
                ),
            ),
        ]
        signals = [ErrorSignal.TEST_FAILURE]
        assert detect_b21_logic_error(events, signals, _ver(tier2=False)) is False

    def test_test_failure_is_import(self):
        """TEST_FAILURE with ImportError → NOT B2.1."""
        events = [
            _evt(
                tool_name="bash",
                params={"command": "pytest tests/"},
                result_success=False,
                result_content=(
                    "FAILED\n"
                    "ImportError: cannot import name 'foo'\n"
                ),
            ),
        ]
        signals = [ErrorSignal.TEST_FAILURE]
        assert detect_b21_logic_error(events, signals, _ver(tier2=False)) is False

    def test_fallback_no_test_events_no_syntax(self):
        """Tier 2 fails, no TEST_FAILURE events, no B2.2 → B2.1 fallback."""
        events = [
            _evt(tool_name="file_edit", target_file="main.py",
                 result_success=True, result_content="OK"),
        ]
        signals = [ErrorSignal.NONE]
        assert detect_b21_logic_error(events, signals, _ver(tier2=False)) is True

    def test_fallback_no_test_events_with_syntax(self):
        """Tier 2 fails, no TEST_FAILURE events, has B2.2 → NOT B2.1."""
        events = [
            _evt(tool_name="file_edit", target_file="main.py",
                 result_success=True, result_content="⚠ SYNTAX ERROR"),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR]
        assert detect_b21_logic_error(events, signals, _ver(tier2=False)) is False

    def test_tier1_fails_not_b21(self):
        """Tier 1 fails (e.g. empty diff or syntax check) → NOT B2.1.

        Logic error presupposes code that compiles and runs. If Tier 1
        fails, the problem is structural (empty diff, syntax error at
        the top level) and should not be labeled as Logic Error even
        if no TEST_FAILURE events are present.
        """
        events: list[ToolCallEvent] = []
        signals: list[ErrorSignal] = []
        # Agent gave up without making edits: tier1 fails (empty diff),
        # tier2 configured and fails, no test events, no B2.2
        ver = _ver(tier1=False, tier2=False)
        assert detect_b21_logic_error(events, signals, ver) is False

    def test_tier1_fails_with_edits_no_b21(self):
        """Tier 1 fails despite agent making edits → still NOT B2.1."""
        events = [
            _evt(tool_name="file_edit", target_file="main.py",
                 result_success=True, result_content="OK"),
        ]
        signals = [ErrorSignal.NONE]
        ver = _ver(tier1=False, tier2=False)
        assert detect_b21_logic_error(events, signals, ver) is False


# ======================================================================
# classify_liu — combined classifier
# ======================================================================


class TestClassifyLiu:
    def test_clean_trajectory(self):
        """All passing, no issues → no categories."""
        events = [
            _evt(tool_name="file_read", target_file="x.py"),
            _evt(tool_name="file_edit", target_file="x.py"),
            _evt(tool_name="bash", params={"command": "pytest tests/"}),
        ]
        signals = [ErrorSignal.NONE, ErrorSignal.NONE, ErrorSignal.NONE]
        result = classify_liu(events, signals, True, _ver())
        assert result.categories == []
        assert result.is_toxic is False

    def test_toxic_c2(self):
        """C2 → toxic."""
        result = classify_liu([], [], True, _ver(tier1=False))
        assert result.c2_premature_success is True
        assert result.is_toxic is True
        assert "C2" in result.categories

    def test_toxic_b22(self):
        """B2.2 → toxic."""
        events = [
            _evt(tool_name="file_edit", target_file="x.py",
                 result_content="⚠ SYNTAX ERROR"),
        ]
        signals = [ErrorSignal.SYNTAX_ERROR]
        result = classify_liu(events, signals, False, _ver(tier2=False))
        assert result.b22_syntax_error is True
        assert result.is_toxic is True

    def test_c2_suppresses_b21(self):
        """§3.1.1: when both B2.1 and C2 apply, C2 takes priority."""
        events = [
            _evt(tool_name="bash",
                 params={"command": "pytest tests/"},
                 result_success=False,
                 result_content="FAILED\nAssertionError\n"),
        ]
        signals = [ErrorSignal.TEST_FAILURE]
        # Agent claims success, tier1 fails (C2), tier2 also fails (B2.1 candidate)
        ver = _ver(tier1=False, tier2=False)
        result = classify_liu(events, signals, True, ver)
        assert result.c2_premature_success is True
        assert result.b21_logic_error is False  # Suppressed by C2

    def test_c1_and_others_independent(self):
        """C1 can coexist with other categories (unusual but possible)."""
        # Agent claims failure, tier2 passes (C1), but has syntax error (B2.2)
        # Read + pytest invocation avoid tripping A1/B1.2.
        events = [
            _evt(tool_name="file_read", target_file="x.py"),
            _evt(tool_name="file_edit", target_file="x.py",
                 result_content="⚠ SYNTAX ERROR"),
            _evt(tool_name="bash", params={"command": "pytest tests/"}),
        ]
        signals = [ErrorSignal.NONE, ErrorSignal.SYNTAX_ERROR, ErrorSignal.NONE]
        result = classify_liu(events, signals, False, _ver(tier2=True))
        assert result.c1_false_negative is True
        assert result.b22_syntax_error is True
        assert sorted(result.categories) == ["B2.2", "C1"]

    def test_to_dict(self):
        result = LiuClassification(
            b21_logic_error=True, c2_premature_success=False,
        )
        d = result.to_dict()
        assert d["b21_logic_error"] is True
        assert d["b22_syntax_error"] is False
        assert d["is_toxic"] is False
        assert d["categories"] == ["B2.1"]


# ======================================================================
# OutcomeLabel assignment
# ======================================================================


class TestOutcomeLabel:
    def test_toxic_from_c2(self):
        liu = LiuClassification(c2_premature_success=True)
        label = assign_outcome_label(liu, _ver(tier1=False), False)
        assert label == OutcomeLabel.TOXIC

    def test_toxic_from_b22(self):
        liu = LiuClassification(b22_syntax_error=True)
        label = assign_outcome_label(liu, _ver(tier2=False), False)
        assert label == OutcomeLabel.TOXIC

    def test_failed(self):
        liu = LiuClassification(b21_logic_error=True)
        label = assign_outcome_label(liu, _ver(tier2=False), False)
        assert label == OutcomeLabel.FAILED

    def test_weakly_verified(self):
        liu = LiuClassification()
        ver = _ver(tier1=True, tier2=False, tier2_cmd="")
        label = assign_outcome_label(liu, ver, False)
        assert label == OutcomeLabel.WEAKLY_VERIFIED

    def test_verified_one_shot(self):
        liu = LiuClassification()
        label = assign_outcome_label(liu, _ver(), False)
        assert label == OutcomeLabel.VERIFIED_ONE_SHOT

    def test_verified_with_recovery(self):
        liu = LiuClassification()
        label = assign_outcome_label(liu, _ver(), True)
        assert label == OutcomeLabel.VERIFIED_WITH_RECOVERY

    def test_failed_no_tier2_no_tier1(self):
        """Neither tier passes, tier2 not configured → FAILED."""
        liu = LiuClassification()
        ver = _ver(tier1=False, tier2=False, tier2_cmd="")
        label = assign_outcome_label(liu, ver, False)
        assert label == OutcomeLabel.FAILED

    def test_toxic_overrides_failed(self):
        """Toxic takes priority over failed."""
        liu = LiuClassification(b22_syntax_error=True, b21_logic_error=True)
        label = assign_outcome_label(liu, _ver(tier2=False), False)
        assert label == OutcomeLabel.TOXIC


# ======================================================================
# Pilot trajectory validation
# ======================================================================


class TestPilotValidation:
    """Validate classifier on real pilot trajectory."""

    @pytest.fixture()
    def pilot_record(self):
        import json
        from pathlib import Path

        from pare.trajectory.schema_v2 import ToolCallEvent as TCEvent

        traj_path = (
            Path(__file__).parent.parent.parent
            / "data" / "pilot" / "traj_v2_minimax_flat.jsonl"
        )
        if not traj_path.exists():
            pytest.skip("Pilot trajectory not available")

        with open(traj_path, encoding="utf-8", errors="surrogateescape") as f:
            raw = f.readline()

        decoder = json.JSONDecoder(strict=False)
        rec, _ = decoder.raw_decode(raw)
        raw_events = rec.get("tool_call_events", [])
        events = [TCEvent.from_dict(e) for e in raw_events]

        verification = VerificationResult(
            final_passed=rec.get("verification", {}).get("final_passed", False),
            tier1_pass=rec.get("verification", {}).get("tier1_pass", False),
            tier2_pass=rec.get("verification", {}).get("tier2_pass", False),
            tier2_command=rec.get("verification", {}).get("tier2_command", ""),
        )
        llm_claimed = rec.get("llm_claimed_success", False)

        return events, verification, llm_claimed

    def test_pilot_no_syntax_error_in_final_state(self, pilot_record):
        """Pilot has edits on .py files but no unresolved syntax errors."""
        events, _, _ = pilot_record
        from pare.trajectory.error_signal_extractor import classify_trajectory_signals
        signals = classify_trajectory_signals(events)
        assert detect_b22_syntax_error(events, signals) is False

    def test_pilot_classification(self, pilot_record):
        """Full classification of pilot trajectory."""
        events, verification, llm_claimed = pilot_record
        from pare.trajectory.error_signal_extractor import classify_trajectory_signals
        signals = classify_trajectory_signals(events)

        result = classify_liu(events, signals, llm_claimed, verification)
        # Pilot trajectory: agent made edits, no unresolved syntax errors
        assert result.b22_syntax_error is False
        # Specific C1/C2 depend on verification results — just ensure no crash
        assert isinstance(result.categories, list)
        assert isinstance(result.is_toxic, bool)


# ======================================================================
# A1: Missing Context
# ======================================================================


class TestA1MissingContext:
    def test_edit_without_read(self):
        """file_edit on file never read → A1."""
        events = [
            _evt(tool_name="file_edit", target_file="main.py"),
        ]
        assert detect_a1_missing_context(events) is True

    def test_edit_after_read(self):
        """file_read then file_edit same target → NOT A1."""
        events = [
            _evt(tool_name="file_read", target_file="main.py"),
            _evt(tool_name="file_edit", target_file="main.py"),
        ]
        assert detect_a1_missing_context(events) is False

    def test_edit_before_read(self):
        """Order matters: edit before read → A1."""
        events = [
            _evt(tool_name="file_edit", target_file="main.py"),
            _evt(tool_name="file_read", target_file="main.py"),
        ]
        assert detect_a1_missing_context(events) is True

    def test_failed_read_does_not_count(self):
        """A failed file_read means the agent never saw the content."""
        events = [
            _evt(tool_name="file_read", target_file="main.py",
                 result_success=False),
            _evt(tool_name="file_edit", target_file="main.py"),
        ]
        assert detect_a1_missing_context(events) is True

    def test_different_files(self):
        """Read of a.py doesn't excuse edit of b.py."""
        events = [
            _evt(tool_name="file_read", target_file="a.py"),
            _evt(tool_name="file_edit", target_file="b.py"),
        ]
        assert detect_a1_missing_context(events) is True

    def test_file_create_not_a1(self):
        """file_create (new file) does not require prior read."""
        events = [
            _evt(tool_name="file_create", target_file="new.py"),
        ]
        assert detect_a1_missing_context(events) is False

    def test_multiple_edits_all_safe(self):
        """Every edit preceded by matching read → NOT A1."""
        events = [
            _evt(tool_name="file_read", target_file="a.py"),
            _evt(tool_name="file_read", target_file="b.py"),
            _evt(tool_name="file_edit", target_file="a.py"),
            _evt(tool_name="file_edit", target_file="b.py"),
        ]
        assert detect_a1_missing_context(events) is False

    def test_multiple_edits_one_bad(self):
        """One edit lacks a prior read → A1."""
        events = [
            _evt(tool_name="file_read", target_file="a.py"),
            _evt(tool_name="file_edit", target_file="a.py"),
            _evt(tool_name="file_edit", target_file="b.py"),
        ]
        assert detect_a1_missing_context(events) is True

    def test_empty_events(self):
        assert detect_a1_missing_context([]) is False

    def test_edit_with_no_target_ignored(self):
        """Edit with empty target_file → skip."""
        events = [
            _evt(tool_name="file_edit", target_file=""),
        ]
        assert detect_a1_missing_context(events) is False


# ======================================================================
# A2: Mislocalization
# ======================================================================


class TestA2Mislocalization:
    def test_edit_referenced_file(self):
        """Error references foo.py, agent edits foo.py → NOT A2."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content='File "foo.py", line 5\nNameError\n',
                 error_signal=ErrorSignal.RUNTIME_ERROR),
            _evt(tool_name="file_edit", target_file="foo.py"),
        ]
        signals = [ErrorSignal.RUNTIME_ERROR, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is False

    def test_edit_wrong_file(self):
        """Error references foo.py, agent edits bar.py → A2."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content='File "foo.py", line 5\nNameError\n',
                 error_signal=ErrorSignal.RUNTIME_ERROR),
            _evt(tool_name="file_edit", target_file="bar.py"),
        ]
        signals = [ErrorSignal.RUNTIME_ERROR, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is True

    def test_pytest_short_format(self):
        """pytest 'path/file.py:N' format is parsed."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content="FAILED tests/test_foo.py:42\nAssertionError\n",
                 error_signal=ErrorSignal.TEST_FAILURE),
            _evt(tool_name="file_edit", target_file="src/other.py"),
        ]
        signals = [ErrorSignal.TEST_FAILURE, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is True

    def test_partial_overlap_not_a2(self):
        """At least one edit on a referenced file → NOT A2."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content='File "foo.py", line 5',
                 error_signal=ErrorSignal.RUNTIME_ERROR),
            _evt(tool_name="file_edit", target_file="foo.py"),
            _evt(tool_name="file_edit", target_file="unrelated.py"),
        ]
        signals = [ErrorSignal.RUNTIME_ERROR, ErrorSignal.NONE, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is False

    def test_no_error_events(self):
        """No error refs → can't diagnose mislocalization."""
        events = [_evt(tool_name="file_edit", target_file="foo.py")]
        signals = [ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is False

    def test_no_edit_events(self):
        """No edits → can't mislocalize."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content='File "foo.py", line 5',
                 error_signal=ErrorSignal.RUNTIME_ERROR),
        ]
        signals = [ErrorSignal.RUNTIME_ERROR]
        assert detect_a2_mislocalization(events, signals) is False

    def test_path_separator_normalization(self):
        """Windows-style paths normalized to match forward-slash edits."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content='File "src\\foo.py", line 5',
                 error_signal=ErrorSignal.RUNTIME_ERROR),
            _evt(tool_name="file_edit", target_file="src/foo.py"),
        ]
        signals = [ErrorSignal.RUNTIME_ERROR, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is False

    def test_multiple_refs_edit_hits_one(self):
        """Error refs foo.py and bar.py; edit on foo.py → NOT A2."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content=(
                     'File "foo.py", line 5\n'
                     'File "bar.py", line 10\n'
                 ),
                 error_signal=ErrorSignal.TEST_FAILURE),
            _evt(tool_name="file_edit", target_file="foo.py"),
        ]
        signals = [ErrorSignal.TEST_FAILURE, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is False

    def test_non_error_signal_ignored(self):
        """Non-error signals don't contribute file refs."""
        events = [
            _evt(tool_name="bash",
                 result_content='File "foo.py", line 5',
                 error_signal=ErrorSignal.NONE),
            _evt(tool_name="file_edit", target_file="bar.py"),
        ]
        signals = [ErrorSignal.NONE, ErrorSignal.NONE]
        assert detect_a2_mislocalization(events, signals) is False


# ======================================================================
# B1.1: Incomplete Fix
# ======================================================================


class TestB11IncompleteFix:
    _GOLD_TWO_FILES = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,2 +5,2 @@
-y = 3
+y = 4
"""

    _FINAL_ONE_FILE = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
"""

    _FINAL_MATCHING = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,2 +5,2 @@
-y = 3
+y = 4
"""

    def test_fewer_files_than_gold(self):
        assert detect_b11_incomplete_fix(
            self._FINAL_ONE_FILE, self._GOLD_TWO_FILES
        ) is True

    def test_matching_diff(self):
        assert detect_b11_incomplete_fix(
            self._FINAL_MATCHING, self._GOLD_TWO_FILES
        ) is False

    def test_fewer_hunks_same_files(self):
        """Same file count but fewer hunks → B1.1."""
        gold = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
@@ -20,2 +20,2 @@
-y = 1
+y = 2
"""
        final = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
"""
        assert detect_b11_incomplete_fix(final, gold) is True

    def test_empty_gold(self):
        """No gold → can't compare, return False."""
        assert detect_b11_incomplete_fix(self._FINAL_ONE_FILE, "") is False

    def test_empty_final(self):
        """Empty final diff vs. non-empty gold → B1.1."""
        assert detect_b11_incomplete_fix("", self._GOLD_TWO_FILES) is True

    def test_both_empty(self):
        assert detect_b11_incomplete_fix("", "") is False

    def test_more_files_than_gold(self):
        """Final covers more than gold — not B1.1 (different category)."""
        bigger = self._GOLD_TWO_FILES + """\
diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1,1 +1,1 @@
-z = 1
+z = 2
"""
        assert detect_b11_incomplete_fix(bigger, self._GOLD_TWO_FILES) is False

    def test_plus_plus_fallback(self):
        """Diff without 'diff --git' lines still parsed via +++ b/ lines."""
        gold = """\
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
--- a/b.py
+++ b/b.py
@@ -5,2 +5,2 @@
-y = 3
+y = 4
"""
        final = """\
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
"""
        assert detect_b11_incomplete_fix(final, gold) is True


# ======================================================================
# B1.2: Insufficient Testing
# ======================================================================


class TestB12InsufficientTesting:
    def test_no_tier2_not_applicable(self):
        """No tier2 configured → False."""
        events = []
        assert detect_b12_insufficient_testing(
            events, _ver(tier2_cmd="")
        ) is False

    def test_pytest_invocation(self):
        events = [
            _evt(tool_name="bash", params={"command": "pytest tests/"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is False

    def test_python_m_pytest(self):
        events = [
            _evt(tool_name="bash",
                 params={"command": "python -m pytest -xvs"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is False

    def test_unittest_runner(self):
        events = [
            _evt(tool_name="bash",
                 params={"command": "python -m unittest discover"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is False

    def test_manage_py_test(self):
        events = [
            _evt(tool_name="bash",
                 params={"command": "python manage.py test myapp"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is False

    def test_tox(self):
        events = [
            _evt(tool_name="bash", params={"command": "tox -e py312"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is False

    def test_nosetests(self):
        events = [
            _evt(tool_name="bash", params={"command": "nosetests -v"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is False

    def test_no_test_runner(self):
        """tier2 configured but no test command → B1.2."""
        events = [
            _evt(tool_name="bash", params={"command": "ls -la"}),
            _evt(tool_name="file_edit", target_file="main.py"),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is True

    def test_substring_false_positive_avoided(self):
        """'pytestfixture' is NOT matched as 'pytest' (word boundary required).

        The agent ran `grep pytestfixture` — this grep does not exercise
        tests. Word-boundary regex correctly rejects the substring match,
        so B1.2 fires (no test runner actually invoked).
        """
        events = [
            _evt(tool_name="bash",
                 params={"command": "grep pytestfixture *.py"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is True

    def test_empty_events_with_tier2(self):
        events: list[ToolCallEvent] = []
        assert detect_b12_insufficient_testing(events, _ver()) is True

    def test_non_bash_tool_ignored(self):
        """Test-runner keywords in file_edit params don't count."""
        events = [
            _evt(tool_name="file_edit", target_file="pytest_helper.py",
                 params={"file_path": "pytest_helper.py"}),
        ]
        assert detect_b12_insufficient_testing(events, _ver()) is True


# ======================================================================
# classify_liu — extended categories integration
# ======================================================================


class TestClassifyLiuExtended:
    def test_a1_populated(self):
        """A1 fires when edit has no prior read."""
        events = [_evt(tool_name="file_edit", target_file="x.py")]
        signals = [ErrorSignal.NONE]
        result = classify_liu(events, signals, True, _ver())
        assert result.a1_missing_context is True
        assert "A1" in result.categories

    def test_a2_populated(self):
        """A2 fires when edit misses referenced file."""
        events = [
            _evt(tool_name="bash",
                 result_success=False,
                 result_content='File "foo.py", line 5',
                 error_signal=ErrorSignal.RUNTIME_ERROR),
            _evt(tool_name="file_edit", target_file="bar.py"),
        ]
        signals = [ErrorSignal.RUNTIME_ERROR, ErrorSignal.NONE]
        result = classify_liu(events, signals, True, _ver())
        assert result.a2_mislocalization is True
        assert "A2" in result.categories

    def test_b11_populated(self):
        """B1.1 fires when final diff is smaller than gold."""
        gold = """\
diff --git a/a.py b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
diff --git a/b.py b/b.py
@@ -1,2 +1,2 @@
-y = 1
+y = 2
"""
        final = """\
diff --git a/a.py b/a.py
@@ -1,3 +1,3 @@
-x = 1
+x = 2
"""
        result = classify_liu(
            [], [], True, _ver(),
            final_diff=final, gold_patch=gold,
        )
        assert result.b11_incomplete_fix is True
        assert "B1.1" in result.categories

    def test_b12_populated(self):
        """B1.2 fires when tier2 set but no test runner called."""
        events = [
            _evt(tool_name="file_edit", target_file="main.py"),
        ]
        signals = [ErrorSignal.NONE]
        result = classify_liu(events, signals, True, _ver())
        assert result.b12_insufficient_testing is True
        assert "B1.2" in result.categories

    def test_extended_do_not_affect_is_toxic(self):
        """A1/A2/B1.1/B1.2 alone must not mark trajectory toxic."""
        result = LiuClassification(
            a1_missing_context=True,
            a2_mislocalization=True,
            b11_incomplete_fix=True,
            b12_insufficient_testing=True,
        )
        assert result.is_toxic is False

    def test_to_dict_includes_extended(self):
        result = LiuClassification(
            a1_missing_context=True,
            b11_incomplete_fix=True,
        )
        d = result.to_dict()
        assert d["a1_missing_context"] is True
        assert d["a2_mislocalization"] is False
        assert d["b11_incomplete_fix"] is True
        assert d["b12_insufficient_testing"] is False
        assert sorted(d["categories"]) == ["A1", "B1.1"]

    def test_c2_does_not_suppress_extended(self):
        """Priority rule is C2-over-B2.1 only; extended cats co-occur."""
        events = [
            _evt(tool_name="file_edit", target_file="x.py"),
        ]
        signals = [ErrorSignal.NONE]
        # Agent claims success, tier1 fails → C2
        result = classify_liu(events, signals, True, _ver(tier1=False))
        assert result.c2_premature_success is True
        assert result.a1_missing_context is True  # Not suppressed
        assert result.b12_insufficient_testing is True  # Not suppressed

    def test_category_ordering(self):
        """categories list is ordered A → B → C."""
        result = LiuClassification(
            a1_missing_context=True,
            a2_mislocalization=True,
            b11_incomplete_fix=True,
            b12_insufficient_testing=True,
            b21_logic_error=True,
            b22_syntax_error=True,
            c1_false_negative=True,
            c2_premature_success=True,
        )
        assert result.categories == [
            "A1", "A2", "B1.1", "B1.2", "B2.1", "B2.2", "C1", "C2"
        ]
