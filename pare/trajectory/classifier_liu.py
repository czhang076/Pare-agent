"""Liu et al. failure taxonomy classifier — core + extended categories.

Implements automated detection for the Liu et al. (2025) 3-stage,
9-category failure taxonomy (plan.md §3.1).

Core 4 (toxic / H2 critical path):
    B2.1: Logic Error      — Tier 2 test fails due to assertion error
    B2.2: Syntax Error     — edited .py file has unresolved syntax error
    C1:   False Negative   — agent claims failure but Tier 2 passes
    C2:   Premature Success — agent claims success but Tier 1 fails

Extended 4 (paper-richness, not H2 critical):
    A1:   Missing Context     — file_edit without prior successful file_read
    A2:   Mislocalization     — error references file G; agent never edits G
    B1.1: Incomplete Fix      — final diff touches fewer files/hunks than gold
    B1.2: Insufficient Testing — tier2 configured, no test-runner bash call

Each detection function is independent and deterministic (no LLM calls).
Priority rule (§3.1.1): when both B2.1 and C2 apply, C2 takes priority.

Toxic label (§3.1.2) is C2 OR B2.2 — extended categories do NOT contribute
to toxicity (they are analytical slices, not filter criteria).

Trajectory-level outcome labels (§3.1.2) are also assigned here:
    toxic, failed, weakly_verified, verified_one_shot, verified_with_recovery
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Sequence

from pare.trajectory.schema import TrajectoryRecord, VerificationResult
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


# ---------------------------------------------------------------------------
# Outcome labels (§3.1.2)
# ---------------------------------------------------------------------------


class OutcomeLabel(str, enum.Enum):
    """Trajectory-level outcome after Liu et al. classification."""

    TOXIC = "toxic"
    FAILED = "failed"
    WEAKLY_VERIFIED = "weakly_verified"
    VERIFIED_ONE_SHOT = "verified_one_shot"
    VERIFIED_WITH_RECOVERY = "verified_with_recovery"


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LiuClassification:
    """Result of Liu et al. classification on a single trajectory.

    Core attributes (contribute to toxicity):
        b21_logic_error: B2.1 — Tier 2 test fails with assertion error.
        b22_syntax_error: B2.2 — Edited .py file has unresolved syntax error.
        c1_false_negative: C1 — Agent reports failure but Tier 2 passes.
        c2_premature_success: C2 — Agent claims success but Tier 1 fails.

    Extended attributes (analytical, do NOT affect is_toxic):
        a1_missing_context: A1 — file_edit without prior successful file_read.
        a2_mislocalization: A2 — error references file G, agent never edits G.
        b11_incomplete_fix: B1.1 — final diff covers fewer files/hunks than gold.
        b12_insufficient_testing: B1.2 — tier2 configured, no test-runner call.
    """

    b21_logic_error: bool = False
    b22_syntax_error: bool = False
    c1_false_negative: bool = False
    c2_premature_success: bool = False
    a1_missing_context: bool = False
    a2_mislocalization: bool = False
    b11_incomplete_fix: bool = False
    b12_insufficient_testing: bool = False

    @property
    def is_toxic(self) -> bool:
        """§3.1.2: toxic = C2 (Premature Success) or B2.2 (Syntax Error).

        Extended categories (A1/A2/B1.1/B1.2) are excluded by design —
        they are analytical slices, not filter criteria for toxic removal.
        """
        return self.c2_premature_success or self.b22_syntax_error

    @property
    def categories(self) -> list[str]:
        """List of active category labels (e.g. ["B2.1", "C2"])."""
        result: list[str] = []
        if self.a1_missing_context:
            result.append("A1")
        if self.a2_mislocalization:
            result.append("A2")
        if self.b11_incomplete_fix:
            result.append("B1.1")
        if self.b12_insufficient_testing:
            result.append("B1.2")
        if self.b21_logic_error:
            result.append("B2.1")
        if self.b22_syntax_error:
            result.append("B2.2")
        if self.c1_false_negative:
            result.append("C1")
        if self.c2_premature_success:
            result.append("C2")
        return result

    def to_dict(self) -> dict:
        return {
            "b21_logic_error": self.b21_logic_error,
            "b22_syntax_error": self.b22_syntax_error,
            "c1_false_negative": self.c1_false_negative,
            "c2_premature_success": self.c2_premature_success,
            "a1_missing_context": self.a1_missing_context,
            "a2_mislocalization": self.a2_mislocalization,
            "b11_incomplete_fix": self.b11_incomplete_fix,
            "b12_insufficient_testing": self.b12_insufficient_testing,
            "is_toxic": self.is_toxic,
            "categories": self.categories,
        }


# ---------------------------------------------------------------------------
# B2.1: Logic Error
# ---------------------------------------------------------------------------

# Patterns that indicate assertion-based failures (logic errors)
_ASSERTION_PATTERNS = re.compile(
    r"AssertionError|AssertError"
    r"|assert\s+.+\bfailed\b"
    r"|FAIL:|FAILED\s"
    r"|failures?=\d+"
    r"|=+\s+FAILURES\s+=+",
    re.IGNORECASE,
)

# Patterns that indicate the test failure is syntax/import, not logic
_NON_LOGIC_PATTERNS = re.compile(
    r"SyntaxError:|IndentationError:|TabError:"
    r"|ImportError:|ModuleNotFoundError:"
    r"|cannot\s+import\s+name",
)


def detect_b21_logic_error(
    events: list[ToolCallEvent],
    signals: list[ErrorSignal],
    verification: VerificationResult,
) -> bool:
    """B2.1: Logic Error — Tier 2 test fails with assertion error.

    Conditions (all must hold):
    1. Tier 1 passed (code is structurally valid — compiles, non-empty diff).
       Logic errors by definition mean the code runs but produces wrong
       results; if Tier 1 fails, the problem is structural, not semantic.
    2. Tier 2 was configured (tier2_command non-empty) AND failed
    3. The trajectory contains TEST_FAILURE events with assertion-pattern
       content that is NOT explained by syntax/import errors

    If Tier 2 failed but no TEST_FAILURE event with assertion content
    exists in the trajectory (e.g. tests were run externally by the
    verifier), falls back to: tier1_pass AND NOT B2.2.
    """
    # Tier 1 must pass — logic error presupposes code that runs
    if not verification.tier1_pass:
        return False
    # Tier 2 must have been configured and failed
    if verification.tier2_pass:
        return False
    if not verification.tier2_command:
        return False

    # Look for TEST_FAILURE events with assertion patterns
    has_test_failure = False
    for evt, sig in zip(events, signals):
        if sig != ErrorSignal.TEST_FAILURE:
            continue
        has_test_failure = True
        content = evt.result_content
        has_assertion = bool(_ASSERTION_PATTERNS.search(content))
        is_syntax_import = bool(_NON_LOGIC_PATTERNS.search(content))
        if has_assertion and not is_syntax_import:
            return True

    # Fallback: tier2 failed, but NO TEST_FAILURE events at all in the
    # trajectory (tests were run externally by the verifier, not by the
    # agent). If B2.2 is also False (no syntax errors), the failure is
    # likely logic.
    if not has_test_failure:
        b22 = detect_b22_syntax_error(events, signals)
        if not b22:
            return True

    return False


# ---------------------------------------------------------------------------
# B2.2: Syntax Error (in final state)
# ---------------------------------------------------------------------------


def detect_b22_syntax_error(
    events: list[ToolCallEvent],
    signals: list[ErrorSignal],
) -> bool:
    """B2.2: Syntax Error — edited .py file has unresolved syntax error.

    Tracks per-file state across the trajectory. For each .py file
    targeted by file_edit or file_create:
    - SYNTAX_ERROR signal → file is in error state
    - Successful edit (NONE signal, result_success) → file is clean

    Returns True if ANY edited .py file is still in error state at
    the end of the trajectory.
    """
    # Track per-file syntax error state: True = has error, False = clean
    file_state: dict[str, bool] = {}

    for evt, sig in zip(events, signals):
        if evt.tool_name not in ("file_edit", "file_create"):
            continue
        if not evt.target_file.endswith(".py"):
            continue

        if sig == ErrorSignal.SYNTAX_ERROR:
            file_state[evt.target_file] = True
        elif evt.result_success and sig == ErrorSignal.NONE:
            file_state[evt.target_file] = False

    return any(file_state.values())


# ---------------------------------------------------------------------------
# C1: False Negative
# ---------------------------------------------------------------------------


def detect_c1_false_negative(
    llm_claimed_success: bool,
    verification: VerificationResult,
) -> bool:
    """C1: False Negative — agent reports failure but Tier 2 passes.

    The agent is too pessimistic: it thinks its changes are wrong,
    but the tests actually pass. Detected at trajectory end only.

    Conditions:
    1. llm_claimed_success == False (agent says it failed)
    2. tier2_pass == True (but tests actually pass)
    3. tier2_command non-empty (Tier 2 was configured)
    """
    if llm_claimed_success:
        return False
    if not verification.tier2_pass:
        return False
    if not verification.tier2_command:
        return False
    return True


# ---------------------------------------------------------------------------
# C2: Premature Success
# ---------------------------------------------------------------------------


def detect_c2_premature_success(
    llm_claimed_success: bool,
    verification: VerificationResult,
) -> bool:
    """C2: Premature Success — agent claims success but Tier 1 fails.

    The agent is deluded: it declares the task done, but even basic
    verification (non-empty diff + files compile) fails. This is the
    most toxic category for SFT — the model would learn to declare
    victory without actually solving the problem.

    Detection is restricted to the agent's final turn (§3.1.1).

    Conditions:
    1. llm_claimed_success == True (agent says it succeeded)
    2. tier1_pass == False (Tier 1 basic check fails)
    """
    if not llm_claimed_success:
        return False
    return not verification.tier1_pass


# ---------------------------------------------------------------------------
# A1: Missing Context
# ---------------------------------------------------------------------------


def detect_a1_missing_context(events: list[ToolCallEvent]) -> bool:
    """A1: Missing Context — file_edit without prior successful file_read.

    Plan.md §3.1: "Agent edits file without prior file_read on that file
    in the same turn sequence." Simplified from Liu et al.'s original
    (which includes import closure analysis).

    Logic:
    - Track set of files that have been successfully file_read'd so far.
    - For each file_edit event, check if target_file is in that set.
    - A failed file_read does NOT count (agent never saw the content).
    - file_create is excluded — new files have nothing to read beforehand.
    - Returns True if ANY file_edit lacked a prior successful read of the
      same target.
    """
    read_files: set[str] = set()

    for evt in events:
        if evt.tool_name == "file_read" and evt.result_success:
            if evt.target_file:
                read_files.add(evt.target_file)
        elif evt.tool_name == "file_edit":
            if evt.target_file and evt.target_file not in read_files:
                return True

    return False


# ---------------------------------------------------------------------------
# A2: Mislocalization
# ---------------------------------------------------------------------------

# File references in error tracebacks / test failures. Captures common
# forms; conservative on false positives (requires a .py extension).
_FILE_REF_PATTERNS = [
    # Python traceback: File "path/to/foo.py", line 42
    re.compile(r'File\s+"([^"\n]+\.py)"'),
    # pytest short / flake8: path/to/foo.py:42
    re.compile(r'(?:^|[\s(])([A-Za-z0-9_./\\-]+\.py):\d+'),
    # Generic: in path/to/foo.py
    re.compile(r'\bin\s+([A-Za-z0-9_./\\-]+\.py)\b'),
]

_ERROR_SIGNALS_WITH_FILE_REFS = {
    ErrorSignal.TEST_FAILURE,
    ErrorSignal.RUNTIME_ERROR,
    ErrorSignal.SYNTAX_ERROR,
}


def _extract_file_refs(content: str) -> set[str]:
    """Extract .py file paths referenced in error text.

    Normalizes path separators to forward slashes for comparison.
    """
    refs: set[str] = set()
    for pattern in _FILE_REF_PATTERNS:
        for match in pattern.findall(content):
            # Normalize separators so "a\b.py" and "a/b.py" compare equal
            refs.add(match.replace("\\", "/"))
    return refs


def detect_a2_mislocalization(
    events: list[ToolCallEvent],
    signals: list[ErrorSignal],
) -> bool:
    """A2: Mislocalization — agent edits wrong file(s) vs. error references.

    Plan.md §3.1: "Agent edits file F, but error/test failure references
    file G ≠ F." We interpret this as: if any error event references a
    .py file, and NO file_edit targets any of the referenced files, the
    agent mislocalized the fix.

    Logic:
    1. Collect referenced files from TEST_FAILURE / RUNTIME_ERROR /
       SYNTAX_ERROR events (parsed from result_content).
    2. Collect files targeted by file_edit events.
    3. If referenced set is non-empty AND disjoint from edited set → A2.

    Returns False when:
    - No error events with file refs exist (can't diagnose mislocalization)
    - No file_edit events exist (nothing to mislocalize)
    - At least one edit lands on a referenced file
    """
    referenced: set[str] = set()
    edited: set[str] = set()

    for evt, sig in zip(events, signals):
        if sig in _ERROR_SIGNALS_WITH_FILE_REFS and evt.result_content:
            referenced.update(_extract_file_refs(evt.result_content))
        if evt.tool_name == "file_edit" and evt.target_file:
            edited.add(evt.target_file.replace("\\", "/"))

    if not referenced or not edited:
        return False

    return referenced.isdisjoint(edited)


# ---------------------------------------------------------------------------
# B1.1: Incomplete Fix
# ---------------------------------------------------------------------------

# Unified diff markers: file header and hunk header
_DIFF_FILE_HEADER = re.compile(r'^diff --git a/(\S+) b/\S+', re.MULTILINE)
_DIFF_FILE_HEADER_ALT = re.compile(r'^\+\+\+ b/(\S+)', re.MULTILINE)
_DIFF_HUNK_HEADER = re.compile(r'^@@ ', re.MULTILINE)


def _count_diff_files_and_hunks(diff: str) -> tuple[int, int]:
    """Parse a unified diff, return (file_count, hunk_count).

    Prefers "diff --git" headers; falls back to "+++ b/" if absent.
    Files are deduplicated; hunks are counted as @@ markers.
    """
    if not diff:
        return (0, 0)

    files = set(_DIFF_FILE_HEADER.findall(diff))
    if not files:
        files = set(_DIFF_FILE_HEADER_ALT.findall(diff))
    hunks = len(_DIFF_HUNK_HEADER.findall(diff))
    return (len(files), hunks)


def detect_b11_incomplete_fix(final_diff: str, gold_patch: str) -> bool:
    """B1.1: Incomplete Fix — final diff covers fewer files or hunks than gold.

    Plan.md §3.1: "Final diff touches fewer files/hunks than the gold
    patch (when gold available)."

    Logic:
    - Parse both diffs, counting files and hunks.
    - If gold_patch is empty → False (can't compare, assume ok).
    - If final_diff is empty but gold is non-empty → True (no fix at all).
    - If final's file count < gold's OR final's hunk count < gold's → True.

    Note: this is a coverage proxy, not a semantic correctness check.
    A trajectory with strictly more files/hunks than gold may still be
    wrong, but is not B1.1.
    """
    if not gold_patch.strip():
        return False

    gold_files, gold_hunks = _count_diff_files_and_hunks(gold_patch)
    if gold_files == 0 and gold_hunks == 0:
        # Degenerate gold — nothing to compare against
        return False

    final_files, final_hunks = _count_diff_files_and_hunks(final_diff)

    return final_files < gold_files or final_hunks < gold_hunks


# ---------------------------------------------------------------------------
# B1.2: Insufficient Testing
# ---------------------------------------------------------------------------

# Test-runner keywords searched in bash command strings. Match is
# whole-word-ish via regex boundaries to avoid "pytestfixture" false hits.
_TEST_RUNNER_PATTERNS = re.compile(
    r"\bpytest\b"
    r"|\bpython\s+-m\s+pytest\b"
    r"|\bpython\s+-m\s+unittest\b"
    r"|\bpython\s+-m\s+nose\b"
    r"|\bunittest\b"
    r"|\bmanage\.py\s+test\b"
    r"|\btox\b"
    r"|\bnosetests?\b",
)


def detect_b12_insufficient_testing(
    events: list[ToolCallEvent],
    verification: VerificationResult,
) -> bool:
    """B1.2: Insufficient Testing — tier2 configured but no test-runner call.

    Plan.md §3.1: "Trajectory contains no bash tool call matching test
    runner keywords (pytest/unittest/python -m/manage.py test/tox/nose)
    despite Tier 2 being configured."

    Logic:
    - If tier2_command is empty → False (not applicable).
    - Scan all bash events' command params for test-runner keywords.
    - If NONE match → B1.2.

    Note: a trajectory that invoked tests externally (verifier, not agent)
    would still trip this. That's intentional — the category measures
    whether the *agent itself* exercised tests during its work.
    """
    if not verification.tier2_command:
        return False

    for evt in events:
        if evt.tool_name != "bash":
            continue
        cmd = evt.params.get("command", "") if evt.params else ""
        if cmd and _TEST_RUNNER_PATTERNS.search(cmd):
            return False

    return True


# ---------------------------------------------------------------------------
# Combined classifier
# ---------------------------------------------------------------------------


def classify_liu(
    events: list[ToolCallEvent],
    signals: list[ErrorSignal],
    llm_claimed_success: bool,
    verification: VerificationResult,
    *,
    final_diff: str = "",
    gold_patch: str = "",
) -> LiuClassification:
    """Run all 8 Liu et al. category detectors (core 4 + extended 4).

    Priority rule (§3.1.1): when both B2.1 and C2 apply, C2 takes
    priority — B2.1 is suppressed because the more severe diagnosis
    (agent is deluded about success) subsumes the logic error.

    Args:
        events: ToolCallEvent sequence from the trajectory.
        signals: Parallel error signal list (from error_signal_extractor).
        llm_claimed_success: Whether the agent claimed success.
        verification: Verification results (tier1, tier2).
        final_diff: Final trajectory diff (optional, for B1.1).
        gold_patch: Gold reference patch (optional, for B1.1).
            When either is empty, B1.1 returns False.

    Returns:
        LiuClassification with all 8 category booleans.
    """
    c2 = detect_c2_premature_success(llm_claimed_success, verification)
    c1 = detect_c1_false_negative(llm_claimed_success, verification)
    b22 = detect_b22_syntax_error(events, signals)

    # B2.1 detection — suppressed if C2 is active (priority rule)
    if c2:
        b21 = False
    else:
        b21 = detect_b21_logic_error(events, signals, verification)

    # Extended 4
    a1 = detect_a1_missing_context(events)
    a2 = detect_a2_mislocalization(events, signals)
    b11 = detect_b11_incomplete_fix(final_diff, gold_patch)
    b12 = detect_b12_insufficient_testing(events, verification)

    return LiuClassification(
        b21_logic_error=b21,
        b22_syntax_error=b22,
        c1_false_negative=c1,
        c2_premature_success=c2,
        a1_missing_context=a1,
        a2_mislocalization=a2,
        b11_incomplete_fix=b11,
        b12_insufficient_testing=b12,
    )


def classify_liu_from_record(
    record: TrajectoryRecord,
    signals: list[ErrorSignal],
    *,
    final_diff: str = "",
    gold_patch: str = "",
) -> LiuClassification:
    """Convenience: classify a TrajectoryRecord directly.

    Requires signals to be pre-computed via ``classify_trajectory_signals()``.
    Optional ``final_diff`` / ``gold_patch`` enable B1.1 detection.
    """
    return classify_liu(
        events=list(record.tool_call_events),
        signals=signals,
        llm_claimed_success=record.llm_claimed_success,
        verification=record.verification,
        final_diff=final_diff,
        gold_patch=gold_patch,
    )


# ---------------------------------------------------------------------------
# Trajectory-level outcome label (§3.1.2)
# ---------------------------------------------------------------------------


def assign_outcome_label(
    liu: LiuClassification,
    verification: VerificationResult,
    contains_recovery: bool,
) -> OutcomeLabel:
    """Assign trajectory-level outcome label per §3.1.2.

    Label priority (first match wins):
    1. toxic       — C2 or B2.2 in final state
    2. failed      — Tier 2 configured and fails, no recovery to pass
    3. weakly_verified — Tier 1 passes, Tier 2 not configured
    4. verified_one_shot — Tier 1 + Tier 2 pass, no recovery
    5. verified_with_recovery — Tier 1 + Tier 2 pass, has recovery
    """
    # 1. Toxic
    if liu.is_toxic:
        return OutcomeLabel.TOXIC

    # 2. Failed — Tier 2 was configured and failed
    if verification.tier2_command and not verification.tier2_pass:
        return OutcomeLabel.FAILED

    # 3. Weakly verified — Tier 1 passes, Tier 2 not configured
    if not verification.tier2_command:
        if verification.tier1_pass:
            return OutcomeLabel.WEAKLY_VERIFIED
        return OutcomeLabel.FAILED

    # 4/5. Tier 1 + Tier 2 both pass
    if verification.tier1_pass and verification.tier2_pass:
        if contains_recovery:
            return OutcomeLabel.VERIFIED_WITH_RECOVERY
        return OutcomeLabel.VERIFIED_ONE_SHOT

    # Edge: tier2 passes but tier1 doesn't (shouldn't happen normally)
    return OutcomeLabel.FAILED
