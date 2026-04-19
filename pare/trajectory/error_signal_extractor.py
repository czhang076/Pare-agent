# Vendored from research branch claude/great-carson-333acd @ 8571fb1f.
# Do not edit here — make changes on the research side and re-vendor.
"""Rule-based error signal extraction from tool call results.

Classifies each ``ToolCallEvent.result_content`` into one of the
``ErrorSignal`` enum values using regex patterns and heuristics.

Classification priority (most specific first):
    1. BLOCKED   — already set at recording time, skip
    2. NONE      — result_success == True and no error patterns
    3. TIMEOUT   — timeout patterns in output
    4. SYNTAX_ERROR — Python SyntaxError / IndentationError
    5. TEST_FAILURE — pytest/unittest with failure indicators
    6. COMMAND_NOT_FOUND — missing command / path
    7. RUNTIME_ERROR — Python traceback / generic exceptions
    8. EMPTY_DIFF — git diff with no changes
    9. OTHER     — result_success == False, no pattern matched

All classification is deterministic. No LLM calls.
"""

from __future__ import annotations

import re

from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent

# ---------------------------------------------------------------------------
# Test runner keywords — used to distinguish TEST_FAILURE from RUNTIME_ERROR
# ---------------------------------------------------------------------------

_TEST_RUNNER_KEYWORDS = re.compile(
    r"pytest|unittest|python\s+-m\s+pytest|python\s+-m\s+unittest"
    r"|manage\.py\s+test|tox|nose|py\.test",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern sets per signal type
# ---------------------------------------------------------------------------

# TIMEOUT
_TIMEOUT_PATTERNS = re.compile(
    r"timed?\s*out|timeout|TimeoutError|TimeoutExpired"
    r"|execution\s+exceeded\s+time\s+limit"
    r"|killed.*timeout",
    re.IGNORECASE,
)

# SYNTAX_ERROR — Python compile-time errors
_SYNTAX_ERROR_PATTERNS = re.compile(
    r"SyntaxError:|IndentationError:|TabError:"
    r"|compile\(\).*?failed"
    r"|⚠ SYNTAX ERROR:",
)

# TEST_FAILURE — pytest / unittest failure indicators
_TEST_FAILURE_PATTERNS = re.compile(
    r"FAILED\s|FAIL:|FAILURES"
    r"|AssertionError|AssertionError"  # common typo in some outputs
    r"|errors?=\d+"
    r"|failures?=\d+"
    r"|ERROR:\s+found\s+no\s+collectors"
    r"|=+\s+ERRORS\s+=+"
    r"|=+\s+FAILURES\s+=+"
    r"|=+\s+short test summary"
    r"|pytest.*Exit\s+code:\s+[1-5]",
    re.IGNORECASE,
)

# COMMAND_NOT_FOUND
_COMMAND_NOT_FOUND_PATTERNS = re.compile(
    r"command\s+not\s+found"
    r"|No\s+such\s+file\s+or\s+directory"
    r"|not\s+recognized\s+as\s+(an?\s+)?(internal|external)(\s+or\s+(internal|external))?\s+command"
    r"|Exit\s+code:\s+127"
    # Windows Chinese locale variants — may appear as proper Unicode or
    # as GBK mojibake (ϵͳ, Ҳ, ָ etc.) depending on encoding chain
    r"|不是内部或外部命令"
    r"|系统找不到指定的"
    r"|找不到指定的路径"
    # GBK mojibake of "系统找不到指定的路径" and "不是内部或外部命令"
    r"|ϵͳ.{0,4}Ҳ.{0,6}ָ"
    r"|ⲿ.{0,4}Ҳ.{0,4}ǿ",
    re.IGNORECASE,
)

# RUNTIME_ERROR — Python tracebacks and generic exceptions
_RUNTIME_ERROR_PATTERNS = re.compile(
    r"Traceback\s+\(most\s+recent\s+call\s+last\)"
    r"|^\w*Error:|^\w*Exception:"
    r"|raise\s+\w+Error"
    r"|STDERR:\s*\n.*Error",
    re.MULTILINE,
)

# EMPTY_DIFF — git diff showing no changes
_EMPTY_DIFF_PATTERNS = re.compile(
    r"git\s+diff.*shows?\s+no\s+changes"
    r"|no\s+changes\s+detected"
    r"|nothing\s+to\s+commit",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


def _is_test_command(tool_name: str, params: dict) -> bool:
    """Check if this tool call is running a test command."""
    if tool_name != "bash":
        return False
    command = params.get("command", "")
    return bool(_TEST_RUNNER_KEYWORDS.search(command))


def extract_error_signal(event: ToolCallEvent) -> ErrorSignal:
    """Classify a single ToolCallEvent's error signal.

    Args:
        event: A ToolCallEvent with result_content populated.

    Returns:
        The most specific ErrorSignal matching the event.
    """
    # Already classified at recording time
    if event.error_signal == ErrorSignal.BLOCKED:
        return ErrorSignal.BLOCKED

    content = event.result_content

    # Success path — check for embedded warnings that indicate errors
    # despite success=True (e.g. syntax check appended by executor)
    if event.result_success:
        if _SYNTAX_ERROR_PATTERNS.search(content):
            return ErrorSignal.SYNTAX_ERROR
        return ErrorSignal.NONE

    # From here: result_success == False

    # TIMEOUT — check first, short-circuits everything else
    if _TIMEOUT_PATTERNS.search(content):
        return ErrorSignal.TIMEOUT

    # SYNTAX_ERROR — Python compile errors
    if _SYNTAX_ERROR_PATTERNS.search(content):
        return ErrorSignal.SYNTAX_ERROR

    # TEST_FAILURE — must be a test runner command with failure indicators
    if _is_test_command(event.tool_name, event.params):
        if _TEST_FAILURE_PATTERNS.search(content):
            return ErrorSignal.TEST_FAILURE

    # COMMAND_NOT_FOUND — missing binary, bad path
    if _COMMAND_NOT_FOUND_PATTERNS.search(content):
        return ErrorSignal.COMMAND_NOT_FOUND

    # RUNTIME_ERROR — Python tracebacks, exceptions
    if _RUNTIME_ERROR_PATTERNS.search(content):
        return ErrorSignal.RUNTIME_ERROR

    # EMPTY_DIFF
    if _EMPTY_DIFF_PATTERNS.search(content):
        return ErrorSignal.EMPTY_DIFF

    # Fallback
    return ErrorSignal.OTHER


def classify_trajectory_signals(
    events: list[ToolCallEvent],
) -> list[ErrorSignal]:
    """Classify error signals for all events in a trajectory.

    Returns a list of ErrorSignal values parallel to the input events.
    """
    return [extract_error_signal(evt) for evt in events]
