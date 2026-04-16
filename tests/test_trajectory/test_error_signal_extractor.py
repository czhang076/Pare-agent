"""Tests for error signal extraction from tool call results.

Uses real error samples from pilot trajectory (traj_v2_minimax_flat.jsonl)
plus synthetic samples for error types not yet observed in production.
"""

from __future__ import annotations

import pytest

from pare.trajectory.error_signal_extractor import (
    extract_error_signal,
    classify_trajectory_signals,
)
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


# ---------------------------------------------------------------------------
# Helper — build a minimal ToolCallEvent for testing
# ---------------------------------------------------------------------------


def _evt(
    tool_name: str = "bash",
    params: dict | None = None,
    result_success: bool = False,
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
        target_file="",
        result_success=result_success,
        result_content=result_content,
        error_signal=error_signal,
        timestamp=0.0,
    )


# ======================================================================
# Real samples from pilot trajectory (traj_v2_minimax_flat.jsonl)
# ======================================================================


class TestRealSamples:
    """Tests using actual result_content from the pilot run."""

    def test_event5_heredoc_error_on_windows(self):
        """cd /tmp fails on Windows — no such directory."""
        evt = _evt(
            result_content="ERROR: Exit code: 1\nSTDERR:\n\u00ca\u00b1\u00d3\u00a6\u00b8\u00c3 <<\u00a1\u00a3",
        )
        assert extract_error_signal(evt) == ErrorSignal.OTHER

    def test_event8_python3_not_found_windows(self):
        """python3 is not a recognized command on Windows."""
        evt = _evt(
            result_content=(
                "ERROR: Exit code: 1\nSTDERR:\n"
                "'python3' 不是内部或外部命令\r\n或批处理文件。"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND

    def test_event10_path_not_found_chinese(self):
        """系统找不到指定的路径 — path not found in Chinese locale."""
        evt = _evt(
            result_content="ERROR: Exit code: 1\nSTDERR:\n系统找不到指定的路径。",
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND

    def test_event27_pytest_no_collectors(self):
        """pytest exit code 4 — no tests collected."""
        evt = _evt(
            params={"command": "python -m pytest tests/test_self.py::TestRunTC -v 2>&1"},
            result_content=(
                "ERROR: Exit code: 4\n"
                "============================= test session starts ========\r\n"
                "platform win32 -- Python 3.13.12\r\n"
                "collecting ... ERROR: found no collectors for tests/test_self.py\r\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.TEST_FAILURE

    def test_event37_python_traceback(self):
        """Python traceback with AssertionError from test script."""
        evt = _evt(
            params={"command": "python test_fix.py"},
            result_content=(
                "ERROR: Exit code: 1\n"
                "Pattern: ^src/gen/.*$\r\n"
                "STDERR:\n"
                "Traceback (most recent call last):\r\n"
                '  File "test_fix.py", line 57, in <module>\r\n'
                "    test_ignore_paths_with_trailing_slash()\r\n"
                "  File \"test_fix.py\", line 48, in test_ignore_paths_with_trailing_slash\r\n"
                "    assert result, 'Path should match'\r\n"
                "AssertionError: Path should match\r\n"
            ),
        )
        # Not a test runner command (just "python test_fix.py"), so → RUNTIME_ERROR
        assert extract_error_signal(evt) == ErrorSignal.RUNTIME_ERROR

    def test_event30_shell_syntax_error_chinese(self):
        """Windows shell syntax error — 命令语法不正确."""
        evt = _evt(
            result_content="ERROR: Exit code: 1\nSTDERR:\n命令语法不正确。",
        )
        # No pattern matches this specifically → OTHER
        assert extract_error_signal(evt) == ErrorSignal.OTHER


# ======================================================================
# Synthetic samples for error types not yet in pilot data
# ======================================================================


class TestSyntaxError:
    def test_python_syntax_error(self):
        evt = _evt(
            tool_name="bash",
            result_content=(
                "ERROR: Exit code: 1\n"
                '  File "main.py", line 10\n'
                "    def foo(:\n"
                "           ^\n"
                "SyntaxError: invalid syntax\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.SYNTAX_ERROR

    def test_indentation_error(self):
        evt = _evt(
            tool_name="bash",
            result_content=(
                "ERROR: Exit code: 1\n"
                '  File "utils.py", line 5\n'
                "    return x\n"
                "    ^\n"
                "IndentationError: unexpected indent\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.SYNTAX_ERROR

    def test_tab_error(self):
        evt = _evt(
            result_content=(
                "ERROR: Exit code: 1\n"
                "TabError: inconsistent use of tabs and spaces\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.SYNTAX_ERROR

    def test_executor_syntax_warning_on_success(self):
        """Executor appends syntax warning to successful file_edit result."""
        evt = _evt(
            tool_name="file_edit",
            result_success=True,
            result_content=(
                "File edited successfully.\n\n"
                "⚠ SYNTAX ERROR: line 10: invalid syntax\n"
                "Please fix this before continuing."
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.SYNTAX_ERROR


class TestTestFailure:
    def test_pytest_failed(self):
        evt = _evt(
            params={"command": "python -m pytest tests/ -v"},
            result_content=(
                "===== test session starts =====\n"
                "collected 5 items\n"
                "tests/test_foo.py::test_bar FAILED\n"
                "===== 1 failed, 4 passed =====\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.TEST_FAILURE

    def test_unittest_failure(self):
        evt = _evt(
            params={"command": "python -m unittest tests.test_foo -v"},
            result_content=(
                "test_bar (tests.test_foo.TestFoo) ... FAIL\n"
                "FAIL: test_bar (tests.test_foo.TestFoo)\n"
                "AssertionError: 1 != 2\n"
                "Ran 3 tests\nFAILURES\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.TEST_FAILURE

    def test_pytest_short_summary(self):
        evt = _evt(
            params={"command": "pytest tests/"},
            result_content=(
                "===== short test summary info =====\n"
                "FAILED tests/test_x.py::test_y\n"
                "===== 1 failed =====\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.TEST_FAILURE

    def test_not_test_command_with_assert(self):
        """AssertionError from non-test command → RUNTIME_ERROR, not TEST_FAILURE."""
        evt = _evt(
            params={"command": "python run_check.py"},
            result_content=(
                "Traceback (most recent call last):\n"
                "  File \"run_check.py\", line 5\n"
                "    assert x == 1\n"
                "AssertionError\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.RUNTIME_ERROR

    def test_manage_py_test(self):
        evt = _evt(
            params={"command": "python manage.py test myapp"},
            result_content="FAIL: test_view (myapp.tests.TestView)\nerrors=2\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.TEST_FAILURE

    def test_tox_test(self):
        evt = _evt(
            params={"command": "tox -e py39"},
            result_content="FAILURES\n===== 2 failed =====\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.TEST_FAILURE


class TestCommandNotFound:
    def test_bash_command_not_found(self):
        evt = _evt(
            result_content="ERROR: Exit code: 127\nbash: foo: command not found\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND

    def test_no_such_file_or_directory(self):
        evt = _evt(
            result_content="ERROR: Exit code: 1\nbash: /usr/bin/foo: No such file or directory\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND

    def test_exit_code_127(self):
        evt = _evt(
            result_content="ERROR: Exit code: 127\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND

    def test_windows_not_recognized(self):
        evt = _evt(
            result_content=(
                "ERROR: Exit code: 1\nSTDERR:\n"
                "'git-lfs' is not recognized as an internal or external command\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND

    def test_path_not_found_chinese_partial(self):
        """Mojibake/partial Chinese for 找不到指定的路径."""
        evt = _evt(
            result_content="ERROR: Exit code: 1\nSTDERR:\n找不到指定的路径\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.COMMAND_NOT_FOUND


class TestRuntimeError:
    def test_python_traceback(self):
        evt = _evt(
            result_content=(
                "Traceback (most recent call last):\n"
                '  File "main.py", line 3, in <module>\n'
                "    1/0\n"
                "ZeroDivisionError: division by zero\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.RUNTIME_ERROR

    def test_import_error(self):
        evt = _evt(
            result_content=(
                "Traceback (most recent call last):\n"
                '  File "app.py", line 1, in <module>\n'
                "    import nonexistent\n"
                "ModuleNotFoundError: No module named 'nonexistent'\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.RUNTIME_ERROR

    def test_key_error(self):
        evt = _evt(
            result_content=(
                "Traceback (most recent call last):\n"
                '  File "x.py", line 2, in <module>\n'
                "    d['missing']\n"
                "KeyError: 'missing'\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.RUNTIME_ERROR


class TestTimeout:
    def test_timeout_error(self):
        evt = _evt(
            result_content="ERROR: TimeoutError: execution exceeded time limit (300s)\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.TIMEOUT

    def test_timed_out(self):
        evt = _evt(
            result_content="Command timed out after 120 seconds\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.TIMEOUT

    def test_killed_timeout(self):
        evt = _evt(
            result_content="Process killed due to timeout\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.TIMEOUT


class TestEmptyDiff:
    def test_git_diff_no_changes(self):
        evt = _evt(
            result_content="git diff shows no changes\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.EMPTY_DIFF

    def test_nothing_to_commit(self):
        evt = _evt(
            result_content="On branch main\nnothing to commit, working tree clean\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.EMPTY_DIFF


class TestBlocked:
    def test_already_blocked(self):
        """Events already tagged BLOCKED at recording time are preserved."""
        evt = _evt(
            result_success=False,
            result_content="[BLOCKED] Step tool call budget exhausted (8).",
            error_signal=ErrorSignal.BLOCKED,
        )
        assert extract_error_signal(evt) == ErrorSignal.BLOCKED


class TestNone:
    def test_successful_file_read(self):
        evt = _evt(
            tool_name="file_read",
            result_success=True,
            result_content="[main.py] lines 1-50 of 100\ndef hello(): ...\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.NONE

    def test_successful_bash(self):
        evt = _evt(
            tool_name="bash",
            result_success=True,
            result_content="/home/user/project\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.NONE

    def test_successful_search(self):
        evt = _evt(
            tool_name="search",
            result_success=True,
            result_content="3 matches:\nfoo.py:10: def bar()\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.NONE


class TestOther:
    def test_unknown_error(self):
        evt = _evt(
            result_content="ERROR: Exit code: 1\nSome unknown error output\n",
        )
        assert extract_error_signal(evt) == ErrorSignal.OTHER

    def test_empty_error(self):
        evt = _evt(
            result_content="",
        )
        assert extract_error_signal(evt) == ErrorSignal.OTHER


# ======================================================================
# classify_trajectory_signals — batch interface
# ======================================================================


class TestClassifyTrajectorySignals:
    def test_batch_classification(self):
        events = [
            _evt(result_success=True, result_content="ok"),
            _evt(result_content="ERROR: Exit code: 127\ncommand not found\n"),
            _evt(
                params={"command": "pytest tests/"},
                result_content="FAILED tests/x.py\n",
            ),
        ]
        signals = classify_trajectory_signals(events)
        assert signals == [
            ErrorSignal.NONE,
            ErrorSignal.COMMAND_NOT_FOUND,
            ErrorSignal.TEST_FAILURE,
        ]

    def test_empty_list(self):
        assert classify_trajectory_signals([]) == []


# ======================================================================
# Priority / edge cases
# ======================================================================


class TestPriority:
    def test_syntax_error_beats_runtime(self):
        """SyntaxError in traceback → SYNTAX_ERROR, not RUNTIME_ERROR."""
        evt = _evt(
            result_content=(
                "Traceback (most recent call last):\n"
                '  File "x.py", line 1\n'
                "    def (\n"
                "        ^\n"
                "SyntaxError: invalid syntax\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.SYNTAX_ERROR

    def test_timeout_beats_everything(self):
        """Timeout signal has highest priority among failure signals."""
        evt = _evt(
            result_content=(
                "Traceback (most recent call last):\n"
                "TimeoutError: timed out\n"
            ),
        )
        assert extract_error_signal(evt) == ErrorSignal.TIMEOUT

    def test_test_failure_requires_test_command(self):
        """FAILED in output but not a test command → not TEST_FAILURE."""
        evt = _evt(
            params={"command": "python build.py"},
            result_content="Build step FAILED\nExit code: 1\n",
        )
        # "python build.py" is not a test runner → falls through to OTHER
        assert extract_error_signal(evt) == ErrorSignal.OTHER
