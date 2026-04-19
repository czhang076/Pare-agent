"""R0 placeholder tests for InstanceContainer.

Real tests land in R1 once the class is implemented. Marked ``skip`` so
``pytest -q`` stays green while the signatures raise NotImplementedError.

The 9 test cases listed in the refactor plan (§验证)::

    1. test_start_stop_idempotent
    2. test_exec_captures_stdout_stderr
    3. test_exec_nonzero_exit
    4. test_exec_timeout
    5. test_read_write_roundtrip
    6. test_write_creates_parents
    7. test_git_diff_empty_when_no_change
    8. test_git_diff_after_write
    9. test_context_manager_cleanup_on_exception

All require a reachable Docker daemon + a pulled swebench image.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="R0 scaffold — InstanceContainer not implemented yet (R1)")


def test_start_stop_idempotent() -> None:
    raise NotImplementedError("R1")


def test_exec_captures_stdout_stderr() -> None:
    raise NotImplementedError("R1")


def test_exec_nonzero_exit() -> None:
    raise NotImplementedError("R1")


def test_exec_timeout() -> None:
    raise NotImplementedError("R1")


def test_read_write_roundtrip() -> None:
    raise NotImplementedError("R1")


def test_write_creates_parents() -> None:
    raise NotImplementedError("R1")


def test_git_diff_empty_when_no_change() -> None:
    raise NotImplementedError("R1")


def test_git_diff_after_write() -> None:
    raise NotImplementedError("R1")


def test_context_manager_cleanup_on_exception() -> None:
    raise NotImplementedError("R1")
