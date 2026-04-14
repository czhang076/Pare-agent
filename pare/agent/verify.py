"""Hard verification — zero-LLM-call checks after tool execution.

Tier 1 (always on, zero config):
- Syntax check: compile() Python files after edit/create
- Diff check: warn if agent claims done but git shows no changes

Tier 2 (opt-in):
- Run a configurable test command after step completion

These checks append warnings to tool results or inject system notes,
so the LLM can self-correct. They never block execution outright.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Tier2CheckResult:
    """Result of optional Tier-2 verification command execution."""

    enabled: bool
    command: str = ""
    passed: bool = False
    return_code: int | None = None
    output: str = ""
    error: str = ""


def _merge_output(stdout: str | None, stderr: str | None) -> str:
    parts: list[str] = []
    if stdout and stdout.strip():
        parts.append(stdout.strip())
    if stderr and stderr.strip():
        parts.append(stderr.strip())
    return "\n".join(parts)


def syntax_check(file_path: Path) -> str | None:
    """Run compile() on a Python file. Returns error message or None.

    Only checks .py files. Returns None for non-Python files or
    files that pass syntax validation.
    """
    if file_path.suffix != ".py":
        return None

    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    try:
        compile(source, str(file_path), "exec")
        return None
    except SyntaxError as e:
        msg = f"SyntaxError in {file_path.name}"
        if e.lineno:
            msg += f" line {e.lineno}"
        if e.msg:
            msg += f": {e.msg}"
        return msg


def git_diff_check(cwd: Path) -> bool:
    """Check if there are uncommitted changes (staged or unstaged).

    Returns True if there are changes, False if working tree is clean.
    Returns True on git errors (fail-open — don't block on non-git repos).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            # Not a git repo or other error — fail open
            return True
        return bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True


def run_tier2_check(
    cwd: Path,
    test_command: str | None,
    *,
    timeout_seconds: int = 300,
) -> Tier2CheckResult:
    """Run optional Tier-2 test command.

    Returns a disabled result when no command is configured.
    """
    command = (test_command or "").strip()
    if not command:
        return Tier2CheckResult(enabled=False)

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=True,
        )
        return Tier2CheckResult(
            enabled=True,
            command=command,
            passed=completed.returncode == 0,
            return_code=completed.returncode,
            output=_merge_output(completed.stdout, completed.stderr),
        )
    except subprocess.TimeoutExpired as e:
        return Tier2CheckResult(
            enabled=True,
            command=command,
            passed=False,
            error=f"Tier2 command timed out after {timeout_seconds}s",
            output=_merge_output(
                e.stdout if isinstance(e.stdout, str) else None,
                e.stderr if isinstance(e.stderr, str) else None,
            ),
        )
    except Exception as e:  # pragma: no cover - defensive
        return Tier2CheckResult(
            enabled=True,
            command=command,
            passed=False,
            error=f"Tier2 command failed: {e}",
        )
