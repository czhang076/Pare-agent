"""Hard verification — zero-LLM-call checks after tool execution.

Tier 1 (always on, zero config):
- Syntax check: compile() Python files after edit/create
- Diff check: warn if agent claims done but git shows no changes

These checks append warnings to tool results or inject system notes,
so the LLM can self-correct. They never block execution outright.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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
