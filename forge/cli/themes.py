"""CLI color and style constants.

Centralized theme so the entire CLI has a consistent look. Uses Rich
style strings — these are passed to Rich's Console and Text objects.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Semantic color roles
# ---------------------------------------------------------------------------

# Agent phases
PHASE = "bold cyan"
STEP = "bold blue"

# Tool calls
TOOL_NAME = "bold yellow"
TOOL_BLOCKED = "bold red"
TOOL_RESULT_OK = "green"
TOOL_RESULT_ERR = "red"

# LLM output
LLM_TEXT = ""  # default terminal color
LLM_THINKING = "dim italic"

# Status
SUCCESS = "bold green"
FAILURE = "bold red"
WARNING = "bold yellow"
INFO = "dim"

# UI elements
PROMPT = "bold green"
HEADER = "bold"
BORDER = "dim"
COST = "cyan"

# Diff
DIFF_ADD = "green"
DIFF_REMOVE = "red"
DIFF_HEADER = "bold blue"
