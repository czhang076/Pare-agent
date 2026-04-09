"""Guardrails — safety checks that run during agent execution.

These are code-level enforcement mechanisms, not prompt-level suggestions.
They protect against:
- Budget overruns (too many tool calls)
- Infinite loops (same tool+params repeated)
- Consecutive errors (agent stuck on failures)
- Hallucinated edits (editing a file without reading it first)

Each guard returns either None (OK) or an error message string. The
executor checks guards before each tool call and injects the error
as a tool_result if blocked, giving the LLM a chance to self-correct.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GuardrailConfig:
    """Configurable limits for guardrails."""

    max_tool_calls: int = 100          # Total tool calls per task
    max_tool_calls_per_step: int = 15  # Tool calls within one plan step
    max_consecutive_errors: int = 3    # Errors in a row before forcing stop
    max_repeated_actions: int = 2      # Same tool+params before warning
    max_file_changes_per_step: int = 10  # Unique files edited per step


@dataclass(slots=True)
class GuardrailState:
    """Mutable state tracked across tool calls within a session."""

    total_tool_calls: int = 0
    step_tool_calls: int = 0
    consecutive_errors: int = 0
    action_hashes: dict[str, int] = field(default_factory=dict)  # hash -> count
    read_files: set[str] = field(default_factory=set)  # files read in current step
    edited_files: set[str] = field(default_factory=set)  # files edited in current step

    def reset_step(self) -> None:
        """Reset per-step counters (called at the start of each plan step).

        NOTE: read_files is intentionally preserved across steps/turns
        so that the read-before-write guardrail remembers files read in
        earlier turns.  Only edited_files and loop detection are reset.
        """
        self.step_tool_calls = 0
        self.action_hashes.clear()
        self.edited_files.clear()

    def reset_all(self) -> None:
        """Full reset including read_files (e.g. new task from scratch)."""
        self.total_tool_calls = 0
        self.step_tool_calls = 0
        self.consecutive_errors = 0
        self.action_hashes.clear()
        self.read_files.clear()
        self.edited_files.clear()


def _hash_action(tool_name: str, params: dict) -> str:
    """Deterministic hash of a tool call for loop detection."""
    raw = json.dumps({"name": tool_name, **params}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:12]  # noqa: S324


class Guardrails:
    """Checks that run before each tool call.

    Usage:
        guard = Guardrails()
        # Before each tool call:
        msg = guard.check(tool_name, params)
        if msg:
            # Don't execute — feed msg back to LLM as tool_result
        else:
            result = await tool.execute(params, ctx)
            guard.record_result(tool_name, params, result.success)
    """

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self.config = config or GuardrailConfig()
        self.state = GuardrailState()

    def check(self, tool_name: str, params: dict) -> str | None:
        """Run all guards. Returns error message if blocked, None if OK."""
        # 1. Total budget
        if self.state.total_tool_calls >= self.config.max_tool_calls:
            return (
                f"Total tool call budget exhausted ({self.config.max_tool_calls}). "
                "You must stop and report your progress to the user."
            )

        # 2. Step budget
        if self.state.step_tool_calls >= self.config.max_tool_calls_per_step:
            return (
                f"Step tool call budget exhausted ({self.config.max_tool_calls_per_step}). "
                "Declare this step completed or failed, and move on."
            )

        # 3. Consecutive errors
        if self.state.consecutive_errors >= self.config.max_consecutive_errors:
            return (
                f"You have failed {self.config.max_consecutive_errors} times in a row. "
                "Try a completely different approach or declare this step failed."
            )

        # 4. Loop detection
        action_hash = _hash_action(tool_name, params)
        current_count = self.state.action_hashes.get(action_hash, 0)
        if current_count >= self.config.max_repeated_actions:
            return (
                "You are repeating the exact same action. This is likely a loop. "
                "Try a different approach or declare this step failed."
            )

        # 5. Read-before-write enforcement
        if tool_name in ("file_edit",):
            file_path = params.get("file_path", "")
            if file_path and file_path not in self.state.read_files:
                return (
                    f"Cannot edit '{file_path}' — you must read it first. "
                    "Call file_read on this file before attempting to edit it."
                )

        # 6. File change limit per step
        if tool_name in ("file_edit", "file_create"):
            file_path = params.get("file_path", "")
            if (
                file_path
                and file_path not in self.state.edited_files
                and len(self.state.edited_files) >= self.config.max_file_changes_per_step
            ):
                return (
                    f"Too many files changed in this step "
                    f"({self.config.max_file_changes_per_step}). "
                    "Finish this step and continue in the next one."
                )

        return None

    def record_call(self, tool_name: str, params: dict) -> None:
        """Record that a tool call is about to execute."""
        self.state.total_tool_calls += 1
        self.state.step_tool_calls += 1

        action_hash = _hash_action(tool_name, params)
        self.state.action_hashes[action_hash] = (
            self.state.action_hashes.get(action_hash, 0) + 1
        )

    def record_result(self, tool_name: str, params: dict, success: bool) -> None:
        """Record the outcome of a tool call."""
        if success:
            self.state.consecutive_errors = 0
        else:
            self.state.consecutive_errors += 1

        # Track file reads and edits
        if tool_name == "file_read" and success:
            file_path = params.get("file_path", "")
            if file_path:
                self.state.read_files.add(file_path)

        if tool_name in ("file_edit", "file_create") and success:
            file_path = params.get("file_path", "")
            if file_path:
                self.state.edited_files.add(file_path)

    def reset_step(self) -> None:
        """Reset per-step state (call at the start of each plan step)."""
        self.state.reset_step()

    @property
    def budget_remaining(self) -> int:
        return self.config.max_tool_calls - self.state.total_tool_calls

    @property
    def step_budget_remaining(self) -> int:
        return self.config.max_tool_calls_per_step - self.state.step_tool_calls
