"""FileEditTool and FileCreateTool — str_replace editing and file creation.

The str_replace approach (used by Claude Code, SWE-agent, Aider) is
deliberately chosen over line-number-based editing because:
- It's unambiguous: old_str must match exactly once
- It's robust to line number drift from prior edits
- The LLM can express intent directly ("replace this with that")

FileCreateTool handles new file creation separately to prevent accidental
overwrites — if a file already exists, use FileEditTool.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

from pare.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolResult,
    validate_workspace_path,
)

logger = logging.getLogger(__name__)


class FileEditTool(Tool):
    name = "file_edit"
    description = (
        "Edit a file by replacing an exact string match. The old_str must "
        "appear exactly once in the file. Returns a diff preview of the change. "
        "You MUST read the file first before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit (relative to project root).",
            },
            "old_str": {
                "type": "string",
                "description": (
                    "The exact string to find and replace. Must match exactly "
                    "once in the file (including whitespace and indentation)."
                ),
            },
            "new_str": {
                "type": "string",
                "description": "The replacement string.",
            },
        },
        "required": ["file_path", "old_str", "new_str"],
    }
    mutation_type = MutationType.WRITE
    permission_level = PermissionLevel.CONFIRM_ONCE

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        file_path_str = params.get("file_path", "")
        old_str = params.get("old_str", "")
        new_str = params.get("new_str", "")

        if not file_path_str:
            return ToolResult(success=False, output="", error="file_path is required")
        if not old_str:
            return ToolResult(success=False, output="", error="old_str is required")
        if old_str == new_str:
            return ToolResult(success=False, output="", error="old_str and new_str are identical")

        forbidden = validate_workspace_path(file_path_str)
        if forbidden:
            return ToolResult(success=False, output="", error=forbidden)

        file_path = (context.cwd / file_path_str).resolve()

        # Security check
        try:
            file_path.relative_to(context.cwd.resolve())
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {file_path_str} is outside the project directory.",
            )

        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path_str}. Use file_create for new files.",
            )

        # Read current content
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as e:
                return ToolResult(success=False, output="", error=f"Cannot read file: {e}")

        # Check uniqueness
        count = content.count(old_str)
        if count == 0:
            # Whitespace-insensitive fallback: normalize whitespace and retry
            match_result = self._whitespace_fallback(content, old_str)
            if match_result is not None:
                matched_str, ws_count = match_result
                if ws_count == 1:
                    logger.warning(
                        "Exact match failed but whitespace-normalized match found in %s. "
                        "Applying edit with warning.",
                        file_path_str,
                    )
                    old_str = matched_str
                    count = 1
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        error=(
                            f"old_str not found exactly, but whitespace-normalized match "
                            f"hits {ws_count} times in {file_path_str}. "
                            "Include more context to make it unique."
                        ),
                    )
            else:
                # No match even with normalization — show similar lines
                hint = self._find_similar(content, old_str)
                error = f"old_str not found in {file_path_str}."
                if hint:
                    error += f" Did you mean:\n{hint}"
                return ToolResult(success=False, output="", error=error)

        if count > 1:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"old_str matches {count} times in {file_path_str}. "
                    "It must match exactly once. Include more surrounding "
                    "context to make the match unique."
                ),
            )

        # Apply the edit
        new_content = content.replace(old_str, new_str, 1)

        # Generate diff for the LLM
        diff = self._generate_diff(content, new_content, file_path_str)

        # Write
        file_path.write_text(new_content, encoding="utf-8")

        return ToolResult(
            success=True,
            output=diff,
            metadata={"file_path": file_path_str},
        )

    @staticmethod
    def _generate_diff(old: str, new: str, filename: str) -> str:
        """Generate a unified diff between old and new content."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=3,
        )
        return "".join(diff)

    @staticmethod
    def _normalize_ws(text: str) -> str:
        """Normalize whitespace: collapse runs of spaces/tabs, strip trailing per line."""
        lines = text.splitlines()
        normalized = []
        for line in lines:
            # Strip trailing whitespace, collapse internal runs of spaces/tabs
            line = line.rstrip()
            line = re.sub(r"[ \t]+", " ", line)
            normalized.append(line)
        return "\n".join(normalized)

    def _whitespace_fallback(self, content: str, old_str: str) -> tuple[str, int] | None:
        """Try whitespace-insensitive matching.

        Returns (matched_original_str, count) if normalized match found, else None.
        """
        norm_old = self._normalize_ws(old_str)
        if not norm_old.strip():
            return None

        # Split content into candidate windows of same line count
        old_lines = old_str.splitlines()
        content_lines = content.splitlines()
        window_size = len(old_lines)

        if window_size == 0 or window_size > len(content_lines):
            return None

        matches: list[str] = []
        for i in range(len(content_lines) - window_size + 1):
            window = "\n".join(content_lines[i : i + window_size])
            if self._normalize_ws(window) == norm_old:
                matches.append(window)

        if not matches:
            return None

        # Deduplicate — if all matches are the same string, count as 1
        unique = set(matches)
        if len(unique) == 1:
            return matches[0], content.count(matches[0])

        return matches[0], len(matches)

    @staticmethod
    def _find_similar(content: str, target: str) -> str:
        """Find lines similar to the target string (for error hints)."""
        target_lines = target.strip().splitlines()
        if not target_lines:
            return ""

        first_line = target_lines[0].strip()
        if not first_line:
            return ""

        content_lines = content.splitlines()
        matches = difflib.get_close_matches(first_line, [l.strip() for l in content_lines], n=3, cutoff=0.6)
        if matches:
            return "\n".join(f"  {m}" for m in matches)
        return ""


class FileCreateTool(Tool):
    name = "file_create"
    description = (
        "Create a new file with the given content. Fails if the file "
        "already exists — use file_edit for modifications. Creates "
        "parent directories automatically."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the new file (relative to project root).",
            },
            "content": {
                "type": "string",
                "description": "The full content of the new file.",
            },
        },
        "required": ["file_path", "content"],
    }
    mutation_type = MutationType.WRITE
    permission_level = PermissionLevel.CONFIRM_ONCE

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        file_path_str = params.get("file_path", "")
        content = params.get("content", "")

        if not file_path_str:
            return ToolResult(success=False, output="", error="file_path is required")

        forbidden = validate_workspace_path(file_path_str)
        if forbidden:
            return ToolResult(success=False, output="", error=forbidden)

        file_path = (context.cwd / file_path_str).resolve()

        # Security check
        try:
            file_path.relative_to(context.cwd.resolve())
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {file_path_str} is outside the project directory.",
            )

        if file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File already exists: {file_path_str}. Use file_edit to modify it.",
            )

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        file_path.write_text(content, encoding="utf-8")

        line_count = len(content.splitlines())
        return ToolResult(
            success=True,
            output=f"Created {file_path_str} ({line_count} lines)",
            metadata={"file_path": file_path_str, "lines": line_count},
        )
