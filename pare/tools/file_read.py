"""FileReadTool — read file contents with optional line range.

Returns content with line numbers prepended (like `cat -n`), which helps
the LLM reference specific lines when making edits. Handles encoding
detection and file size limits.
"""

from __future__ import annotations

import logging
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

_MAX_FILE_SIZE = 100 * 1024  # 100KB
_MAX_LINES_DEFAULT = 200


class FileReadTool(Tool):
    name = "file_read"
    description = (
        "Read the contents of a file. Returns the content with line numbers. "
        "Use the 'start_line' and 'end_line' parameters to read specific "
        "sections of large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read (relative to project root).",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed). Default: 1.",
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "Last line to read (inclusive). Default: start_line + 200. "
                    "Use this to limit output for large files."
                ),
            },
        },
        "required": ["file_path"],
    }
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        file_path_str = params.get("file_path", "")
        if not file_path_str:
            return ToolResult(success=False, output="", error="file_path is required")

        forbidden = validate_workspace_path(file_path_str)
        if forbidden:
            return ToolResult(success=False, output="", error=forbidden)

        if context.exec_target == "container":
            return await self._execute_in_container(file_path_str, params, context)

        # Resolve relative to working directory
        file_path = (context.cwd / file_path_str).resolve()

        # Security: ensure the path is within the working directory
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
                error=f"File not found: {file_path_str}",
            )

        if not file_path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a file: {file_path_str} (is it a directory?)",
            )

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > _MAX_FILE_SIZE:
            return await self._read_large_file(file_path, file_path_str, file_size, params)

        # Read with encoding detection
        content = self._read_file(file_path)
        if content is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read {file_path_str}: unable to decode file content.",
            )

        lines = content.splitlines()
        total_lines = len(lines)

        # Apply line range
        start = max(1, params.get("start_line", 1))
        end = params.get("end_line", start + _MAX_LINES_DEFAULT - 1)
        end = min(end, total_lines)

        selected = lines[start - 1 : end]

        # Format with line numbers
        width = len(str(end))
        numbered = [f"{i:{width}d}\t{line}" for i, line in enumerate(selected, start=start)]
        output = "\n".join(numbered)

        # Add file info header
        if start > 1 or end < total_lines:
            header = f"[{file_path_str}] lines {start}-{end} of {total_lines}"
        else:
            header = f"[{file_path_str}] {total_lines} lines"

        return ToolResult(
            success=True,
            output=f"{header}\n{output}",
            metadata={"total_lines": total_lines, "start": start, "end": end},
        )

    async def _read_large_file(
        self, file_path: Path, file_path_str: str, file_size: int, params: dict
    ) -> ToolResult:
        """Handle files larger than _MAX_FILE_SIZE."""
        # If specific lines requested, try to read just those
        if "start_line" in params:
            content = self._read_file(file_path)
            if content is None:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to decode large file: {file_path_str}",
                )
            lines = content.splitlines()
            start = max(1, params.get("start_line", 1))
            end = min(params.get("end_line", start + _MAX_LINES_DEFAULT - 1), len(lines))
            selected = lines[start - 1 : end]
            width = len(str(end))
            numbered = [f"{i:{width}d}\t{line}" for i, line in enumerate(selected, start=start)]
            header = f"[{file_path_str}] lines {start}-{end} of {len(lines)} (file: {file_size // 1024}KB)"
            return ToolResult(
                success=True,
                output=f"{header}\n" + "\n".join(numbered),
                metadata={"total_lines": len(lines), "start": start, "end": end},
            )

        # No specific lines: show head + tail
        content = self._read_file(file_path)
        if content is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to decode large file: {file_path_str}",
            )
        lines = content.splitlines()
        head = lines[:50]
        tail = lines[-20:]
        head_str = "\n".join(f"{i:4d}\t{line}" for i, line in enumerate(head, 1))
        tail_start = len(lines) - 19
        tail_str = "\n".join(
            f"{i:4d}\t{line}" for i, line in enumerate(tail, start=tail_start)
        )
        skipped = len(lines) - 70
        output = (
            f"[{file_path_str}] {len(lines)} lines ({file_size // 1024}KB) — "
            f"showing first 50 + last 20 lines\n"
            f"{head_str}\n"
            f"\n... [{skipped} lines omitted] ...\n\n"
            f"{tail_str}"
        )
        return ToolResult(
            success=True,
            output=output,
            metadata={"total_lines": len(lines), "truncated": True},
        )

    @staticmethod
    def _read_file(path: Path) -> str | None:
        """Read file with encoding fallback: UTF-8 → Latin-1."""
        for encoding in ("utf-8", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, ValueError):
                continue
        return None

    async def _execute_in_container(
        self, file_path_str: str, params: dict, context: ToolContext
    ) -> ToolResult:
        """Container-mode read — delegates to InstanceContainer.read_file.

        Path handling mirrors the host path: relative paths resolve against
        ``context.cwd`` which, for container mode, is the container's
        working directory (``/testbed`` by default). Absolute paths outside
        ``/testbed`` are rejected for symmetry with the host branch's
        relative_to() check — we do not want the agent writing to ``/etc``.
        """
        if context.container is None:
            return ToolResult(
                success=False,
                output="",
                error="file_read container mode requires ToolContext.container",
            )

        # Build absolute container path.
        if file_path_str.startswith("/"):
            abs_path = file_path_str
        else:
            cwd = str(context.cwd).replace("\\", "/").rstrip("/")
            abs_path = f"{cwd}/{file_path_str}"

        workdir = str(context.cwd).replace("\\", "/").rstrip("/")
        if not abs_path.startswith(workdir + "/") and abs_path != workdir:
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {file_path_str} is outside {workdir}.",
            )

        try:
            content = await context.container.read_file(
                abs_path, max_bytes=_MAX_FILE_SIZE * 10
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read {file_path_str}: {e}",
            )

        lines = content.splitlines()
        total_lines = len(lines)
        start = max(1, params.get("start_line", 1))
        end = params.get("end_line", start + _MAX_LINES_DEFAULT - 1)
        end = min(end, total_lines)
        selected = lines[start - 1 : end]
        width = len(str(end)) if end else 1
        numbered = [
            f"{i:{width}d}\t{line}"
            for i, line in enumerate(selected, start=start)
        ]
        output = "\n".join(numbered)
        if start > 1 or end < total_lines:
            header = f"[{file_path_str}] lines {start}-{end} of {total_lines}"
        else:
            header = f"[{file_path_str}] {total_lines} lines"
        return ToolResult(
            success=True,
            output=f"{header}\n{output}" if output else header,
            metadata={"total_lines": total_lines, "start": start, "end": end},
        )
