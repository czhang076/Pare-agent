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

import ast
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

        if context.exec_target == "container":
            return await self._execute_in_container(
                file_path_str, old_str, new_str, context
            )

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

        # Phase 3.13.1 post-edit lint: surface syntax errors immediately.
        # The file is NOT rolled back — the agent must issue a follow-up
        # file_edit to fix its own mistake, and Liu B2.2 "Syntax Error
        # After Edit" becomes observable via extract_error_signal. The
        # lint skips when pre-edit content was already invalid Python —
        # attributing broken-before-edit to the agent is wrong.
        syntax_error = _lint_python_host(
            file_path_str, pre_content=content, post_content=new_content
        )
        if syntax_error is not None:
            return ToolResult(
                success=False,
                output=diff,
                # Literal marker ``⚠ SYNTAX ERROR:`` is required by
                # ``extract_error_signal`` to classify this event as
                # Liu B2.2. Keep the colon right after "ERROR" or the
                # regex misses it and it falls to OTHER.
                error=(
                    f"⚠ SYNTAX ERROR: {syntax_error} "
                    "(edit was applied; you must issue a follow-up "
                    "file_edit to fix it)"
                ),
                metadata={
                    "file_path": file_path_str,
                    "syntax_error": syntax_error,
                },
            )

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

    async def _execute_in_container(
        self,
        file_path_str: str,
        old_str: str,
        new_str: str,
        context: ToolContext,
    ) -> ToolResult:
        """Container-mode edit — read via container, match + replace, write back.

        Mirrors the host branch's single-match / whitespace-fallback logic
        but sources content from :meth:`InstanceContainer.read_file` and
        writes via :meth:`write_file`. The diff returned in ``output``
        matches the host shape so downstream ToolCallEvent consumers see
        uniform structure.
        """
        if context.container is None:
            return ToolResult(
                success=False,
                output="",
                error="file_edit container mode requires ToolContext.container",
            )

        abs_path = _abs_container_path(file_path_str, context.cwd)
        if abs_path is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {file_path_str} is outside the project directory.",
            )

        try:
            content = await context.container.read_file(abs_path, max_bytes=10 * 1024 * 1024)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path_str}. Use file_create for new files. ({e})",
            )

        count = content.count(old_str)
        if count == 0:
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

        new_content = content.replace(old_str, new_str, 1)
        diff = self._generate_diff(content, new_content, file_path_str)
        try:
            await context.container.write_file(abs_path, new_content)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write {file_path_str}: {e}",
            )

        # Phase 3.13.1 post-edit lint inside the container. We call the
        # container's own python (via ``py_compile``) rather than
        # re-parsing on the host so the version that will run the tests
        # is the one that validates the syntax — catches f-string /
        # match-statement / walrus deltas across Python versions. The
        # host-side pre-check short-circuits the subprocess when the
        # pre-edit file was already broken.
        syntax_error = await _lint_python_container(
            file_path_str, abs_path, context, pre_content=content
        )
        if syntax_error is not None:
            return ToolResult(
                success=False,
                output=diff,
                # Literal marker ``⚠ SYNTAX ERROR:`` is required by
                # ``extract_error_signal`` to classify this event as
                # Liu B2.2. Keep the colon right after "ERROR" or the
                # regex misses it and it falls to OTHER.
                error=(
                    f"⚠ SYNTAX ERROR: {syntax_error} "
                    "(edit was applied; you must issue a follow-up "
                    "file_edit to fix it)"
                ),
                metadata={
                    "file_path": file_path_str,
                    "syntax_error": syntax_error,
                },
            )

        return ToolResult(
            success=True,
            output=diff,
            metadata={"file_path": file_path_str},
        )


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

        if context.exec_target == "container":
            return await self._execute_in_container(file_path_str, content, context)

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

    async def _execute_in_container(
        self, file_path_str: str, content: str, context: ToolContext
    ) -> ToolResult:
        """Container-mode create — refuses overwrite, writes via put_archive."""
        if context.container is None:
            return ToolResult(
                success=False,
                output="",
                error="file_create container mode requires ToolContext.container",
            )

        abs_path = _abs_container_path(file_path_str, context.cwd)
        if abs_path is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {file_path_str} is outside the project directory.",
            )

        # Refuse overwrite via an explicit `test -e`; cheaper than a full read.
        check = await context.container.exec(
            f"test -e {_q(abs_path)}", timeout=10.0
        )
        if check.exit_code == 0:
            return ToolResult(
                success=False,
                output="",
                error=f"File already exists: {file_path_str}. Use file_edit to modify it.",
            )

        try:
            await context.container.write_file(abs_path, content)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write {file_path_str}: {e}",
            )

        line_count = len(content.splitlines())
        return ToolResult(
            success=True,
            output=f"Created {file_path_str} ({line_count} lines)",
            metadata={"file_path": file_path_str, "lines": line_count},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abs_container_path(file_path_str: str, cwd: Path) -> str | None:
    """Resolve ``file_path_str`` against container cwd; reject escapes.

    Returns absolute posix path inside ``cwd`` or ``None`` if the computed
    path would sit outside ``cwd`` (mirrors the host path's relative_to()
    escape check).
    """
    cwd_str = str(cwd).replace("\\", "/").rstrip("/")
    if file_path_str.startswith("/"):
        abs_path = file_path_str
    else:
        abs_path = f"{cwd_str}/{file_path_str}"
    # Collapse ``..`` lexically — we do not call realpath to avoid a second
    # container roundtrip; lexical check is enough to deny escapes.
    parts: list[str] = []
    for seg in abs_path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(seg)
    normalised = "/" + "/".join(parts)
    if normalised != cwd_str and not normalised.startswith(cwd_str + "/"):
        return None
    return normalised


def _q(path: str) -> str:
    """Shell-quote a posix path for use inside ``bash -lc`` commands."""
    import shlex
    return shlex.quote(path)


# ---------------------------------------------------------------------------
# Phase 3.13.1: post-edit syntax lint
# ---------------------------------------------------------------------------
#
# SWE-agent's ACI surfaces compile errors immediately after every edit so
# the model can self-correct in the same turn. Pare used to lack this
# feedback loop, which meant Liu B2.2 "Syntax Error After Edit" signals
# only showed up indirectly (pytest collection errors, ImportError when
# another tool imported the broken module, etc.) — by the time they
# reached ``extract_error_signal`` the causal link to the edit was lost.
#
# Design choices:
# - Host mode uses ``ast.parse(new_content)``: zero subprocess, fast,
#   deterministic, no dependency on any particular host Python version
#   matching the target — we only need to know "is this file syntactically
#   valid Python in the ast of whichever runtime we're using".
# - Container mode uses ``python -m py_compile <path>`` inside the
#   container: the runtime that will execute the tests is the one that
#   validates the syntax. This matters for version-dependent features
#   (match statements, walrus operator, new-style f-strings).
# - Both return ``None`` on success and a short error string on failure.
#   Non-Python files (by extension) skip the check — the lint is a
#   guardrail for code edits, not a format policy.
# - On lint failure, ``FileEditTool`` returns ``success=False`` but the
#   file stays written. The agent sees the diff it produced + the
#   syntax error and must issue a follow-up edit. ``success=False``
#   routes through ``extract_error_signal`` → ``SYNTAX_ERROR`` (pattern
#   ``⚠ SYNTAX ERROR:``), which is exactly the Liu B2.2 signal we want.


_LINTABLE_SUFFIXES: tuple[str, ...] = (".py",)


def _pre_content_was_valid_python(content: str) -> bool:
    """True iff ``content`` already parses as Python before the edit.

    The lint only measures "edit introduced a syntax error" (Liu B2.2);
    if the pre-edit file was already broken — a stub, a template, a
    test fixture that put plain text in a ``.py`` file — attributing
    the breakage to the agent's edit is wrong, so we skip the lint
    entirely. Uses the host Python's ast which may not catch every
    container-Python-specific syntax, but that's fine: we're not
    asserting the pre-file was valid everywhere, just "valid enough
    that measuring delta is meaningful".
    """
    try:
        ast.parse(content)
    except (SyntaxError, ValueError):
        return False
    return True


def _lint_python_host(
    file_path_str: str, *, pre_content: str, post_content: str
) -> str | None:
    """Host-mode syntax check. Returns error string or ``None``.

    Skips when the file extension isn't Python OR the pre-edit content
    was itself not valid Python (see :func:`_pre_content_was_valid_python`).
    """
    if not _should_lint(file_path_str):
        return None
    if not _pre_content_was_valid_python(pre_content):
        return None
    try:
        ast.parse(post_content)
    except SyntaxError as e:
        location = (
            f" at line {e.lineno}" if e.lineno else ""
        )
        return f"{e.__class__.__name__}: {e.msg}{location}"
    except ValueError as e:
        # Embedded NUL bytes — still worth surfacing.
        return f"ValueError: {e}"
    return None


async def _lint_python_container(
    file_path_str: str,
    abs_path: str,
    context: "ToolContext",
    *,
    pre_content: str,
) -> str | None:
    """Container-mode syntax check via ``python -m py_compile``.

    The check is bounded to 15 s — a clean py_compile on any sane source
    file finishes in well under a second, so timeout here is a symptom
    of container trouble, not a syntax issue. We treat a timeout as
    "don't block the edit" rather than "syntax error" to avoid false
    positives poisoning the error_signal histogram.
    """
    if not _should_lint(file_path_str):
        return None
    if not _pre_content_was_valid_python(pre_content):
        return None
    if context.container is None:
        return None
    try:
        r = await context.container.exec(
            f"python -m py_compile {_q(abs_path)}",
            timeout=15.0,
        )
    except Exception as e:
        logger.warning("post-edit lint container exec failed: %s", e)
        return None
    if r.exit_code == 0:
        return None
    if r.timed_out:
        logger.warning("post-edit lint timed out for %s", file_path_str)
        return None
    err_text = (r.stderr or r.stdout or "").strip()
    if not err_text:
        return f"py_compile exit={r.exit_code}"
    # py_compile prints the full traceback; keep the last two non-empty
    # lines, which carry the file:line + message.
    lines = [ln for ln in err_text.splitlines() if ln.strip()]
    snippet = "\n".join(lines[-2:]) if len(lines) >= 2 else err_text
    return snippet[:500]


def _should_lint(file_path_str: str) -> bool:
    """Return True iff ``file_path_str`` is a Python source file we
    should lint after editing."""
    return file_path_str.endswith(_LINTABLE_SUFFIXES)
