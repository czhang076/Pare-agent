"""SearchTool — code search via ripgrep (rg) or Python fallback.

Prefers ripgrep for performance, but falls back to a pure-Python
implementation using re + pathlib if ripgrep is not installed. This
ensures the tool works in any environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path

from pare.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolResult,
)

logger = logging.getLogger(__name__)

_MAX_RESULTS = 50
_HAS_RIPGREP: bool | None = None


def _check_ripgrep() -> bool:
    """Check if ripgrep is available on PATH (cached)."""
    global _HAS_RIPGREP
    if _HAS_RIPGREP is None:
        _HAS_RIPGREP = shutil.which("rg") is not None
    return _HAS_RIPGREP


class SearchTool(Tool):
    name = "search"
    description = (
        "Search for a pattern (regex) in files under the project directory. "
        "Returns matching lines with file paths and line numbers. "
        "Optionally filter by file glob pattern (e.g., '*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory or file to search in (relative to project root). "
                    "Default: project root."
                ),
            },
            "file_glob": {
                "type": "string",
                "description": (
                    "Glob pattern to filter files (e.g., '*.py', '*.ts'). "
                    "Default: all files."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum matches to return. Default: {_MAX_RESULTS}.",
            },
        },
        "required": ["pattern"],
    }
    mutation_type = MutationType.READ
    permission_level = PermissionLevel.AUTO

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        pattern = params.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, output="", error="pattern is required")

        search_path = params.get("path", ".")
        file_glob = params.get("file_glob")
        max_results = params.get("max_results", _MAX_RESULTS)

        if context.exec_target == "container":
            return await self._search_in_container(
                pattern, search_path, file_glob, max_results, context
            )

        # Resolve search path
        full_path = (context.cwd / search_path).resolve()
        try:
            full_path.relative_to(context.cwd.resolve())
        except ValueError:
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {search_path} is outside the project directory.",
            )

        if not full_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {search_path}",
            )

        if _check_ripgrep():
            return await self._search_ripgrep(
                pattern, full_path, file_glob, max_results, context
            )
        else:
            return await self._search_python(
                pattern, full_path, file_glob, max_results
            )

    async def _search_ripgrep(
        self,
        pattern: str,
        path: Path,
        file_glob: str | None,
        max_results: int,
        context: ToolContext,
    ) -> ToolResult:
        """Search using ripgrep subprocess."""
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color=never",
            f"--max-count={max_results}",
        ]

        if file_glob:
            cmd.extend(["--glob", file_glob])

        # Skip common non-code directories
        for skip in (".git", "node_modules", "__pycache__", ".venv", "venv"):
            cmd.extend(["--glob", f"!{skip}"])

        cmd.extend([pattern, str(path)])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(context.cwd),
                env={**os.environ, **context.env} if context.env else None,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            return ToolResult(success=False, output="", error="Search timed out after 30s")
        except FileNotFoundError:
            # rg disappeared between check and use — fall back
            return await self._search_python(pattern, path, file_glob, max_results)

        output = stdout.decode("utf-8", errors="replace").strip()

        if process.returncode == 1:
            # rg returns 1 when no matches found
            return ToolResult(success=True, output="No matches found.", metadata={"match_count": 0})

        if process.returncode not in (0, 1):
            err = stderr.decode("utf-8", errors="replace").strip()
            return ToolResult(success=False, output="", error=f"ripgrep error: {err}")

        # Count and truncate matches
        lines = output.splitlines()
        match_count = len(lines)
        if match_count > max_results:
            lines = lines[:max_results]
            output = "\n".join(lines) + f"\n\n[{match_count - max_results} more matches truncated]"
        else:
            output = "\n".join(lines)

        return ToolResult(
            success=True,
            output=f"{min(match_count, max_results)} matches:\n{output}",
            metadata={"match_count": min(match_count, max_results)},
        )

    async def _search_python(
        self,
        pattern: str,
        path: Path,
        file_glob: str | None,
        max_results: int,
    ) -> ToolResult:
        """Fallback pure-Python search (slower but always available)."""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, output="", error=f"Invalid regex: {e}")

        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        results: list[str] = []

        def _walk(p: Path) -> None:
            if len(results) >= max_results:
                return

            if p.is_file():
                _search_file(p)
                return

            try:
                entries = sorted(p.iterdir())
            except PermissionError:
                return

            for entry in entries:
                if len(results) >= max_results:
                    return
                if entry.is_dir():
                    if entry.name in skip_dirs:
                        continue
                    _walk(entry)
                elif entry.is_file():
                    if file_glob and not entry.match(file_glob):
                        continue
                    _search_file(entry)

        def _search_file(fp: Path) -> None:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except (PermissionError, OSError):
                return

            for i, line in enumerate(text.splitlines(), 1):
                if len(results) >= max_results:
                    return
                if regex.search(line):
                    try:
                        rel = fp.relative_to(path)
                    except ValueError:
                        rel = fp
                    results.append(f"{rel}:{i}:{line.rstrip()}")

        # Run in thread to avoid blocking the event loop
        await asyncio.to_thread(_walk, path)

        if not results:
            return ToolResult(success=True, output="No matches found.", metadata={"match_count": 0})

        output = "\n".join(results)
        return ToolResult(
            success=True,
            output=f"{len(results)} matches:\n{output}",
            metadata={"match_count": len(results)},
        )

    async def _search_in_container(
        self,
        pattern: str,
        search_path: str,
        file_glob: str | None,
        max_results: int,
        context: ToolContext,
    ) -> ToolResult:
        """Container-mode search — rg if installed, else grep -rn fallback.

        Relies on the derived ``pare-eval.<iid>:latest`` image shipping
        ripgrep. If ``rg`` is missing (offline image build fell back to the
        base swebench tag), we use ``grep -rn`` instead — slower but
        functionally equivalent. Both paths respect ``max_results`` by
        trimming the output list; skip-dir handling for ``.git`` /
        ``__pycache__`` / ``node_modules`` / virtualenvs matches the host
        branch.
        """
        if context.container is None:
            return ToolResult(
                success=False,
                output="",
                error="search container mode requires ToolContext.container",
            )

        # Build absolute container path; default "." maps to workdir.
        cwd_str = str(context.cwd).replace("\\", "/").rstrip("/")
        if search_path in (".", ""):
            abs_path = cwd_str
        elif search_path.startswith("/"):
            abs_path = search_path
        else:
            abs_path = f"{cwd_str}/{search_path}"

        if not (abs_path == cwd_str or abs_path.startswith(cwd_str + "/")):
            return ToolResult(
                success=False,
                output="",
                error=f"Access denied: {search_path} is outside the project directory.",
            )

        has_rg = await self._container_has_rg(context)
        if has_rg:
            return await self._search_ripgrep_in_container(
                pattern, abs_path, file_glob, max_results, context
            )
        return await self._search_grep_in_container(
            pattern, abs_path, file_glob, max_results, context
        )

    @staticmethod
    async def _container_has_rg(context: ToolContext) -> bool:
        """Probe for ripgrep inside the container; result cached on the context.

        The probe runs once per InstanceContainer lifetime — the second
        call reads the cached boolean and skips the roundtrip. We reuse
        the ``context.env`` dict to stash the probe result by the key
        ``_PARE_HAS_RG`` since ``ToolContext`` has no scratch field and
        the env dict is already per-session.
        """
        cached = context.env.get("_PARE_HAS_RG")
        if cached is not None:
            return cached == "1"
        r = await context.container.exec("command -v rg", timeout=10.0)
        has = r.exit_code == 0 and bool(r.stdout.strip())
        context.env["_PARE_HAS_RG"] = "1" if has else "0"
        return has

    async def _search_ripgrep_in_container(
        self,
        pattern: str,
        abs_path: str,
        file_glob: str | None,
        max_results: int,
        context: ToolContext,
    ) -> ToolResult:
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color=never",
            f"--max-count={max_results}",
        ]
        if file_glob:
            cmd.extend(["--glob", file_glob])
        for skip in (".git", "node_modules", "__pycache__", ".venv", "venv"):
            cmd.extend(["--glob", f"!{skip}"])
        cmd.extend(["--", pattern, abs_path])

        r = await context.container.exec(cmd, timeout=30.0)
        if r.timed_out:
            return ToolResult(success=False, output="", error="Search timed out after 30s")
        if r.exit_code == 1:
            return ToolResult(
                success=True,
                output="No matches found.",
                metadata={"match_count": 0},
            )
        if r.exit_code not in (0, 1):
            return ToolResult(
                success=False,
                output="",
                error=f"ripgrep error: {r.stderr.strip() or 'exit=' + str(r.exit_code)}",
            )
        return self._format_grep_output(r.stdout, max_results)

    async def _search_grep_in_container(
        self,
        pattern: str,
        abs_path: str,
        file_glob: str | None,
        max_results: int,
        context: ToolContext,
    ) -> ToolResult:
        """``grep -rn`` fallback for images without ripgrep.

        Build: ``grep -rnE <pattern> <path> --include=<glob>
        --exclude-dir={.git,node_modules,__pycache__,.venv,venv}``. Pipe
        through ``head -n`` for max_results trimming. grep's exit codes:
        0 = matches found, 1 = none, >=2 = error.
        """
        import shlex

        parts = [
            "grep", "-rnE",
        ]
        for skip in (".git", "node_modules", "__pycache__", ".venv", "venv"):
            parts.append(f"--exclude-dir={skip}")
        if file_glob:
            parts.append(f"--include={file_glob}")
        parts.extend(["--", pattern, abs_path])
        cmd = " ".join(shlex.quote(p) for p in parts) + f" | head -n {int(max_results) + 1}"

        r = await context.container.exec(cmd, timeout=30.0)
        if r.timed_out:
            return ToolResult(success=False, output="", error="Search timed out after 30s")
        # When `head` cuts off grep early, grep sometimes exits non-zero
        # because of SIGPIPE. Treat empty output with non-zero as "no
        # matches" only when stderr is clean.
        if not r.stdout.strip() and r.exit_code not in (0, 1, 141):
            return ToolResult(
                success=False,
                output="",
                error=f"grep error: {r.stderr.strip() or 'exit=' + str(r.exit_code)}",
            )
        if not r.stdout.strip():
            return ToolResult(
                success=True,
                output="No matches found.",
                metadata={"match_count": 0},
            )
        return self._format_grep_output(r.stdout, max_results)

    @staticmethod
    def _format_grep_output(raw: str, max_results: int) -> ToolResult:
        lines = [ln for ln in raw.splitlines() if ln]
        match_count = len(lines)
        if match_count > max_results:
            lines = lines[:max_results]
            output = "\n".join(lines) + (
                f"\n\n[{match_count - max_results} more matches truncated]"
            )
        else:
            output = "\n".join(lines)
        return ToolResult(
            success=True,
            output=f"{min(match_count, max_results)} matches:\n{output}",
            metadata={"match_count": min(match_count, max_results)},
        )
