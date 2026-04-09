"""Orient phase — zero-LLM-call repository scanning.

Scans the repository to build a RepoContext object that gives the LLM
a "map" of the codebase.  This runs before any LLM call and costs zero
tokens to produce.

Four scans:
1. Directory tree (depth-limited, ignoring noise directories)
2. Code signatures (regex-based: def/class/function/func)
3. Key files (README, pyproject.toml, etc. — first 50 lines)
4. Git status (branch, recent commits, uncommitted changes)

The output is written into the Memory Index (Layer 1) so it's always
available to the LLM.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories to always skip during scanning
_IGNORE_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".env", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".eggs", "*.egg-info",
    ".next", ".nuxt", "target", "vendor",
})

# Key files to read (first 50 lines) for project context
_KEY_FILES = [
    "README.md", "README.rst", "README.txt",
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "Cargo.toml", "go.mod",
    "Makefile", "Dockerfile",
    "CONTRIBUTING.md",
]

# Regex patterns for code signatures (language-agnostic MVP)
_SIGNATURE_PATTERNS = [
    # Python
    re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE),
    re.compile(r"^class\s+(\w+)[\s:(]", re.MULTILINE),
    # JavaScript / TypeScript
    re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE),
    re.compile(r"^(?:export\s+)?class\s+(\w+)\s", re.MULTILINE),
    # Go
    re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE),
    # Rust
    re.compile(r"^(?:pub\s+)?fn\s+(\w+)\s*[<(]", re.MULTILINE),
    re.compile(r"^(?:pub\s+)?struct\s+(\w+)", re.MULTILINE),
]

# File extensions to scan for signatures
_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift",
})


@dataclass
class FileSignature:
    """Signatures (functions, classes) found in a single file."""
    path: str
    signatures: list[str] = field(default_factory=list)


@dataclass
class GitInfo:
    """Git repository status."""
    branch: str = ""
    recent_commits: list[str] = field(default_factory=list)
    uncommitted_changes: list[str] = field(default_factory=list)
    is_git_repo: bool = False


@dataclass
class RepoContext:
    """Complete repository context from the Orient phase."""
    tree: str = ""                          # Directory tree string
    signatures: list[FileSignature] = field(default_factory=list)
    key_file_previews: dict[str, str] = field(default_factory=dict)
    git: GitInfo = field(default_factory=GitInfo)
    total_files: int = 0
    total_dirs: int = 0
    total_lines: int = 0

    def to_markdown(self) -> str:
        """Render as Markdown for injection into the Memory Index."""
        parts: list[str] = []

        # Structure
        if self.tree:
            parts.append(f"## Structure\n```\n{self.tree}\n```")
            parts.append(
                f"Total: {self.total_dirs} directories, "
                f"{self.total_files} files, ~{self.total_lines:,} lines"
            )

        # Key signatures
        if self.signatures:
            parts.append("\n## Key Signatures")
            for fs in self.signatures[:20]:  # Cap to avoid bloat
                sigs = ", ".join(fs.signatures[:10])
                parts.append(f"- `{fs.path}`: {sigs}")

        # Git info
        if self.git.is_git_repo:
            parts.append(f"\n## Git")
            parts.append(f"Branch: `{self.git.branch}`")
            if self.git.uncommitted_changes:
                changes = ", ".join(self.git.uncommitted_changes[:10])
                parts.append(f"Uncommitted: {changes}")
            if self.git.recent_commits:
                parts.append("Recent commits:")
                for c in self.git.recent_commits[:5]:
                    parts.append(f"  - {c}")

        return "\n".join(parts)


class RepoScanner:
    """Scans a repository to produce a RepoContext.

    Usage:
        scanner = RepoScanner(cwd=Path("."))
        context = await scanner.scan()
        markdown = context.to_markdown()
    """

    def __init__(
        self,
        cwd: Path,
        max_depth: int = 3,
        max_files_for_signatures: int = 100,
    ) -> None:
        self.cwd = cwd.resolve()
        self.max_depth = max_depth
        self.max_files_for_signatures = max_files_for_signatures

    async def scan(self) -> RepoContext:
        """Run all four scans and return the combined RepoContext."""
        ctx = RepoContext()

        # Run scans concurrently where possible
        tree_task = asyncio.to_thread(self._scan_tree)
        sigs_task = asyncio.to_thread(self._scan_signatures)
        keys_task = asyncio.to_thread(self._scan_key_files)
        git_task = self._scan_git()

        tree_result, sigs_result, keys_result, git_result = await asyncio.gather(
            tree_task, sigs_task, keys_task, git_task,
            return_exceptions=True,
        )

        if isinstance(tree_result, tuple):
            ctx.tree, ctx.total_files, ctx.total_dirs, ctx.total_lines = tree_result
        elif isinstance(tree_result, BaseException):
            logger.warning("Tree scan failed: %s", tree_result)

        if isinstance(sigs_result, list):
            ctx.signatures = sigs_result
        elif isinstance(sigs_result, BaseException):
            logger.warning("Signature scan failed: %s", sigs_result)

        if isinstance(keys_result, dict):
            ctx.key_file_previews = keys_result
        elif isinstance(keys_result, BaseException):
            logger.warning("Key file scan failed: %s", keys_result)

        if isinstance(git_result, GitInfo):
            ctx.git = git_result
        elif isinstance(git_result, BaseException):
            logger.warning("Git scan failed: %s", git_result)

        return ctx

    # ------------------------------------------------------------------
    # Scan 1: Directory tree
    # ------------------------------------------------------------------

    def _scan_tree(self) -> tuple[str, int, int, int]:
        """Build a depth-limited directory tree string."""
        lines: list[str] = []
        total_files = 0
        total_dirs = 0
        total_lines = 0

        def _walk(path: Path, prefix: str, depth: int) -> None:
            nonlocal total_files, total_dirs, total_lines

            if depth > self.max_depth:
                return

            try:
                entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return

            dirs = []
            files = []
            for entry in entries:
                if entry.name.startswith(".") and entry.name != ".pare":
                    continue
                if entry.is_dir():
                    if entry.name in _IGNORE_DIRS:
                        continue
                    dirs.append(entry)
                elif entry.is_file():
                    files.append(entry)

            items = dirs + files
            for i, entry in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")

                if entry.is_dir():
                    total_dirs += 1
                    # Count children for annotation
                    try:
                        child_count = sum(
                            1 for c in entry.iterdir()
                            if not c.name.startswith(".")
                            and c.name not in _IGNORE_DIRS
                        )
                    except PermissionError:
                        child_count = 0
                    lines.append(f"{prefix}{connector}{entry.name}/ ({child_count} items)")
                    _walk(entry, child_prefix, depth + 1)
                else:
                    total_files += 1
                    try:
                        size = entry.stat().st_size
                        # Estimate lines for code files
                        if entry.suffix in _CODE_EXTENSIONS and size < 500_000:
                            try:
                                line_count = entry.read_text(
                                    encoding="utf-8", errors="ignore"
                                ).count("\n") + 1
                                total_lines += line_count
                            except Exception:
                                pass
                    except OSError:
                        size = 0
                    lines.append(f"{prefix}{connector}{entry.name}")

        lines.append(f"{self.cwd.name}/")
        _walk(self.cwd, "", 0)

        return "\n".join(lines), total_files, total_dirs, total_lines

    # ------------------------------------------------------------------
    # Scan 2: Code signatures
    # ------------------------------------------------------------------

    def _scan_signatures(self) -> list[FileSignature]:
        """Extract function/class signatures from code files."""
        results: list[FileSignature] = []
        files_scanned = 0

        for path in self._iter_code_files():
            if files_scanned >= self.max_files_for_signatures:
                break
            files_scanned += 1

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, PermissionError):
                continue

            # Don't scan huge files
            if len(content) > 200_000:
                continue

            sigs: list[str] = []
            for pattern in _SIGNATURE_PATTERNS:
                for match in pattern.finditer(content):
                    name = match.group(1)
                    # Skip private/dunder unless it's __init__
                    if name.startswith("_") and name != "__init__":
                        continue
                    if name not in sigs:
                        sigs.append(name)

            if sigs:
                rel_path = str(path.relative_to(self.cwd)).replace("\\", "/")
                results.append(FileSignature(path=rel_path, signatures=sigs))

        return results

    def _iter_code_files(self):
        """Yield code files in the repo, skipping ignored directories."""
        for path in self.cwd.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in _CODE_EXTENSIONS:
                continue
            # Check if any parent is in ignore set
            parts = path.relative_to(self.cwd).parts
            if any(part in _IGNORE_DIRS for part in parts):
                continue
            if any(part.startswith(".") and part != ".pare" for part in parts):
                continue
            yield path

    # ------------------------------------------------------------------
    # Scan 3: Key files
    # ------------------------------------------------------------------

    def _scan_key_files(self) -> dict[str, str]:
        """Read first 50 lines of key project files."""
        previews: dict[str, str] = {}

        for name in _KEY_FILES:
            path = self.cwd / name
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()[:50]
                    previews[name] = "\n".join(lines)
                except (OSError, PermissionError):
                    continue

        return previews

    # ------------------------------------------------------------------
    # Scan 4: Git status
    # ------------------------------------------------------------------

    async def _scan_git(self) -> GitInfo:
        """Get git branch, recent commits, and uncommitted changes."""
        info = GitInfo()

        # Check if git repo
        if not (self.cwd / ".git").exists():
            return info
        info.is_git_repo = True

        # Current branch
        branch = await self._git("rev-parse", "--abbrev-ref", "HEAD")
        if branch:
            info.branch = branch.strip()

        # Recent commits (one-line format)
        log = await self._git("log", "--oneline", "-5")
        if log:
            info.recent_commits = [
                line.strip() for line in log.strip().splitlines() if line.strip()
            ]

        # Uncommitted changes
        status = await self._git("status", "--porcelain")
        if status:
            info.uncommitted_changes = [
                line[3:].strip() for line in status.strip().splitlines()
                if line.strip()
            ]

        return info

    async def _git(self, *args: str) -> str:
        """Run a git command. Returns stdout or empty string on failure."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.cwd),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="replace")
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            pass
        return ""
