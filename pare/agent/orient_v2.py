"""Optional zero-LLM repo-orientation pre-pass with Aider-style repo map.

``run_agent`` calls :func:`run_orient` once before the main ReAct loop
(when ``LoopConfig.use_orient=True``) and injects the output into the
system prompt as a "Repository Context" section via
:func:`format_orient_for_system_prompt`.

The output has three sections, each with a token budget:

1. **README head** — first ``readme_max_chars`` characters of the
   top-level README, so the agent sees project-level intent (what this
   codebase is) before it starts tool-calling.
2. **Top-level listing** — non-recursive directory listing so the
   agent doesn't waste a tool call on ``ls /testbed``.
3. **Repo map (optional)** — Aider-style ranked list of Python files
   with their top-level ``class`` / ``def`` signatures. The rank is a
   lightweight PageRank substitute: each file gets a score from
   (sqrt(line_count) × 1/(1+depth) × non_test_bonus). For the top-K
   files we parse with stdlib :mod:`ast` — no tree-sitter dependency —
   and emit headers only (no bodies).

Phase 3.13.2 lives in section 3 (the repo map). Section 1+2 is the
"naive orient" that the plan originally specced — available by passing
``use_repo_map=False``.

Fails open: any internal exception returns an empty string. The caller
should format with :func:`format_orient_for_system_prompt` to get a
no-op section when the blob is empty.
"""

from __future__ import annotations

import ast
import logging
import posixpath
import shlex
from dataclasses import dataclass

from pare.sandbox.instance_container import InstanceContainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — tuned for a ~4k-token system-prompt budget
# ---------------------------------------------------------------------------

DEFAULT_README_MAX_CHARS = 1500
DEFAULT_TOP_LIST_MAX = 40
DEFAULT_REPO_MAP_TOP_K = 12
DEFAULT_REPO_MAP_MAX_SIGS_PER_FILE = 15
DEFAULT_MAX_FILES_TO_RANK = 1500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_orient(
    container: InstanceContainer,
    *,
    readme_max_chars: int = DEFAULT_README_MAX_CHARS,
    top_list_max: int = DEFAULT_TOP_LIST_MAX,
    use_repo_map: bool = True,
    repo_map_top_k: int = DEFAULT_REPO_MAP_TOP_K,
    repo_map_max_sigs_per_file: int = DEFAULT_REPO_MAP_MAX_SIGS_PER_FILE,
) -> str:
    """Produce a markdown repo-context blob for the agent's system prompt.

    Never raises — returns empty string on any internal failure. All
    sub-sections that fail individually are silently dropped, not
    propagated.
    """
    try:
        sections: list[str] = []

        readme = await _render_readme(container, readme_max_chars)
        if readme:
            sections.append(readme)

        listing = await _render_top_listing(container, top_list_max)
        if listing:
            sections.append(listing)

        if use_repo_map:
            repo_map = await _render_repo_map(
                container,
                top_k=repo_map_top_k,
                max_sigs=repo_map_max_sigs_per_file,
            )
            if repo_map:
                sections.append(repo_map)
    except Exception as e:
        logger.warning("orient_v2 failed: %s", e)
        return ""

    return "\n\n".join(sections)


def format_orient_for_system_prompt(blob: str) -> str:
    """Wrap ``blob`` in a labelled markdown section, or return ``""``."""
    if not blob.strip():
        return ""
    return (
        "\n\n## Repository Context (orient_v2 pre-pass)\n\n"
        + blob.strip()
        + "\n"
    )


# ---------------------------------------------------------------------------
# Section 1: README head
# ---------------------------------------------------------------------------


_README_CANDIDATES = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "readme.md",
    "readme.rst",
)


async def _render_readme(
    container: InstanceContainer, max_chars: int
) -> str:
    """Return the first ``max_chars`` of the first matching README, or ""."""
    for name in _README_CANDIDATES:
        path = posixpath.join(container.workdir, name)
        try:
            # read_file auto-truncates at max_bytes — give it a healthy
            # budget above ``max_chars`` so we can re-trim precisely.
            text = await container.read_file(path, max_bytes=max_chars * 4)
        except Exception:
            continue
        head = text[:max_chars].strip()
        if not head:
            continue
        suffix = ""
        if len(text) > max_chars:
            suffix = f"\n\n[...truncated at {max_chars} chars]"
        return f"### README ({name})\n\n{head}{suffix}"
    return ""


# ---------------------------------------------------------------------------
# Section 2: top-level listing
# ---------------------------------------------------------------------------


async def _render_top_listing(
    container: InstanceContainer, max_entries: int
) -> str:
    """List the top-level entries of ``container.workdir`` as markdown."""
    r = await container.exec(
        "ls -1 --group-directories-first 2>/dev/null || ls -1",
        timeout=15.0,
    )
    if r.exit_code != 0:
        return ""
    entries = [
        line.strip()
        for line in r.stdout.splitlines()
        if line.strip() and line.strip() not in (".", "..")
    ]
    if not entries:
        return ""
    if len(entries) > max_entries:
        shown = entries[:max_entries]
        tail = f"\n[... {len(entries) - max_entries} more entries]"
    else:
        shown = entries
        tail = ""
    joined = "\n".join(f"- {name}" for name in shown)
    return (
        f"### Top-level listing ({container.workdir})\n\n{joined}{tail}"
    )


# ---------------------------------------------------------------------------
# Section 3: Aider-style repo map
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FileRank:
    """One ranked file — score is higher-is-better."""

    path: str           # posix path relative to container.workdir
    score: float
    line_count: int


async def _render_repo_map(
    container: InstanceContainer,
    *,
    top_k: int,
    max_sigs: int,
) -> str:
    """Rank .py files and emit top-level class/def signatures for the top-K.

    Ranking heuristic is a lightweight substitute for Aider's PageRank
    over symbol reference graphs: files near the repo root with
    substantive line counts and not in test/example/docs folders score
    highest. Good enough for agent orientation; tree-sitter + real
    PageRank is a possible future upgrade noted in plan §3.13.2.
    """
    files = await _list_python_files(container)
    if not files:
        return ""
    ranked = _rank_files(files)[:top_k]
    blocks: list[str] = []
    for fr in ranked:
        sigs = await _extract_signatures(
            container, fr.path, max_sigs
        )
        if not sigs:
            continue
        body = "\n".join(f"  {s}" for s in sigs)
        blocks.append(
            f"- **{fr.path}** ({fr.line_count} lines)\n{body}"
        )
    if not blocks:
        return ""
    return (
        f"### Repo map (top {len(blocks)} by heuristic rank)\n\n"
        + "\n\n".join(blocks)
    )


async def _list_python_files(
    container: InstanceContainer,
    *,
    max_files: int = DEFAULT_MAX_FILES_TO_RANK,
) -> list[tuple[str, int]]:
    """Return ``(rel_posix_path, line_count)`` for tracked .py files.

    Uses ``git ls-files`` to respect ``.gitignore``; falls back to
    ``find`` when git is unavailable. Line counts come from one batched
    ``wc -l`` call — per-file roundtrips would make this section the
    slowest part of the pre-pass.
    """
    r = await container.exec(
        "git ls-files -z '*.py' 2>/dev/null "
        "|| find . -name '*.py' -not -path '*/.*' -print0",
        timeout=30.0,
    )
    if r.exit_code != 0:
        return []
    raw_paths = [
        p.strip().lstrip("./")
        for p in r.stdout.split("\0")
        if p.strip()
    ]
    if not raw_paths:
        return []
    paths = raw_paths[:max_files]

    # Batch wc -l. MAX_ARG on Linux is ≥ 2 MiB so 1500 paths fit easily.
    quoted = " ".join(shlex.quote(p) for p in paths)
    wc = await container.exec(f"wc -l {quoted}", timeout=45.0)
    counts: dict[str, int] = {}
    if wc.exit_code == 0:
        for line in wc.stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.endswith(" total"):
                continue
            parts = stripped.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                counts[parts[1]] = int(parts[0])
            except ValueError:
                continue

    return [(p, counts.get(p, 0)) for p in paths]


def _rank_files(files: list[tuple[str, int]]) -> list[_FileRank]:
    """Compute the heuristic rank over all files; sort desc by score."""
    ranked: list[_FileRank] = []
    for path, lines in files:
        if lines < 2:
            # Skip empty / one-line stub files (e.g. __init__.py with only
            # an import). Two-line files still make it in — the smallest
            # informative file (one import + one symbol) deserves a row.
            continue
        depth = path.count("/")
        depth_factor = 1.0 / (1.0 + depth)
        lowered = path.lower()
        is_auxiliary = any(
            seg in lowered
            for seg in ("/test", "tests/", "/example", "/examples/", "/docs/")
        )
        aux_factor = 0.3 if is_auxiliary else 1.0
        # Square root damps the effect of one giant file dominating the list.
        size_factor = lines ** 0.5
        score = size_factor * depth_factor * aux_factor
        ranked.append(
            _FileRank(path=path, score=score, line_count=lines)
        )
    ranked.sort(key=lambda fr: fr.score, reverse=True)
    return ranked


async def _extract_signatures(
    container: InstanceContainer, rel_path: str, max_sigs: int
) -> list[str]:
    """Parse ``rel_path`` with stdlib ast; emit top-level / class sigs.

    Bodies are never emitted — just ``class Foo:`` and
    ``def bar(args):`` headers. SyntaxError returns an empty list
    (skip this file) rather than failing the whole pre-pass.
    """
    abs_path = posixpath.join(container.workdir, rel_path)
    try:
        source = await container.read_file(abs_path, max_bytes=200_000)
    except Exception:
        return []
    if not source.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Container Python version may parse constructs the host can't,
        # or vice versa. Either way, we skip — no signal beats wrong signal.
        return []

    sigs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs.append(_render_func(node, prefix=""))
        elif isinstance(node, ast.ClassDef):
            sigs.append(f"class {node.name}:")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sigs.append(_render_func(child, prefix="    "))
        if len(sigs) >= max_sigs:
            sigs.append("    ...")
            break
    return sigs


def _render_func(
    node: "ast.FunctionDef | ast.AsyncFunctionDef", *, prefix: str
) -> str:
    """Render one function signature. Only positional arg names — types
    and defaults are omitted because the point is orientation, not
    autocomplete."""
    args = [a.arg for a in node.args.args]
    keyword = (
        "async def"
        if isinstance(node, ast.AsyncFunctionDef)
        else "def"
    )
    return f"{prefix}{keyword} {node.name}({', '.join(args)})"
