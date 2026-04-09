"""Memory index and topic files — Layers 1 & 2 of the 3-layer architecture.

Layer 1: Memory Index (MEMORY.md)
  A small Markdown summary (~500-1000 tokens) that is ALWAYS included in
  every LLM call.  Contains: repo structure, key signatures, current task,
  plan status.  Updated via "strict write discipline" — only after
  successful operations, never on speculation.

Layer 2: Topic Files (topics/*.md)
  Detailed per-module knowledge files, loaded on-demand when a plan step
  targets a specific area.  Written by the agent during Orient phase or
  after deep exploration.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum size for the memory index (in characters).
# At ~3.5 chars/token, this keeps it under ~1000 tokens.
_MAX_INDEX_CHARS = 3500


class MemoryIndex:
    """Layer 1 — always-in-context memory index.

    Reads/writes a Markdown file (default: .pare/MEMORY.md) that is
    injected into every LLM call as part of the system prompt.

    Usage:
        mem = MemoryIndex(cwd=Path("."))
        content = mem.load()               # Read current index
        mem.update_section("Task", "Fix login bug")
        mem.save()                         # Write back
    """

    def __init__(self, cwd: Path, filename: str = "MEMORY.md") -> None:
        self._path = cwd / ".pare" / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sections: dict[str, str] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> str:
        """Load the memory index from disk. Returns the raw content."""
        if self._path.exists():
            content = self._path.read_text(encoding="utf-8")
            self._sections = self._parse_sections(content)
            self._loaded = True
            return content
        self._loaded = True
        return ""

    def save(self) -> None:
        """Write the current sections back to disk."""
        content = self._render_sections()
        # Enforce max size
        if len(content) > _MAX_INDEX_CHARS:
            content = content[:_MAX_INDEX_CHARS] + "\n... [truncated]\n"
            logger.warning("Memory index exceeds max size, truncated.")
        self._path.write_text(content, encoding="utf-8")

    def get_content(self) -> str:
        """Get the current memory index content for injection into prompts.

        Loads from disk if not already loaded.
        """
        if not self._loaded:
            self.load()
        return self._render_sections()

    def update_section(self, section: str, content: str) -> None:
        """Update a named section in the index.

        Sections are rendered as `## {section}\\n{content}`.
        """
        self._sections[section] = content.strip()

    def remove_section(self, section: str) -> None:
        """Remove a section from the index."""
        self._sections.pop(section, None)

    def get_section(self, section: str) -> str:
        """Get content of a specific section."""
        return self._sections.get(section, "")

    def clear(self) -> None:
        """Clear all sections."""
        self._sections.clear()

    @property
    def sections(self) -> dict[str, str]:
        return dict(self._sections)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sections(content: str) -> dict[str, str]:
        """Parse Markdown H2 sections from the index file."""
        sections: dict[str, str] = {}
        current_key = ""
        current_lines: list[str] = []

        for line in content.splitlines():
            if line.startswith("## "):
                # Save previous section
                if current_key:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = line[3:].strip()
                current_lines = []
            elif line.startswith("# ") and not current_key:
                # Top-level header — use as a special "__title__" section
                sections["__title__"] = line.strip()
            else:
                current_lines.append(line)

        # Save last section
        if current_key:
            sections[current_key] = "\n".join(current_lines).strip()

        return sections

    def _render_sections(self) -> str:
        """Render sections back to Markdown."""
        parts: list[str] = []

        # Title first if present
        title = self._sections.get("__title__", "")
        if title:
            parts.append(title)
            parts.append("")

        for key, value in self._sections.items():
            if key == "__title__":
                continue
            parts.append(f"## {key}")
            if value:
                parts.append(value)
            parts.append("")

        return "\n".join(parts).strip() + "\n"


class TopicStore:
    """Layer 2 — on-demand topic files.

    Topic files live in `.pare/topics/` and contain detailed knowledge
    about specific modules or areas of the codebase.  They are loaded
    only when the orchestrator decides they're relevant to the current
    step.

    Usage:
        topics = TopicStore(cwd=Path("."))
        topics.write("auth_module", "## Auth Module\\nUses JWT tokens...")
        content = topics.read("auth_module")
        all_names = topics.list_topics()
    """

    def __init__(self, cwd: Path) -> None:
        self._dir = cwd / ".pare" / "topics"
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    def write(self, topic: str, content: str) -> Path:
        """Write or overwrite a topic file. Returns the file path."""
        path = self._topic_path(topic)
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote topic file: %s", path.name)
        return path

    def read(self, topic: str) -> str:
        """Read a topic file. Returns empty string if it doesn't exist."""
        path = self._topic_path(topic)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def exists(self, topic: str) -> bool:
        return self._topic_path(topic).exists()

    def delete(self, topic: str) -> bool:
        """Delete a topic file. Returns True if it existed."""
        path = self._topic_path(topic)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_topics(self) -> list[str]:
        """List all topic names (without .md extension)."""
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.md"))

    def _topic_path(self, topic: str) -> Path:
        # Sanitize topic name for filesystem safety
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic)
        return self._dir / f"{safe}.md"
