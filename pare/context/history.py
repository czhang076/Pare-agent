"""Session history — Layer 3 of the 3-layer memory architecture.

Raw conversation history stored as append-only JSONL.  Never bulk-loaded
into the LLM context — instead, it is searchable via keyword grep so the
agent (or compactor) can retrieve relevant past snippets on demand.

Each entry is one message (user, assistant, tool_result) serialized with
enough metadata to reconstruct the conversation if needed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pare.llm.base import ContentBlock, ContentBlockType, Message

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HistoryEntry:
    """A single serializable history record."""

    role: str
    text: str  # Flattened text representation
    tool_name: str = ""  # For tool_result entries
    turn: int = 0  # Monotonic turn counter
    tokens_estimate: int = 0  # Rough token count for this entry

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> HistoryEntry:
        data = json.loads(raw)
        return cls(**data)


class SessionHistory:
    """Append-only JSONL history with keyword search.

    Usage:
        history = SessionHistory(Path(".pare/history.jsonl"))
        history.append(message, turn=3)
        results = history.search("login function")
        history.close()
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._turn = 0

    @property
    def path(self) -> Path:
        return self._path

    def next_turn(self) -> int:
        """Advance and return the turn counter."""
        self._turn += 1
        return self._turn

    def append(self, message: Message, *, turn: int | None = None) -> None:
        """Serialize and append a message to the history file."""
        if turn is None:
            turn = self._turn

        entries = self._message_to_entries(message, turn)
        for entry in entries:
            self._file.write(entry.to_json() + "\n")
        self._file.flush()

    def append_many(self, messages: list[Message], *, turn: int | None = None) -> None:
        """Append multiple messages in one batch."""
        if turn is None:
            turn = self._turn
        for msg in messages:
            self.append(msg, turn=turn)

    def search(self, query: str, *, max_results: int = 20) -> list[HistoryEntry]:
        """Search history for entries matching the query keywords.

        Simple keyword search: all words in the query must appear in the
        entry text (case-insensitive).  Returns most recent matches first.
        """
        words = query.lower().split()
        if not words:
            return []

        matches: list[HistoryEntry] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = HistoryEntry.from_json(line)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    text_lower = entry.text.lower()
                    if all(w in text_lower for w in words):
                        matches.append(entry)
        except FileNotFoundError:
            return []

        # Most recent first, limited
        return matches[-max_results:][::-1]

    def recent(self, n: int = 10) -> list[HistoryEntry]:
        """Return the N most recent history entries."""
        entries: list[HistoryEntry] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(HistoryEntry.from_json(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except FileNotFoundError:
            return []
        return entries[-n:]

    def total_entries(self) -> int:
        """Count total entries in the history file."""
        count = 0
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except FileNotFoundError:
            pass
        return count

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _message_to_entries(message: Message, turn: int) -> list[HistoryEntry]:
        """Convert a Message to one or more HistoryEntry records."""
        entries: list[HistoryEntry] = []

        if isinstance(message.content, str):
            entries.append(HistoryEntry(
                role=message.role,
                text=message.content,
                turn=turn,
                tokens_estimate=max(1, len(message.content) // 4),
            ))
        elif isinstance(message.content, list):
            for block in message.content:
                if block.type == ContentBlockType.TEXT and block.text:
                    entries.append(HistoryEntry(
                        role=message.role,
                        text=block.text,
                        turn=turn,
                        tokens_estimate=max(1, len(block.text) // 4),
                    ))
                elif block.type == ContentBlockType.TOOL_USE and block.tool_call:
                    tc = block.tool_call
                    text = f"[tool_call] {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})"
                    entries.append(HistoryEntry(
                        role=message.role,
                        text=text,
                        tool_name=tc.name,
                        turn=turn,
                        tokens_estimate=max(1, len(text) // 4),
                    ))
                elif block.type == ContentBlockType.TOOL_RESULT:
                    text = block.text or ""
                    # Truncate very long tool results in history
                    if len(text) > 2000:
                        text = text[:2000] + "\n... [truncated in history]"
                    entries.append(HistoryEntry(
                        role="tool_result",
                        text=text,
                        tool_name=block.tool_call_id or "",
                        turn=turn,
                        tokens_estimate=max(1, len(text) // 4),
                    ))
        return entries
