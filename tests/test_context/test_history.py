"""Tests for SessionHistory — JSONL session history with search."""

from pathlib import Path

import pytest

from pare.context.history import HistoryEntry, SessionHistory
from pare.llm.base import ContentBlock, ContentBlockType, Message, ToolCallRequest


@pytest.fixture
def history(tmp_path: Path) -> SessionHistory:
    return SessionHistory(tmp_path / "history.jsonl")


class TestHistoryEntry:
    def test_round_trip(self):
        entry = HistoryEntry(role="user", text="hello", turn=1, tokens_estimate=2)
        json_str = entry.to_json()
        restored = HistoryEntry.from_json(json_str)
        assert restored.role == "user"
        assert restored.text == "hello"
        assert restored.turn == 1


class TestSessionHistory:
    def test_append_string_message(self, history: SessionHistory):
        msg = Message(role="user", content="Fix the bug")
        history.append(msg, turn=1)
        history.close()

        entries = history.recent(10)
        assert len(entries) == 1
        assert entries[0].text == "Fix the bug"
        assert entries[0].role == "user"

    def test_append_blocks_message(self, history: SessionHistory):
        msg = Message(role="assistant", content=[
            ContentBlock(type=ContentBlockType.TEXT, text="Let me read the file."),
            ContentBlock(
                type=ContentBlockType.TOOL_USE,
                tool_call=ToolCallRequest(id="1", name="file_read", arguments={"file_path": "main.py"}),
            ),
        ])
        history.append(msg, turn=2)
        history.close()

        entries = history.recent(10)
        assert len(entries) == 2
        assert "Let me read" in entries[0].text
        assert "file_read" in entries[1].text
        assert entries[1].tool_name == "file_read"

    def test_append_tool_result(self, history: SessionHistory):
        msg = Message(role="tool_result", content=[
            ContentBlock(
                type=ContentBlockType.TOOL_RESULT,
                tool_call_id="1",
                text="line 1: hello\nline 2: world",
            ),
        ])
        history.append(msg, turn=3)
        history.close()

        entries = history.recent(10)
        assert len(entries) == 1
        assert "hello" in entries[0].text

    def test_search_finds_matches(self, history: SessionHistory):
        history.append(Message(role="user", content="Fix the login bug"), turn=1)
        history.append(Message(role="assistant", content="I'll read auth.py"), turn=1)
        history.append(Message(role="user", content="Now add tests"), turn=2)
        history.close()

        results = history.search("login")
        assert len(results) == 1
        assert "login" in results[0].text

    def test_search_multiple_words(self, history: SessionHistory):
        history.append(Message(role="user", content="Fix the login bug in auth module"), turn=1)
        history.append(Message(role="user", content="Fix the login issue"), turn=2)
        history.close()

        results = history.search("login auth")
        assert len(results) == 1
        assert "auth" in results[0].text

    def test_search_empty_query(self, history: SessionHistory):
        history.append(Message(role="user", content="hello"), turn=1)
        history.close()
        assert history.search("") == []

    def test_recent(self, history: SessionHistory):
        for i in range(20):
            history.append(Message(role="user", content=f"message {i}"), turn=i)
        history.close()

        recent = history.recent(5)
        assert len(recent) == 5
        assert "message 19" in recent[-1].text

    def test_total_entries(self, history: SessionHistory):
        for i in range(5):
            history.append(Message(role="user", content=f"msg {i}"), turn=i)
        history.close()
        assert history.total_entries() == 5

    def test_turn_counter(self, history: SessionHistory):
        assert history.next_turn() == 1
        assert history.next_turn() == 2
        assert history.next_turn() == 3

    def test_truncates_long_tool_results(self, history: SessionHistory):
        long_text = "x" * 5000
        msg = Message(role="tool_result", content=[
            ContentBlock(type=ContentBlockType.TOOL_RESULT, tool_call_id="1", text=long_text),
        ])
        history.append(msg, turn=1)
        history.close()

        entries = history.recent(1)
        assert len(entries[0].text) < 3000  # Should be truncated
