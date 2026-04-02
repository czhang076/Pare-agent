"""Tests for forge/telemetry.py."""

import json
from pathlib import Path

import pytest

from forge.telemetry import EventLog


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "test_session.jsonl"


@pytest.fixture
def event_log(log_path: Path) -> EventLog:
    log = EventLog(log_path)
    yield log
    log.close()


class TestEventLog:
    def test_log_creates_file(self, event_log: EventLog, log_path: Path):
        event_log.log("test_event", key="value")
        assert log_path.exists()

    def test_log_writes_valid_jsonl(self, event_log: EventLog, log_path: Path):
        event_log.log("llm_request", model="claude", messages=5)
        event_log.log("tool_call", tool="bash", params={"command": "ls"})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        event1 = json.loads(lines[0])
        assert event1["type"] == "llm_request"
        assert event1["data"]["model"] == "claude"
        assert "timestamp" in event1

        event2 = json.loads(lines[1])
        assert event2["type"] == "tool_call"
        assert event2["data"]["tool"] == "bash"

    def test_token_accumulation(self, event_log: EventLog):
        event_log.log(
            "llm_response",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
        event_log.log(
            "llm_response",
            usage={"input_tokens": 200, "output_tokens": 75, "cache_read_tokens": 30},
        )

        totals = event_log.total_tokens()
        assert totals["input"] == 300
        assert totals["output"] == 125
        assert totals["cache_read"] == 30

    def test_cost_estimate(self, event_log: EventLog):
        event_log.log(
            "llm_response",
            usage={"input_tokens": 1_000_000, "output_tokens": 500_000},
        )
        # $3/M input, $15/M output (Claude Sonnet pricing example)
        cost = event_log.total_cost_estimate(
            input_price_per_mtok=3.0,
            output_price_per_mtok=15.0,
        )
        assert cost == pytest.approx(3.0 + 7.5)

    def test_read_events_all(self, event_log: EventLog):
        event_log.log("a", x=1)
        event_log.log("b", x=2)
        event_log.log("a", x=3)

        events = event_log.read_events()
        assert len(events) == 3

    def test_read_events_filtered(self, event_log: EventLog):
        event_log.log("llm_request", model="claude")
        event_log.log("tool_call", tool="bash")
        event_log.log("llm_request", model="gpt-4o")

        events = event_log.read_events("llm_request")
        assert len(events) == 2
        assert all(e.type == "llm_request" for e in events)

    def test_context_manager(self, log_path: Path):
        with EventLog(log_path) as log:
            log.log("test", data="value")
        # File should be closed after context manager exits
        assert log_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "session.jsonl"
        log = EventLog(deep_path)
        log.log("test", ok=True)
        log.close()
        assert deep_path.exists()
