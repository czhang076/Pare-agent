"""Structured JSONL event log for observability.

Every significant event in the agent lifecycle is recorded as a JSON line.
This serves three purposes:
1. Debugging — understand exactly what the agent did and why.
2. Data source — /history and /cost CLI commands read from this log.
3. Learning — reviewing logs helps understand agent behavior patterns.

Events are append-only and flushed immediately so no data is lost on crash.
The log file grows unbounded within a session; rotation is the caller's
responsibility (e.g., one file per session).

Event types:
    llm_request   — LLM call started (model, message count, tool count)
    llm_response  — LLM call completed (token usage, stop reason, duration)
    llm_error     — LLM call failed (error type, retryable flag)
    tool_call     — Tool execution started (tool name, params summary)
    tool_result   — Tool execution completed (success, output summary, duration)
    guardrail     — Guardrail triggered (type, message)
    compaction    — Context compaction (stage, tokens before/after)
    agent_event   — General agent lifecycle event (phase, step, status)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Event:
    """A single telemetry event."""

    type: str
    data: dict[str, Any]
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


class EventLog:
    """Append-only JSONL event logger.

    Usage:
        log = EventLog(Path("session.jsonl"))
        log.log("llm_request", model="claude-sonnet-4-20250514", messages=12)
        log.log("tool_call", tool="file_read", params={"file_path": "main.py"})

        # Query cost
        total = log.total_tokens()

        # Close when done
        log.close()
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._token_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}

    def log(self, event_type: str, **data: Any) -> None:
        """Record an event. Flushes immediately."""
        event = Event(type=event_type, data=data, timestamp=time.time())
        self._file.write(event.to_json() + "\n")
        self._file.flush()

        # Accumulate token usage from llm_response events
        if event_type == "llm_response":
            usage = data.get("usage", {})
            self._token_usage["input"] += usage.get("input_tokens", 0)
            self._token_usage["output"] += usage.get("output_tokens", 0)
            self._token_usage["cache_read"] += usage.get("cache_read_tokens", 0)
            self._token_usage["cache_create"] += usage.get("cache_create_tokens", 0)

    def total_tokens(self) -> dict[str, int]:
        """Get accumulated token usage across all LLM calls."""
        return dict(self._token_usage)

    def total_cost_estimate(self, input_price_per_mtok: float, output_price_per_mtok: float) -> float:
        """Rough cost estimate in USD based on token counts and per-million-token pricing."""
        input_cost = (self._token_usage["input"] / 1_000_000) * input_price_per_mtok
        output_cost = (self._token_usage["output"] / 1_000_000) * output_price_per_mtok
        return input_cost + output_cost

    def read_events(self, event_type: str | None = None) -> list[Event]:
        """Read events from the log file, optionally filtered by type.

        Note: this re-reads the file from disk. For large logs, consider
        streaming with read_events_iter().
        """
        events: list[Event] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    event = Event(
                        type=raw["type"],
                        data=raw["data"],
                        timestamp=raw["timestamp"],
                    )
                    if event_type is None or event.type == event_type:
                        events.append(event)
        except FileNotFoundError:
            pass
        return events

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
