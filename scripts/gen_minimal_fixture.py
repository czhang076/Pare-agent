"""Regenerate tests/inspector/fixtures/minimal.jsonl.

Two hand-built TrajectoryRecord rows exercised by the W1 Day 1 tests:
    traj-001 — clean success, hits VERIFIED_ONE_SHOT, no Liu tags
    traj-002 — file_edit that leaves a lingering SyntaxError, trips B2.2
               (-> TOXIC per §3.1.2)

Run this once and commit the resulting .jsonl. Tests load the committed
file so there is no runtime dependency on this script.
"""

from __future__ import annotations

from pathlib import Path

from pare.trajectory.schema import TrajectoryRecord
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent

FIXTURE = Path(__file__).resolve().parents[1] / "tests/inspector/fixtures/minimal.jsonl"


def _record_a() -> TrajectoryRecord:
    events = [
        ToolCallEvent.create(
            turn_id=0, call_index_in_turn=0, global_index=0,
            tool_name="file_read",
            params={"file_path": "calc.py"},
            result_success=True,
            result_content="def add(x, y):\n    return x+1\n",
            timestamp=0.5,
        ),
        ToolCallEvent.create(
            turn_id=1, call_index_in_turn=0, global_index=1,
            tool_name="file_edit",
            params={"file_path": "calc.py", "old": "x+1", "new": "x + 1"},
            result_success=True, result_content="1 edit applied.",
            timestamp=1.0,
        ),
        ToolCallEvent.create(
            turn_id=2, call_index_in_turn=0, global_index=2,
            tool_name="bash",
            params={"command": "pytest tests/test_calc.py -q"},
            result_success=True, result_content="2 passed in 0.01s",
            timestamp=2.0,
        ),
    ]
    return TrajectoryRecord.from_dict({
        "schema_version": "2.0",
        "trajectory_id": "traj-001",
        "instance_id": "demo__calc-ws",
        "task": "tighten whitespace in calc.py",
        "model": "deepseek/deepseek-chat",
        "seed": 0,
        "created_at": 1.0,
        "llm_claimed_success": True,
        "verification": {
            "final_passed": True,
            "tier1_pass": True,
            "tier2_pass": True,
            "tier2_command": "pytest tests/",
        },
        "tool_call_events": [e.to_dict() for e in events],
    })


def _record_b() -> TrajectoryRecord:
    events = [
        ToolCallEvent.create(
            turn_id=0, call_index_in_turn=0, global_index=0,
            tool_name="file_read",
            params={"file_path": "calc.py"},
            result_success=True,
            result_content="def add(x, y):\n    return x\n",
            timestamp=0.5,
        ),
        ToolCallEvent.create(
            turn_id=1, call_index_in_turn=0, global_index=1,
            tool_name="file_edit",
            params={"file_path": "calc.py", "old": "return x", "new": "return x +"},
            result_success=False,
            result_content=(
                "  File \"calc.py\", line 3\n"
                "    return x +\n"
                "              ^\n"
                "SyntaxError: invalid syntax"
            ),
            error_signal=ErrorSignal.SYNTAX_ERROR,
            timestamp=1.0,
        ),
        ToolCallEvent.create(
            turn_id=2, call_index_in_turn=0, global_index=2,
            tool_name="bash",
            params={"command": "pytest tests/test_calc.py -q"},
            result_success=False,
            result_content=(
                "  File \"calc.py\", line 3\n"
                "    return x +\n"
                "              ^\n"
                "SyntaxError: invalid syntax"
            ),
            timestamp=2.0,
        ),
    ]
    return TrajectoryRecord.from_dict({
        "schema_version": "2.0",
        "trajectory_id": "traj-002",
        "instance_id": "demo__calc-broken",
        "task": "extend calc.py — agent leaves a syntax error",
        "model": "anthropic/claude-3.5-sonnet",
        "seed": 0,
        "created_at": 2.0,
        "llm_claimed_success": True,
        "verification": {
            "final_passed": False,
            "tier1_pass": True,
            "tier2_pass": False,
            "tier2_command": "pytest tests/",
        },
        "tool_call_events": [e.to_dict() for e in events],
    })


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w", encoding="utf-8") as fh:
        fh.write(_record_a().to_json_line() + "\n")
        fh.write(_record_b().to_json_line() + "\n")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
