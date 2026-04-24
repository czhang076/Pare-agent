"""Tests for ``experiments.export_sft_dataset`` CLI wrapper.

The wrapper itself is thin — the heavy lifting lives in
``pare.trajectory.sft_export`` and is covered by its own test file.
Here we lock argparse defaults, the filter-flag wiring, and that
``main()`` writes the report JSON next to the output.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.export_sft_dataset import build_parser, main
from pare.trajectory import SCHEMA_VERSION


def _traj(trajectory_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "instance_id": "swe-1",
        "task": "Fix it",
        "model": "minimax",
        "seed": 0,
        "created_at": 1710000000.0,
        "llm_claimed_success": True,
        "verification": {
            "final_passed": True,
            "tier1_pass": True,
            "tier2_pass": True,
            "tier2_command": "",
        },
        "attempts": [],
        "tool_call_events": [
            {
                "turn_id": 0,
                "call_index_in_turn": 0,
                "global_index": 0,
                "tool_name": "bash",
                "params": {"command": "ls"},
                "result_success": True,
                "result_content": "out",
                "error_signal": "NONE",
                "timestamp": 0.0,
            }
        ],
        "token_usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "cache_create_tokens": 0,
        },
        "metadata": {},
    }


def _label(trajectory_id: str, outcome: str, contains_recovery: bool = False) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "instance_id": "swe-1",
        "seed": 0,
        "outcome": outcome,
        "liu_categories": [],
        "is_toxic": False,
        "contains_recovery": contains_recovery,
        "highest_recovery_level": "L2" if contains_recovery else None,
        "recovery_event_count": 1 if contains_recovery else 0,
        "recovery_events": [],
        "error_signal_counts": {},
        "liu_detail": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestExportSftDatasetCli:
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "--trajectory-jsonl", "traj.jsonl",
            "--output-jsonl", "sft.jsonl",
        ])
        assert args.include_outcomes is None
        assert args.include_recovery_only is False
        assert args.keep_toxic is False
        assert args.keep_empty_events is False
        assert args.system_prompt == ""
        assert args.max_trajectories is None

    def test_include_outcome_is_repeatable(self):
        """``--include-outcome`` accumulates (append action) so the
        3-arm ablation's "all successes" bucket can combine one_shot +
        with_recovery in a single invocation."""
        parser = build_parser()
        args = parser.parse_args([
            "--trajectory-jsonl", "t.jsonl",
            "--output-jsonl", "o.jsonl",
            "--include-outcome", "verified_one_shot",
            "--include-outcome", "verified_with_recovery",
        ])
        assert args.include_outcomes == [
            "verified_one_shot",
            "verified_with_recovery",
        ]

    def test_main_round_trip_writes_report_next_to_output(self, tmp_path: Path):
        """``main()`` writes both the JSONL and a sibling report.json
        — the report.json is the audit artefact for data-card claims."""
        traj_path = tmp_path / "arm.jsonl"
        out_path = tmp_path / "sft.jsonl"
        _write_jsonl(traj_path, [_traj("t1"), _traj("t2")])
        _write_jsonl(tmp_path / "arm.labels.jsonl", [
            _label("t1", "verified_with_recovery", contains_recovery=True),
            _label("t2", "verified_one_shot"),
        ])

        code = main([
            "--trajectory-jsonl", str(traj_path),
            "--output-jsonl", str(out_path),
            "--include-recovery-only",
        ])
        assert code == 0
        assert out_path.exists()

        # Report sidecar written at <output-jsonl>.report.json by default.
        report_path = Path(str(out_path) + ".report.json")
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["rows_written"] == 1
        assert report["filters"]["include_recovery_only"] is True

        # Exactly one SFT row, the recovery one.
        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["metadata"]["trajectory_id"] == "t1"

    def test_main_system_prompt_file_takes_precedence(self, tmp_path: Path):
        """Passing --system-prompt-file overrides --system-prompt (the
        file-based path is the realistic one for long multi-line system
        templates)."""
        traj_path = tmp_path / "arm.jsonl"
        out_path = tmp_path / "sft.jsonl"
        sys_file = tmp_path / "sys.txt"
        sys_file.write_text("FROM FILE", encoding="utf-8")
        _write_jsonl(traj_path, [_traj("t1")])

        code = main([
            "--trajectory-jsonl", str(traj_path),
            "--output-jsonl", str(out_path),
            "--system-prompt", "inline-ignored",
            "--system-prompt-file", str(sys_file),
            "--keep-toxic",  # no labels filter needed
        ])
        assert code == 0
        obj = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
        assert obj["messages"][0] == {"role": "system", "content": "FROM FILE"}
