"""Tests for batch trajectory generation script.

R5 state: legacy ``run_headless`` / ``--tier2-python`` / ``--test-command``
/ ``--loop`` / ``--max-tool-calls`` wiring was deleted with the 3-layer
orchestrator. The surviving surface is a thin wrapper around
``run_headless_flat_react``; tests exercise argparse defaults, JSONL
loading, seed parsing, exit-code accounting, and report writing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from experiments.generate_trajectories import (
    build_parser,
    GenerationTask,
    generate_trajectories,
    load_tasks_jsonl,
    main,
    parse_seed_list,
)


def _write_tasks(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestGenerateTrajectories:
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-jsonl", "tasks.jsonl",
            "--trajectory-jsonl", "traj.jsonl",
        ])
        assert args.provider == "openai"
        assert args.seeds == "0"
        assert args.max_steps == 50
        assert args.verify is False
        assert args.dataset == "princeton-nlp/SWE-bench_Verified"
        assert args.split == "test"
        # Pre-passes must default off so existing pilot scripts keep
        # their behaviour — opt-in is the contract callers rely on.
        assert args.use_orient is False
        assert args.use_planner is False

    def test_parser_accepts_prepass_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-jsonl", "tasks.jsonl",
            "--trajectory-jsonl", "traj.jsonl",
            "--use-orient",
            "--use-planner",
        ])
        assert args.use_orient is True
        assert args.use_planner is True

    def test_parse_seed_list(self):
        assert parse_seed_list("0,1,2") == [0, 1, 2]
        assert parse_seed_list("0, 1, 1, 2") == [0, 1, 2]

    def test_load_tasks_jsonl(self, tmp_path: Path):
        tasks_path = tmp_path / "tasks.jsonl"
        _write_tasks(tasks_path, [
            {"instance_id": "swe-1", "task": "Fix parser"},
            {"instance_id": "swe-2", "task": "Add test", "cwd": str(tmp_path)},
        ])

        tasks = load_tasks_jsonl(tasks_path)
        assert len(tasks) == 2
        assert tasks[0].instance_id == "swe-1"
        assert tasks[1].cwd == str(tmp_path)

    def test_load_tasks_jsonl_carries_tier2_command(self, tmp_path: Path):
        """Row-level ``tier2_command`` still round-trips through the loader
        for forward-compat with task JSONLs authored before Tier 2 moved
        inside the container. The value is ignored by the flat ReAct loop
        (``--verify`` controls Tier 2), but preserving it avoids silently
        dropping data when operators re-run older task lists."""
        tasks_path = tmp_path / "tasks.jsonl"
        _write_tasks(tasks_path, [
            {
                "instance_id": "swe-1",
                "task": "Fix bug",
                "tier2_command": "python -m pytest tests/foo.py::test_x -x",
            },
        ])
        tasks = load_tasks_jsonl(tasks_path)
        assert tasks[0].tier2_command == "python -m pytest tests/foo.py::test_x -x"

    async def test_generate_counts_with_mixed_exit_codes(self, tmp_path: Path):
        tasks = [
            GenerationTask(instance_id="swe-1", task="Task A"),
            GenerationTask(instance_id="swe-2", task="Task B"),
        ]

        mock_run = AsyncMock(side_effect=[0, 1, 2, 0])
        with patch(
            "experiments.generate_trajectories.run_headless_flat_react",
            mock_run,
        ):
            report = await generate_trajectories(
                tasks,
                trajectory_jsonl=tmp_path / "traj.jsonl",
                seeds=[0, 1],
            )

        assert report.runs_requested == 4
        assert report.runs_completed == 3
        assert report.runs_succeeded == 2
        assert report.runs_agent_failed == 1
        assert report.runs_setup_failed == 1
        assert mock_run.await_count == 4
        kwargs = mock_run.await_args_list[0].kwargs
        assert kwargs["instance_id"] == "swe-1"
        assert kwargs["seed"] == 0
        assert kwargs["max_steps"] == 50
        assert kwargs["verify"] is False

    async def test_stop_on_setup_error(self, tmp_path: Path):
        tasks = [
            GenerationTask(instance_id="swe-1", task="Task A"),
            GenerationTask(instance_id="swe-2", task="Task B"),
        ]

        mock_run = AsyncMock(side_effect=[2, 0, 0, 0])
        with patch(
            "experiments.generate_trajectories.run_headless_flat_react",
            mock_run,
        ):
            report = await generate_trajectories(
                tasks,
                trajectory_jsonl=tmp_path / "traj.jsonl",
                seeds=[0, 1],
                stop_on_setup_error=True,
            )

        assert report.runs_setup_failed == 1
        assert mock_run.await_count == 1

    def test_main_writes_report(self, tmp_path: Path):
        tasks_path = tmp_path / "tasks.jsonl"
        report_path = tmp_path / "report.json"
        _write_tasks(tasks_path, [{"instance_id": "swe-1", "task": "Task"}])

        mock_run = AsyncMock(return_value=0)
        with patch(
            "experiments.generate_trajectories.run_headless_flat_react",
            mock_run,
        ):
            code = main([
                "--tasks-jsonl", str(tasks_path),
                "--trajectory-jsonl", str(tmp_path / "traj.jsonl"),
                "--report-json", str(report_path),
            ])

        assert code == 0
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["runs_requested"] == 1
        assert payload["runs_succeeded"] == 1

    def test_main_passes_verify_flag(self, tmp_path: Path):
        """``--verify`` must reach ``run_headless_flat_react`` as ``verify=True``."""
        tasks_path = tmp_path / "tasks.jsonl"
        _write_tasks(tasks_path, [{"instance_id": "swe-1", "task": "Task"}])

        mock_run = AsyncMock(return_value=0)
        with patch(
            "experiments.generate_trajectories.run_headless_flat_react",
            mock_run,
        ):
            code = main([
                "--tasks-jsonl", str(tasks_path),
                "--trajectory-jsonl", str(tmp_path / "traj.jsonl"),
                "--verify",
                "--max-steps", "7",
            ])

        assert code == 0
        kwargs = mock_run.await_args_list[0].kwargs
        assert kwargs["verify"] is True
        assert kwargs["max_steps"] == 7
        # Default state: pre-passes off, so every prior pilot invocation
        # remains byte-identical in behaviour.
        assert kwargs["use_orient"] is False
        assert kwargs["use_planner"] is False

    def test_main_passes_prepass_flags_through_to_headless(self, tmp_path: Path):
        """``--use-orient`` / ``--use-planner`` must reach the headless entry
        point. Without this wire, Phase 3.13.2 (orient_v2 repo map) and
        planner_v2 are unreachable from batch pilot runs — the feature exists
        in ``LoopConfig`` but no CLI path turns it on."""
        tasks_path = tmp_path / "tasks.jsonl"
        _write_tasks(tasks_path, [{"instance_id": "swe-1", "task": "Task"}])

        mock_run = AsyncMock(return_value=0)
        with patch(
            "experiments.generate_trajectories.run_headless_flat_react",
            mock_run,
        ):
            code = main([
                "--tasks-jsonl", str(tasks_path),
                "--trajectory-jsonl", str(tmp_path / "traj.jsonl"),
                "--use-orient",
                "--use-planner",
            ])

        assert code == 0
        kwargs = mock_run.await_args_list[0].kwargs
        assert kwargs["use_orient"] is True
        assert kwargs["use_planner"] is True
