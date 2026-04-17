"""Tests for batch trajectory generation script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from experiments.generate_trajectories import (
    _resolve_tier2_command,
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
    def test_parser_defaults_for_planning_and_budget(self):
        parser = build_parser()
        args = parser.parse_args([
            "--tasks-jsonl", "tasks.jsonl",
            "--trajectory-jsonl", "traj.jsonl",
        ])
        assert args.use_planning is True
        assert args.max_tool_calls == 40
        assert args.max_tool_calls_per_step == 12

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

    async def test_tier2_python_cli_overrides_sys_executable(self, tmp_path: Path):
        """`--tier2-python` must take precedence over sys.executable when substituting `{python}`."""
        tasks = [
            GenerationTask(
                instance_id="swe-1",
                task="Task A",
                tier2_command="{python} -m pytest foo.py",
            ),
        ]
        mock_run = AsyncMock(return_value=0)
        fake_bin = "/opt/venv/bin/python"
        with patch("experiments.generate_trajectories.run_headless", mock_run):
            await generate_trajectories(
                tasks,
                trajectory_jsonl=tmp_path / "traj.jsonl",
                seeds=[0],
                tier2_python=fake_bin,
            )

        resolved = mock_run.await_args_list[0].kwargs["test_command"]
        assert resolved is not None
        assert fake_bin in resolved
        assert "{python}" not in resolved

    async def test_per_instance_tier2_overrides_cli(self, tmp_path: Path):
        """Row-level tier2_command must win over the global --test-command CLI arg."""
        tasks = [
            GenerationTask(
                instance_id="swe-1",
                task="Task A",
                tier2_command="pytest instance_specific.py",
            ),
            GenerationTask(instance_id="swe-2", task="Task B"),
        ]
        mock_run = AsyncMock(return_value=0)
        with patch("experiments.generate_trajectories.run_headless", mock_run):
            await generate_trajectories(
                tasks,
                trajectory_jsonl=tmp_path / "traj.jsonl",
                seeds=[0],
                test_command="pytest global.py",
            )

        assert mock_run.await_count == 2
        assert mock_run.await_args_list[0].kwargs["test_command"] == "pytest instance_specific.py"
        assert mock_run.await_args_list[1].kwargs["test_command"] == "pytest global.py"

    def test_resolve_tier2_substitutes_python_placeholder(self):
        """Spaces in the python path must be shell-quoted."""
        resolved = _resolve_tier2_command(
            "{python} -m pytest foo.py", "C:/venv with space/python.exe"
        )
        assert resolved is not None
        assert "{python}" not in resolved
        assert "pytest foo.py" in resolved
        assert "C:/venv with space/python.exe" in resolved
        # Spaces require quoting on both platforms.
        assert "'" in resolved or '"' in resolved

    def test_resolve_tier2_no_quote_when_no_whitespace(self):
        """Paths without whitespace must not be wrapped in quotes — cmd.exe
        treats POSIX single quotes as literal characters, which breaks the
        command on Windows."""
        resolved = _resolve_tier2_command(
            "{python} -m pytest foo.py", r"E:\venv\Scripts\python.exe"
        )
        assert resolved is not None
        assert resolved.startswith(r"E:\venv\Scripts\python.exe -m pytest")
        assert "'" not in resolved
        assert '"' not in resolved

    def test_resolve_tier2_passthrough_without_placeholder(self):
        assert _resolve_tier2_command("pytest foo.py", "/x/py") == "pytest foo.py"
        assert _resolve_tier2_command(None, "/x/py") is None
        assert _resolve_tier2_command("", "/x/py") is None

    async def test_generate_counts_with_mixed_exit_codes(self, tmp_path: Path):
        tasks = [
            GenerationTask(instance_id="swe-1", task="Task A"),
            GenerationTask(instance_id="swe-2", task="Task B"),
        ]

        mock_run = AsyncMock(side_effect=[0, 1, 2, 0])
        with patch("experiments.generate_trajectories.run_headless", mock_run):
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
        assert kwargs["use_planning"] is True
        assert kwargs["max_tool_calls"] == 40
        assert kwargs["max_tool_calls_per_step"] == 12

    async def test_stop_on_setup_error(self, tmp_path: Path):
        tasks = [
            GenerationTask(instance_id="swe-1", task="Task A"),
            GenerationTask(instance_id="swe-2", task="Task B"),
        ]

        mock_run = AsyncMock(side_effect=[2, 0, 0, 0])
        with patch("experiments.generate_trajectories.run_headless", mock_run):
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
        with patch("experiments.generate_trajectories.run_headless", mock_run):
            code = main([
                "--tasks-jsonl",
                str(tasks_path),
                "--trajectory-jsonl",
                str(tmp_path / "traj.jsonl"),
                "--report-json",
                str(report_path),
            ])

        assert code == 0
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["runs_requested"] == 1
        assert payload["runs_succeeded"] == 1
