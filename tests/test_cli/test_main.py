"""Tests for pare/main.py — argument parsing."""

import pytest

from pare.main import build_parser


class TestArgParser:
    def test_task_is_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_values_with_task(self):
        parser = build_parser()
        args = parser.parse_args(["fix the bug"])
        assert args.task == "fix the bug"
        assert args.provider == "openai"
        assert args.model is None
        assert args.trajectory_jsonl is None
        assert args.instance_id == "local-run"
        assert args.test_command is None
        assert args.test_timeout == 300
        assert args.verbose is False
        assert args.output is None

    def test_one_shot_task(self):
        parser = build_parser()
        args = parser.parse_args(["fix the bug"])
        assert args.task == "fix the bug"

    def test_provider_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--provider", "minimax", "fix bug"])
        assert args.provider == "minimax"

    def test_provider_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-p", "openai", "fix bug"])
        assert args.provider == "openai"

    def test_model_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--model", "MiniMax-M2.5", "fix bug"])
        assert args.model == "MiniMax-M2.5"

    def test_output_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "result.json", "fix bug"])
        assert args.output == "result.json"

    def test_trajectory_jsonl_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--trajectory-jsonl", "traj.jsonl", "fix bug"])
        assert args.trajectory_jsonl == "traj.jsonl"

    def test_instance_id_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--instance-id", "swe-123", "fix bug"])
        assert args.instance_id == "swe-123"

    def test_test_command_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--test-command", "pytest -q", "fix bug"])
        assert args.test_command == "pytest -q"

    def test_test_timeout_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--test-timeout", "120", "fix bug"])
        assert args.test_timeout == 120

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "-p", "minimax",
            "-m", "MiniMax-M2.5",
            "--base-url", "https://api.minimax.io/v1",
            "--cwd", "/tmp/project",
            "-v",
            "do something",
        ])
        assert args.provider == "minimax"
        assert args.model == "MiniMax-M2.5"
        assert args.base_url == "https://api.minimax.io/v1"
        assert args.cwd == "/tmp/project"
        assert args.verbose is True
        assert args.task == "do something"
