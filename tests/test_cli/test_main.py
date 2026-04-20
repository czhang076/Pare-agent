"""Tests for pare/main.py — argument parsing.

R5 state: ``--cwd`` / ``--test-command`` / ``--test-timeout`` were dropped
along with the host-mode 3-layer agent. The working directory is always
``/testbed`` inside the InstanceContainer and Tier 2 runs via ``--verify``
(SWE-bench eval inside the same container).
"""

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
        assert args.verbose is False
        assert args.output is None
        assert args.dataset == "princeton-nlp/SWE-bench_Verified"
        assert args.split == "test"
        assert args.max_steps == 50
        assert args.verify is False

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

    def test_verify_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--verify", "fix bug"])
        assert args.verify is True

    def test_max_steps_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--max-steps", "12", "fix bug"])
        assert args.max_steps == 12

    def test_dataset_split_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--dataset", "princeton-nlp/SWE-bench_Lite",
            "--split", "dev",
            "fix bug",
        ])
        assert args.dataset == "princeton-nlp/SWE-bench_Lite"
        assert args.split == "dev"

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "-p", "minimax",
            "-m", "MiniMax-M2.5",
            "--base-url", "https://api.minimax.io/v1",
            "--instance-id", "swe-9",
            "--max-steps", "8",
            "--verify",
            "-v",
            "do something",
        ])
        assert args.provider == "minimax"
        assert args.model == "MiniMax-M2.5"
        assert args.base_url == "https://api.minimax.io/v1"
        assert args.instance_id == "swe-9"
        assert args.max_steps == 8
        assert args.verify is True
        assert args.verbose is True
        assert args.task == "do something"
