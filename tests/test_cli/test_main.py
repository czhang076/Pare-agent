"""Tests for pare/main.py — argument parsing."""

from pare.main import build_parser


class TestArgParser:
    def test_default_values(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.task is None
        assert args.provider == "anthropic"
        assert args.model is None
        assert args.verbose is False

    def test_one_shot_task(self):
        parser = build_parser()
        args = parser.parse_args(["fix the bug"])
        assert args.task == "fix the bug"

    def test_provider_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--provider", "minimax"])
        assert args.provider == "minimax"

    def test_provider_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-p", "openai"])
        assert args.provider == "openai"

    def test_model_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--model", "MiniMax-M2.5"])
        assert args.model == "MiniMax-M2.5"

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
