"""Pare entry point — CLI argument parsing and app bootstrap.

Usage:
    pare                              # Interactive mode
    pare "fix the bug in main.py"     # One-shot mode
    pare --provider minimax --model MiniMax-M2.5 "task"
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pare",
        description="Pare — the coding agent that never breaks your repo",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Task to execute (one-shot mode). Omit for interactive mode.",
    )
    parser.add_argument(
        "--provider", "-p",
        default="anthropic",
        choices=["anthropic", "openai", "minimax", "openrouter"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name (default: provider's default model)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: read from environment variable)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL (for local vLLM, etc.)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        import logging
        logging.basicConfig(level=logging.WARNING)

    from pathlib import Path
    from pare.cli.app import PareApp

    app = PareApp(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        cwd=Path(args.cwd) if args.cwd else None,
    )

    if args.task:
        exit_code = asyncio.run(app.run_once(args.task))
        sys.exit(exit_code)
    else:
        asyncio.run(app.run_interactive())


if __name__ == "__main__":
    main()
