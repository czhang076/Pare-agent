"""Pare entry point — headless batch execution.

Usage:
    pare "fix the bug in main.py"
    pare "task" --output result.json
    pare "task" --provider openrouter --model deepseek/deepseek-chat -o out.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pare",
        description="Pare — headless coding agent for trajectory generation",
    )
    parser.add_argument(
        "task",
        help="Task to execute.",
    )
    parser.add_argument(
        "--provider", "-p",
        default="openai",
        choices=["openai", "minimax", "openrouter"],
        help="LLM provider (default: openai)",
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
        "--output", "-o",
        default=None,
        help="Write structured JSON result to this path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
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
    from pare.cli.headless import run_headless

    exit_code = asyncio.run(run_headless(
        task=args.task,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        cwd=Path(args.cwd) if args.cwd else None,
        output_path=Path(args.output) if args.output else None,
        verbose=args.verbose,
    ))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
