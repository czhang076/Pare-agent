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
        "--trajectory-jsonl",
        default=None,
        help="Append one trajectory record to this JSONL path.",
    )
    parser.add_argument(
        "--instance-id",
        default="local-run",
        help="Instance identifier recorded in trajectory JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--test-command",
        default=None,
        help="Optional Tier-2 verification command (for example: pytest -q).",
    )
    parser.add_argument(
        "--test-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for Tier-2 verification command.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--loop",
        choices=["new", "legacy"],
        default=None,
        help=(
            "Which agent loop to run. 'new' uses the flat ReAct loop "
            "+ long-lived InstanceContainer (R3+). 'legacy' uses the 3-layer "
            "orchestrator / executor. Default: read PARE_USE_NEW_LOOP env "
            "(1 → new, else → legacy). R4 flips this to 'new' by default; "
            "R5 deletes the legacy path."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help=(
            "Dataset name for --loop new (resolves instance_id → image). "
            "Ignored in legacy mode."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split for --loop new (default: test).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max LLM turns for --loop new (default: 50).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "--loop new only: after the ReAct loop, run Tier-2 verification "
            "inside the same container via SWE-bench's eval_script."
        ),
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
    from pare.cli.headless import (
        _flat_react_requested,
        run_headless,
        run_headless_flat_react,
    )

    if _flat_react_requested(args.loop):
        # R3 path: flat ReAct + long-lived container. Requires instance_id
        # to resolve a SWE-bench image. cwd / test_command are ignored —
        # the container's /testbed is the working directory.
        exit_code = asyncio.run(run_headless_flat_react(
            task=args.task,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            output_path=Path(args.output) if args.output else None,
            trajectory_path=Path(args.trajectory_jsonl) if args.trajectory_jsonl else None,
            instance_id=args.instance_id,
            dataset_name=args.dataset,
            split=args.split,
            seed=args.seed if args.seed is not None else 0,
            max_steps=args.max_steps,
            verify=args.verify,
            verbose=args.verbose,
        ))
    else:
        exit_code = asyncio.run(run_headless(
            task=args.task,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            cwd=Path(args.cwd) if args.cwd else None,
            output_path=Path(args.output) if args.output else None,
            trajectory_path=Path(args.trajectory_jsonl) if args.trajectory_jsonl else None,
            instance_id=args.instance_id,
            seed=args.seed if args.seed is not None else 0,
            test_command=args.test_command,
            test_timeout=args.test_timeout,
            verbose=args.verbose,
        ))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
