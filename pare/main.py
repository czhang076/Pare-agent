"""Pare entry point — headless batch execution.

Usage:
    pare "fix the bug in main.py" --instance-id sympy__sympy-11618
    pare "task" --output result.json
    pare "task" --provider openrouter --model deepseek/deepseek-chat -o out.json

R5 state: only the flat ReAct loop inside an InstanceContainer remains.
``--cwd`` / ``--test-command`` are gone — the container's ``/testbed`` is
the sole working directory and Tier 2 runs via ``--verify`` inside that
container.
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
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name (resolves instance_id → image).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max LLM turns (default: 50).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After the ReAct loop, run Tier-2 verification inside the same "
            "container via SWE-bench's eval_script."
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
    from pare.cli.headless import run_headless_flat_react

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
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
