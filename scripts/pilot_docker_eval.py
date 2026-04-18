"""Phase A: exercise DockerEvalSession standalone, without touching Pare.

Usage:
    python scripts/pilot_docker_eval.py --instance sympy__sympy-20639 --patch-file gold.diff
    python scripts/pilot_docker_eval.py --instance sympy__sympy-20639 --empty-patch

This script exists to validate the harness wiring before any trajectory
generation. It loads one instance from SWE-bench Verified, feeds a patch
(or an intentionally-empty one) to run_instance, and prints the resolved
bool plus the full Tier2CheckResult.

Requires the docker-eval extra:
    pip install -e ".[docker-eval]"

And a running Docker daemon.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pare.sandbox.docker_eval import DockerEvalConfig, build_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pilot_docker_eval",
        description="Run one SWE-bench instance through the docker harness.",
    )
    parser.add_argument("--instance", required=True, help="instance_id, e.g. sympy__sympy-20639")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patch-file", type=Path, help="Path to a unified-diff file.")
    group.add_argument(
        "--empty-patch",
        action="store_true",
        help="Use an empty diff — expect passed=False with empty_diff_skipped_docker or "
        "harness patch-apply failure (sanity test).",
    )
    parser.add_argument("--run-id", default="pare-pilot", help="Harness run_id (default: pare-pilot).")
    parser.add_argument(
        "--model-name",
        default="pare_pilot",
        help="model_name_or_path recorded in harness predictions.",
    )
    parser.add_argument("--timeout", type=int, default=1800)

    args = parser.parse_args(argv)

    if args.empty_patch:
        diff = ""
    else:
        if not args.patch_file.exists():
            print(f"[err] patch file not found: {args.patch_file}", file=sys.stderr)
            return 2
        diff = args.patch_file.read_text(encoding="utf-8")
        print(f"[pilot] loaded {len(diff)} bytes from {args.patch_file}", file=sys.stderr)

    session = build_session(
        DockerEvalConfig(
            model_name=args.model_name,
            run_id=args.run_id,
            timeout=args.timeout,
        )
    )
    try:
        result = session.verify_diff(args.instance, diff)
    finally:
        session.close()

    print(f"instance:    {args.instance}")
    print(f"passed:      {result.passed}")
    print(f"return_code: {result.return_code}")
    print(f"command:     {result.command}")
    if result.error:
        print(f"error:       {result.error}")
    if result.output:
        truncated = result.output if len(result.output) < 500 else result.output[:500] + "..."
        print(f"output:      {truncated}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
