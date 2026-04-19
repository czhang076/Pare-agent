"""``pare inspect`` CLI entry point.

R0 scaffold — argument parsing only, command bodies raise NotImplementedError.
Real implementation lands in W1 Day 1-2 (single-trajectory mode) and
W2 Day 1-3 (diff mode + HTML output).

Usage:

    pare inspect traj.jsonl                      # single-trajectory classification
    pare inspect a.jsonl b.jsonl --diff          # success-vs-failure diff (W2)
    pare inspect --langfuse-trace <id>           # pull from Langfuse (W3)
    pare inspect --langfuse-diff <ok> <fail>     # diff via Langfuse (W3)
    pare inspect traj.jsonl --json               # machine-readable, for CI
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group(invoke_without_command=True)
@click.version_option(package_name="pare")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Pare 2.0 — coding-agent trajectory inspector."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("trajectories", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--diff", is_flag=True, help="Two-trajectory diff mode (requires 2 args).")
@click.option("--langfuse-trace", "langfuse_trace_id", default=None,
              help="Pull a single trajectory from Langfuse by trace id (W3).")
@click.option("--langfuse-diff", nargs=2, default=None,
              help="Pull two trajectories from Langfuse for diff mode (W3).")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("report.html"),
              help="HTML output path (default: report.html).")
@click.option("--json", "json_out", is_flag=True,
              help="Emit machine-readable JSON instead of HTML (for CI consumption).")
@click.option("--classify-only", is_flag=True,
              help="Print Liu classifications to stdout, skip rendering.")
def inspect(
    trajectories: tuple[Path, ...],
    diff: bool,
    langfuse_trace_id: str | None,
    langfuse_diff: tuple[str, str] | None,
    output: Path,
    json_out: bool,
    classify_only: bool,
) -> None:
    """Inspect one or two agent trajectories."""
    raise NotImplementedError(
        "pare inspect: implementation lands in W1-W2. "
        "See plan.md §3 'Week 1-2: Trajectory Inspector MVP'."
    )


if __name__ == "__main__":
    main()
