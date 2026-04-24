"""``pare inspect`` CLI entry point.

Usage:

    pare inspect traj.jsonl --classify-only      # W1 Day 1 — classify + print table
    pare inspect traj.jsonl                      # single-trajectory HTML (W2)
    pare inspect a.jsonl b.jsonl --diff          # success-vs-failure diff (W2)
    pare inspect --langfuse-trace <id>           # pull from Langfuse (W3)
    pare inspect --langfuse-diff <ok> <fail>     # diff via Langfuse (W3)
    pare inspect traj.jsonl --json               # machine-readable, for CI
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from pare.inspector.annotator import AnnotatedTrajectory, annotate
from pare.inspector.loader import load_jsonl
from pare.inspector.semantic_tags import outcome_label


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
    if langfuse_trace_id or langfuse_diff:
        raise NotImplementedError("Langfuse sources land W3")
    if diff:
        raise NotImplementedError("--diff lands W1 Day 3-5 / W2")

    if not classify_only:
        raise NotImplementedError("HTML single-trajectory render lands W2")

    if len(trajectories) != 1:
        raise click.UsageError("--classify-only expects exactly one JSONL path")

    records = load_jsonl(trajectories[0])
    annotated = [annotate(r) for r in records]
    _render_summary_table(annotated)


def _render_summary_table(annotated: list[AnnotatedTrajectory]) -> None:
    table = Table(title="pare inspect — trajectory classification")
    table.add_column("trajectory_id")
    table.add_column("model")
    table.add_column("outcome")
    table.add_column("liu")
    table.add_column("tier2")

    for a in annotated:
        cats = a.liu_classification.categories
        liu_col = ", ".join(cats) if cats else "(none)"
        tier2 = "yes" if a.record.verification.tier2_pass else "no"
        outcome_col = f"{a.outcome_label} — {outcome_label(a.outcome_label)}"
        table.add_row(
            a.record.trajectory_id,
            a.record.model,
            outcome_col,
            liu_col,
            tier2,
        )

    Console().print(table)


if __name__ == "__main__":
    main()
