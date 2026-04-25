"""Batch failure-injection runner.

Iterates ``REGISTRY × tasks × seeds``, applies each fault to a per-task
workdir, runs the agent, reverts, and appends one
``FaultInjectionResult`` row per (fault, task, seed) tuple to a JSONL.

v0 scope (this file)
--------------------

The CLI defaults to a **dry-run stub** agent runner that returns a
synthetic trajectory without actually invoking the LLM. The point of
v0 is to lock down:

- registry × task × seed iteration
- per-row JSONL output shape (``FaultInjectionResult.to_dict()``)
- summary table aggregation (per-fault counts)
- the apply→run→revert orchestration around a real workdir

Real integration with ``run_headless_flat_react`` is deferred to P1,
because the current headless runner manages its workdir inside an
``InstanceContainer`` (Docker), and a host-side fault mutation
doesn't reach into the container without either (a) a host-mode
agent variant or (b) container-side fault application. Either is a
larger architectural change than the scaffold below should carry.

Callers who want a real agent today can pass their own
``agent_runner`` to ``run_fault_injection_batch`` — the CLI's
dry-run path is just one possible value of that callback.

Typical invocation::

    python -m experiments.run_failure_injection \\
        --tasks-jsonl   data/sympy20/tasks.jsonl \\
        --output-jsonl  data/eval/fault_injection.jsonl \\
        --faults        wrong_import,empty_baseline \\
        --workdir-root  data/eval/workdirs \\
        --seeds         0 \\
        --max-instances 5

The CLI prints a per-fault summary table at the end. The output JSONL
is the auditable artefact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.generate_trajectories import (
    GenerationTask,
    load_tasks_jsonl,
    parse_seed_list,
)
from pare.eval.failure_injection import (
    REGISTRY,
    AgentRunner,
    FaultInjectionResult,
    run_with_fault,
)


# ---------------------------------------------------------------------------
# Dry-run stub agent runner
# ---------------------------------------------------------------------------


def dry_run_agent_runner(instance_id: str, workdir: Path) -> tuple[int, dict[str, Any]]:
    """Stub that returns a synthetic trajectory without running an LLM.

    Used as the CLI default until the headless runner gains a host-mode
    or in-container fault-injection path. Distinct ``trajectory_id``
    per call so downstream JSONL row-uniqueness checks still hold.
    """
    return (
        0,
        {
            "trajectory_id": f"dryrun_{instance_id}_{workdir.name}",
            "instance_id": instance_id,
            "tool_call_events": [],
            "_dry_run": True,
        },
    )


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FaultInjectionRunReport:
    """Aggregate counts for one batch run.

    ``per_fault`` rolls up exit-code and error counts so the CLI can
    print a quick table without re-reading the JSONL. ``rows_written``
    is the source-of-truth count for how many ``FaultInjectionResult``
    rows landed in the output file.
    """

    tasks_loaded: int
    tasks_run: int
    faults_run: list[str]
    seeds: list[int]
    rows_written: int
    rows_with_revert_failure: int
    rows_with_agent_failure: int
    per_fault: dict[str, dict[str, int]] = field(default_factory=dict)
    output_jsonl: Path = field(default_factory=lambda: Path())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_jsonl"] = str(self.output_jsonl)
        return d


def _classify_result(result: FaultInjectionResult) -> dict[str, bool]:
    """Convert a single result into the bucket flags the report counts."""
    err = result.error or ""
    return {
        "revert_failed": "revert_failed" in err,
        "agent_raised": result.agent_exit_code is None,
        "agent_nonzero_exit": (
            result.agent_exit_code is not None and result.agent_exit_code != 0
        ),
    }


def run_fault_injection_batch(
    tasks: list[GenerationTask],
    *,
    fault_names: list[str],
    output_jsonl: Path,
    seeds: list[int],
    agent_runner: AgentRunner,
    workdir_for: Callable[[str], Path],
    max_instances: int | None = None,
) -> FaultInjectionRunReport:
    """Iterate ``faults × tasks × seeds`` and append one result row per call.

    Args:
        tasks: From ``experiments.generate_trajectories.load_tasks_jsonl``.
        fault_names: Subset of ``REGISTRY`` keys to run; unknown names raise.
        output_jsonl: Output JSONL path. Parent dirs are created.
        seeds: Seeds to vary; each (task, fault) is run once per seed.
        agent_runner: ``(instance_id, workdir) -> (exit_code, trajectory)``.
                      Pass ``dry_run_agent_runner`` for smoke tests.
        workdir_for: Maps an instance_id to its on-disk workdir. The
                     caller owns workdir creation; faults mutate that
                     directory in place and revert on exit.
        max_instances: Optional cap; useful for smoke runs.

    Returns:
        ``FaultInjectionRunReport`` with per-fault aggregates.

    Raises:
        KeyError: if any name in ``fault_names`` is not in ``REGISTRY``.
    """
    unknown = [n for n in fault_names if n not in REGISTRY]
    if unknown:
        raise KeyError(
            f"unknown fault(s): {unknown}; known: {sorted(REGISTRY)}"
        )

    selected_tasks = (
        tasks[:max_instances] if max_instances is not None else list(tasks)
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    rows_with_revert_failure = 0
    rows_with_agent_failure = 0
    per_fault: dict[str, dict[str, int]] = {
        name: {
            "n_runs": 0,
            "n_agent_ok": 0,
            "n_agent_failed": 0,
            "n_agent_raised": 0,
            "n_revert_failed": 0,
        }
        for name in fault_names
    }

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for fault_name in fault_names:
            for task in selected_tasks:
                workdir = workdir_for(task.instance_id)
                for seed in seeds:
                    result = run_with_fault(
                        fault_name=fault_name,
                        instance_id=task.instance_id,
                        workdir=workdir,
                        agent_runner=agent_runner,
                    )
                    row = result.to_dict()
                    # Carry seed in the output row even though
                    # FaultInjectionResult itself is seed-agnostic — the
                    # batch is what knows about seeds.
                    row["seed"] = seed
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1

                    flags = _classify_result(result)
                    bucket = per_fault[fault_name]
                    bucket["n_runs"] += 1
                    if flags["agent_raised"]:
                        bucket["n_agent_raised"] += 1
                        rows_with_agent_failure += 1
                    elif flags["agent_nonzero_exit"]:
                        bucket["n_agent_failed"] += 1
                        rows_with_agent_failure += 1
                    else:
                        bucket["n_agent_ok"] += 1
                    if flags["revert_failed"]:
                        bucket["n_revert_failed"] += 1
                        rows_with_revert_failure += 1

    return FaultInjectionRunReport(
        tasks_loaded=len(tasks),
        tasks_run=len(selected_tasks),
        faults_run=list(fault_names),
        seeds=list(seeds),
        rows_written=rows_written,
        rows_with_revert_failure=rows_with_revert_failure,
        rows_with_agent_failure=rows_with_agent_failure,
        per_fault=per_fault,
        output_jsonl=output_jsonl,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_fault_list(raw: str) -> list[str]:
    """``'all'`` or comma-separated fault names. Preserves order, dedups."""
    if raw.strip() == "all":
        return sorted(REGISTRY)
    seen: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        if name not in seen:
            seen.append(name)
    if not seen:
        raise ValueError("--faults must list at least one fault name (or 'all')")
    return seen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_failure_injection",
        description=(
            "Batch-iterate REGISTRY x tasks x seeds, applying each fault, "
            "running the agent, and recording a FaultInjectionResult row."
        ),
    )
    parser.add_argument(
        "--tasks-jsonl",
        required=False,
        help="Tasks JSONL (same shape generate_trajectories consumes).",
    )
    parser.add_argument(
        "--output-jsonl",
        required=False,
        help="Output JSONL path; one FaultInjectionResult.to_dict() per line.",
    )
    parser.add_argument(
        "--faults",
        default="all",
        help="Comma-separated fault names, or 'all'. Default: 'all'.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seed list. Default: '0'.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Optional cap on tasks (after parsing the JSONL).",
    )
    parser.add_argument(
        "--workdir-root",
        default=None,
        help=(
            "Per-task workdir root. Each task gets "
            "<workdir-root>/<instance_id>/. The directory must exist; "
            "faults mutate files inside it and revert on exit."
        ),
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help=(
            "Optional path to write the FaultInjectionRunReport JSON. "
            "Defaults to <output-jsonl>.report.json."
        ),
    )
    parser.add_argument(
        "--list-faults",
        action="store_true",
        help="Print the registered faults and exit.",
    )
    return parser


def _print_list_faults() -> int:
    """Pretty-print the REGISTRY contents, one per line."""
    width = max((len(name) for name in REGISTRY), default=0)
    for name in sorted(REGISTRY):
        fault = REGISTRY[name]
        liu = f"[{fault.applies_to_liu}]" if fault.applies_to_liu else "[--]"
        print(f"  {name.ljust(width)}  {liu:6s}  {fault.description}")
    return 0


def _print_summary(report: FaultInjectionRunReport) -> None:
    """Compact per-fault summary, easy to eyeball or paste into a writeup."""
    print(
        f"[fault-injection-ok] "
        f"tasks={report.tasks_run}/{report.tasks_loaded} "
        f"faults={len(report.faults_run)} "
        f"seeds={len(report.seeds)} "
        f"rows={report.rows_written} "
        f"agent_failures={report.rows_with_agent_failure} "
        f"revert_failures={report.rows_with_revert_failure} "
        f"output={report.output_jsonl}"
    )
    if not report.per_fault:
        return
    name_width = max((len(name) for name in report.per_fault), default=0)
    print(
        f"  {'fault'.ljust(name_width)}  runs   ok  failed  raised  revert-failed"
    )
    for name in sorted(report.per_fault):
        b = report.per_fault[name]
        print(
            f"  {name.ljust(name_width)}  "
            f"{b['n_runs']:>4}  "
            f"{b['n_agent_ok']:>3}  "
            f"{b['n_agent_failed']:>6}  "
            f"{b['n_agent_raised']:>6}  "
            f"{b['n_revert_failed']:>13}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_faults:
        return _print_list_faults()

    # All non-list-faults invocations require these.
    missing = [
        flag
        for flag, val in (
            ("--tasks-jsonl", args.tasks_jsonl),
            ("--output-jsonl", args.output_jsonl),
            ("--workdir-root", args.workdir_root),
        )
        if not val
    ]
    if missing:
        print(
            f"[fault-injection-failed] missing required flag(s): "
            f"{', '.join(missing)}",
            file=sys.stderr,
        )
        return 1

    try:
        tasks = load_tasks_jsonl(Path(args.tasks_jsonl))
        seeds = parse_seed_list(args.seeds)
        fault_names = parse_fault_list(args.faults)
        output_jsonl = Path(args.output_jsonl)
        workdir_root = Path(args.workdir_root)

        def _workdir_for(instance_id: str) -> Path:
            wd = workdir_root / instance_id
            if not wd.exists():
                raise FileNotFoundError(
                    f"workdir does not exist: {wd}; "
                    "run experiments.materialize_swe_bench_workdirs first"
                )
            return wd

        report = run_fault_injection_batch(
            tasks,
            fault_names=fault_names,
            output_jsonl=output_jsonl,
            seeds=seeds,
            agent_runner=dry_run_agent_runner,
            workdir_for=_workdir_for,
            max_instances=args.max_instances,
        )
    except Exception as e:
        print(f"[fault-injection-failed] {e}", file=sys.stderr)
        return 1

    report_path = (
        Path(args.report_json)
        if args.report_json
        else Path(str(output_jsonl) + ".report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
