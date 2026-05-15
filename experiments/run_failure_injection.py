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

    Used for fast smoke tests and as the safe default — distinct
    ``trajectory_id`` per call so downstream JSONL row-uniqueness
    checks still hold.
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


def make_container_agent_runner(
    *,
    provider: str,
    api_key: str,
    model: str | None = None,
    max_steps: int = 30,
    use_orient: bool = False,
    use_planner: bool = False,
    seed: int = 0,
    task_text_for: "Callable[[str], str]",
    trajectory_path: Path | None = None,
    verify: bool = False,
):
    """Build a ``ContainerAgentRunner`` callback for the SWE-bench Docker
    pipeline.

    Returns an ``async (instance_id, container) -> (exit_code, dict)``
    function that runs ``pare.agent.run_agent`` against a pre-built
    ``InstanceContainer`` (the same one the fault was applied to).

    Container lifecycle is **the caller's responsibility** — typically
    ``run_fault_injection_batch_in_container`` builds one container per
    (fault, task, seed) tuple and passes the live container in here.

    Lazy imports keep docker-eval and trajectory schema off the path
    for non-container invocations (--list-faults, --agent-mode=dry_run).
    """
    import time as _time
    from pare.agent.loop import LoopConfig, run_agent
    from pare.cli.headless import _loop_result_to_record
    from pare.llm import create_llm
    from pare.tools.base import create_default_registry
    from pare.trajectory.schema import append_trajectory_jsonl

    llm = create_llm(provider=provider, model=model, api_key=api_key)
    registry = create_default_registry()

    async def _runner(instance_id: str, container) -> tuple[int, dict[str, Any]]:
        task_text = task_text_for(instance_id)
        config = LoopConfig(
            system_prompt="",
            max_steps=max_steps,
            # Tier-2 inside the SWE-bench eval container is meaningful
            # (real pytest on the right env) — opt-in via --verify.
            verify_instance_id=instance_id if verify else None,
            use_orient=use_orient,
            use_planner=use_planner,
            use_test_nudge=False,
        )
        start = _time.time()
        loop_result = await run_agent(
            llm=llm,
            task=task_text,
            container=container,
            registry=registry,
            config=config,
        )
        elapsed = _time.time() - start
        record = _loop_result_to_record(
            task=task_text,
            instance_id=instance_id,
            provider=provider,
            model=llm.model,
            seed=seed,
            created_at=start,
            elapsed_seconds=elapsed,
            loop_result=loop_result,
            system_prompt="",
        )
        if trajectory_path is not None:
            append_trajectory_jsonl(trajectory_path, record)
        return (0 if loop_result.success else 1, record.to_dict())

    return _runner


async def run_fault_injection_batch_in_container(
    tasks: list[GenerationTask],
    *,
    fault_names: list[str],
    output_jsonl: Path,
    seeds: list[int],
    container_agent_runner,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    max_instances: int | None = None,
) -> FaultInjectionRunReport:
    """Container-mode parallel of ``run_fault_injection_batch``.

    For each (fault, task, seed): build an InstanceContainer for the
    task's SWE-bench instance, apply the fault inside it, run the
    agent against the same container, revert, tear down. One container
    per row (~10-30s boot overhead each) — wasteful but isolates state.

    Imports of ``InstanceContainer`` are deferred to function call to
    keep the docker-eval extras off the import path for non-container
    invocations.
    """
    from pare.sandbox.instance_container import InstanceContainer
    from pare.eval.failure_injection import run_with_fault_in_container

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
                for seed in seeds:
                    # One container per row. Wasteful (cold boot each
                    # time) but each fault sees a pristine SWE-bench
                    # image — no leaked state between runs.
                    container_cm = await InstanceContainer.build(
                        task.instance_id,
                        dataset_name=dataset_name,
                        split=split,
                    )
                    async with container_cm as container:
                        result = await run_with_fault_in_container(
                            fault_name=fault_name,
                            instance_id=task.instance_id,
                            container=container,
                            agent_runner=container_agent_runner,
                        )
                    row = result.to_dict()
                    row["seed"] = seed
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
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


def make_host_agent_runner(
    *,
    provider: str,
    api_key: str,
    model: str | None = None,
    max_steps: int = 30,
    use_orient: bool = False,
    use_planner: bool = False,
    seed: int = 0,
    task_text_for: Callable[[str], str],
    trajectory_path: Path | None = None,
) -> AgentRunner:
    """Build an ``AgentRunner`` callback that runs a real LLM agent
    against a host workdir (no Docker).

    Each call:
    1. Constructs a ``HostContainer`` for the workdir.
    2. Runs ``pare.agent.loop.run_agent`` with that container + the LLM.
    3. Converts ``LoopResult`` → ``TrajectoryRecord`` and optionally
       appends to ``trajectory_path``.
    4. Returns ``(exit_code, trajectory_dict)`` matching the
       ``AgentRunner`` protocol.

    ``task_text_for(instance_id)`` is a lookup function the caller
    supplies so this factory stays oblivious to how tasks are loaded —
    we only need the prompt text per instance.

    Lazy imports inside: keep the docker-eval extras + agent loop off
    the import path for ``--list-faults`` / ``--agent-mode=dry_run``
    invocations that don't need them.
    """
    import asyncio
    from pare.agent.loop import LoopConfig, run_agent
    from pare.cli.headless import _loop_result_to_record
    from pare.llm import create_llm
    from pare.sandbox.host_container import HostContainer
    from pare.tools.base import create_default_registry
    from pare.trajectory.schema import append_trajectory_jsonl

    llm = create_llm(provider=provider, model=model, api_key=api_key)
    registry = create_default_registry()

    def _runner(instance_id: str, workdir: Path) -> tuple[int, dict[str, Any]]:
        async def _go() -> tuple[int, dict[str, Any]]:
            task_text = task_text_for(instance_id)
            container = await HostContainer.from_workdir(workdir)
            config = LoopConfig(
                system_prompt="",
                max_steps=max_steps,
                # Tier-2 not available in host-mode (no SWE-bench eval
                # image). ``verify_instance_id=None`` keeps tier2_enabled
                # False on the LoopResult; classifier downstream judges
                # outcome from the trajectory shape alone.
                verify_instance_id=None,
                use_orient=use_orient,
                use_planner=use_planner,
                use_test_nudge=False,
            )
            import time as _time
            start = _time.time()
            try:
                async with container:
                    loop_result = await run_agent(
                        llm=llm,
                        task=task_text,
                        container=container,
                        registry=registry,
                        config=config,
                    )
            except Exception as e:
                # Surface as runner-raised so the FaultInjectionResult
                # captures it correctly (agent_exit_code=None).
                raise RuntimeError(
                    f"run_agent crashed on {instance_id}: {type(e).__name__}: {e}"
                ) from e

            elapsed = _time.time() - start
            record = _loop_result_to_record(
                task=task_text,
                instance_id=instance_id,
                provider=provider,
                model=llm.model,
                seed=seed,
                created_at=start,
                elapsed_seconds=elapsed,
                loop_result=loop_result,
                system_prompt="",
            )
            if trajectory_path is not None:
                append_trajectory_jsonl(trajectory_path, record)

            return (0 if loop_result.success else 1, record.to_dict())

        return asyncio.run(_go())

    return _runner


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

    # -- real agent (LLM-backed, host-mode) ---------------------------------
    parser.add_argument(
        "--agent-mode",
        default="dry_run",
        choices=("dry_run", "host", "container"),
        help=(
            "dry_run (default): synthetic trajectory, no LLM, ~0 tokens. "
            "host: real LLM through pare.agent.run_agent on a host "
            "workdir via HostContainer (Linux/WSL only, no Tier-2). "
            "container: real LLM inside an InstanceContainer per task "
            "(Docker required, supports Tier-2 via --verify). "
            "Both 'host' and 'container' consume provider tokens."
        ),
    )
    parser.add_argument(
        "--provider",
        default="minimax",
        choices=("openai", "minimax", "openrouter", "glm"),
        help="LLM provider (host-mode only). Default: minimax.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (host-mode only). Defaults to the provider's default.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key (host-mode only). If omitted, read from "
            "<PROVIDER>_API_KEY env var."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help=(
            "Max LLM turns per agent run (host-mode only). Default: 30 — "
            "lower than generate_trajectories' 50 because faulted runs "
            "tend to be either trivial recoveries or instant declarations."
        ),
    )
    parser.add_argument(
        "--trajectory-jsonl",
        default=None,
        help=(
            "Per-trajectory record JSONL (host-mode only). Appends one "
            "TrajectoryRecord per (fault, task, seed). Required when "
            "--agent-mode=host. Same shape generate_trajectories writes."
        ),
    )
    parser.add_argument(
        "--use-orient",
        action="store_true",
        help="Enable orient_v2 pre-pass (host or container mode).",
    )
    parser.add_argument(
        "--use-planner",
        action="store_true",
        help="Enable planner_v2 pre-pass (host or container mode).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Tier-2 verification inside the SWE-bench eval container. "
            "Container-mode only — host-mode has no SWE-bench eval image."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name (container-mode only).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (container-mode only).",
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

    # All invocations need tasks + output. ``--workdir-root`` is only
    # required for dry_run and host modes (those mutate host paths);
    # container mode lets InstanceContainer materialize the workdir
    # from the SWE-bench image itself.
    required = [("--tasks-jsonl", args.tasks_jsonl), ("--output-jsonl", args.output_jsonl)]
    if args.agent_mode in ("dry_run", "host"):
        required.append(("--workdir-root", args.workdir_root))
    missing = [flag for flag, val in required if not val]
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

        # task text lookup used by both host and container modes
        task_text_by_id = {t.instance_id: t.task for t in tasks}

        def _task_text_for(iid: str) -> str:
            if iid not in task_text_by_id:
                raise KeyError(f"instance_id {iid!r} not in tasks JSONL")
            return task_text_by_id[iid]

        def _resolve_key() -> str:
            import os as _os
            key = (
                args.api_key
                or _os.environ.get(f"{args.provider.upper()}_API_KEY", "")
            )
            if not key:
                raise ValueError(
                    f"--agent-mode={args.agent_mode} requires an API key. "
                    f"Pass --api-key or set "
                    f"{args.provider.upper()}_API_KEY env var."
                )
            return key

        # -- agent-mode dispatch -------------------------------------------
        if args.agent_mode == "container":
            import asyncio as _asyncio
            if not args.trajectory_jsonl:
                raise ValueError(
                    "--agent-mode=container requires --trajectory-jsonl "
                    "(where to record one TrajectoryRecord per agent run)"
                )
            resolved_key = _resolve_key()
            container_agent_runner = make_container_agent_runner(
                provider=args.provider,
                api_key=resolved_key,
                model=args.model,
                max_steps=args.max_steps,
                use_orient=args.use_orient,
                use_planner=args.use_planner,
                seed=seeds[0] if seeds else 0,
                task_text_for=_task_text_for,
                trajectory_path=Path(args.trajectory_jsonl),
                verify=args.verify,
            )
            report = _asyncio.run(
                run_fault_injection_batch_in_container(
                    tasks,
                    fault_names=fault_names,
                    output_jsonl=output_jsonl,
                    seeds=seeds,
                    container_agent_runner=container_agent_runner,
                    dataset_name=args.dataset,
                    split=args.split,
                    max_instances=args.max_instances,
                )
            )
        else:
            # dry_run or host — both use workdir_root + the sync batch.
            workdir_root = Path(args.workdir_root)

            def _workdir_for(instance_id: str) -> Path:
                wd = workdir_root / instance_id
                if not wd.exists():
                    raise FileNotFoundError(
                        f"workdir does not exist: {wd}; "
                        "run experiments.materialize_swe_bench_workdirs first"
                    )
                return wd

            if args.agent_mode == "host":
                resolved_key = _resolve_key()
                if not args.trajectory_jsonl:
                    raise ValueError(
                        "--agent-mode=host requires --trajectory-jsonl"
                    )
                agent_runner = make_host_agent_runner(
                    provider=args.provider,
                    api_key=resolved_key,
                    model=args.model,
                    max_steps=args.max_steps,
                    use_orient=args.use_orient,
                    use_planner=args.use_planner,
                    seed=seeds[0] if seeds else 0,
                    task_text_for=_task_text_for,
                    trajectory_path=Path(args.trajectory_jsonl),
                )
            else:
                agent_runner = dry_run_agent_runner

            report = run_fault_injection_batch(
                tasks,
                fault_names=fault_names,
                output_jsonl=output_jsonl,
                seeds=seeds,
                agent_runner=agent_runner,
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
