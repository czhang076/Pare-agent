"""Batch trajectory generation script for pilot/full experiments.

Reads task definitions from JSONL and executes headless agent runs for each
task and seed, appending one trajectory record per run.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pare.cli.headless import run_headless


def _resolve_tier2_command(template: str | None, python_bin: str) -> str | None:
    """Substitute the `{python}` placeholder in a tier2 command template.

    Returns None when template is None or empty. The python binary is
    shlex-quoted so venv paths with whitespace survive `shell=True`.
    """
    if not template:
        return None
    if "{python}" not in template:
        return template
    return template.replace("{python}", shlex.quote(python_bin))


class GenerationError(ValueError):
    """Raised when generation input/config is invalid."""


@dataclass(frozen=True, slots=True)
class GenerationTask:
    instance_id: str
    task: str
    cwd: str | None = None
    tier2_command: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationReport:
    tasks_loaded: int
    tasks_run: int
    runs_requested: int
    runs_completed: int
    runs_succeeded: int
    runs_agent_failed: int
    runs_setup_failed: int
    seeds: list[int]
    trajectory_jsonl: Path

    def to_dict(self) -> dict:
        data = asdict(self)
        data["trajectory_jsonl"] = str(self.trajectory_jsonl)
        return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_trajectories",
        description="Run batched headless tasks and append trajectory JSONL records.",
    )
    parser.add_argument("--tasks-jsonl", required=True, help="Input tasks JSONL path.")
    parser.add_argument("--trajectory-jsonl", required=True, help="Output trajectory JSONL path.")
    parser.add_argument(
        "--provider", default="openai",
        choices=["openai", "minimax", "openrouter", "glm"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--cwd", default=None, help="Default working directory for tasks.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seed list, e.g. 0,1,2")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--test-command", default=None)
    parser.add_argument("--test-timeout", type=int, default=300)
    parser.add_argument(
        "--tier2-python",
        default=None,
        help=(
            "Python interpreter substituted for the `{python}` placeholder "
            "in tier2_command. Defaults to sys.executable. Override this "
            "when the calling shell's `python` is not the venv, so tier2 "
            "does not silently fall back to a global interpreter without "
            "the repo's dependencies."
        ),
    )
    parser.add_argument(
        "--use-planning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Orient -> Plan -> Execute mode (default: true).",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=40,
        help="Total tool-call budget per task (default: 40).",
    )
    parser.add_argument(
        "--max-tool-calls-per-step",
        type=int,
        default=12,
        help="Per-step tool-call budget (default: 12).",
    )
    parser.add_argument("--report-json", default=None, help="Optional report JSON output path.")
    parser.add_argument(
        "--stop-on-setup-error",
        action="store_true",
        help="Stop batch run when a setup/runtime infra error happens (exit code != 0/1).",
    )
    return parser


def parse_seed_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            seed = int(item)
        except ValueError as e:
            raise GenerationError(f"Invalid seed value: {item}") from e
        if seed not in values:
            values.append(seed)

    if not values:
        raise GenerationError("At least one seed is required.")
    return values


def load_tasks_jsonl(path: Path) -> list[GenerationTask]:
    if not path.exists():
        raise GenerationError(f"tasks JSONL does not exist: {path}")

    tasks: list[GenerationTask] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                raise GenerationError(f"{path}:{line_no}: invalid JSON: {e}") from e

            if not isinstance(payload, dict):
                raise GenerationError(f"{path}:{line_no}: task must be an object")

            instance_id = payload.get("instance_id")
            task = payload.get("task")
            cwd = payload.get("cwd")
            tier2_command = payload.get("tier2_command")

            if not isinstance(instance_id, str) or not instance_id.strip():
                raise GenerationError(f"{path}:{line_no}: instance_id must be non-empty str")
            if not isinstance(task, str) or not task.strip():
                raise GenerationError(f"{path}:{line_no}: task must be non-empty str")
            if cwd is not None and (not isinstance(cwd, str) or not cwd.strip()):
                raise GenerationError(f"{path}:{line_no}: cwd must be non-empty str when provided")
            if tier2_command is not None and (
                not isinstance(tier2_command, str) or not tier2_command.strip()
            ):
                raise GenerationError(
                    f"{path}:{line_no}: tier2_command must be non-empty str when provided"
                )

            tasks.append(
                GenerationTask(
                    instance_id=instance_id,
                    task=task,
                    cwd=cwd,
                    tier2_command=tier2_command,
                )
            )

    if not tasks:
        raise GenerationError("No tasks loaded from tasks JSONL.")
    return tasks


async def generate_trajectories(
    tasks: list[GenerationTask],
    *,
    trajectory_jsonl: Path,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    default_cwd: Path | None = None,
    seeds: list[int] | None = None,
    max_instances: int | None = None,
    test_command: str | None = None,
    test_timeout: int = 300,
    tier2_python: str | None = None,
    use_planning: bool = True,
    max_tool_calls: int = 40,
    max_tool_calls_per_step: int = 12,
    stop_on_setup_error: bool = False,
) -> GenerationReport:
    if not tasks:
        raise GenerationError("tasks list is empty")

    run_seeds = seeds or [0]
    selected_tasks = tasks[:max_instances] if max_instances is not None else list(tasks)
    python_bin = tier2_python or sys.executable

    runs_requested = len(selected_tasks) * len(run_seeds)
    runs_completed = 0
    runs_succeeded = 0
    runs_agent_failed = 0
    runs_setup_failed = 0

    for task in selected_tasks:
        run_cwd = Path(task.cwd).resolve() if task.cwd else default_cwd
        raw_test_command = task.tier2_command or test_command
        run_test_command = _resolve_tier2_command(raw_test_command, python_bin)

        for seed in run_seeds:
            exit_code = await run_headless(
                task=task.task,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                cwd=run_cwd,
                output_path=None,
                trajectory_path=trajectory_jsonl,
                instance_id=task.instance_id,
                seed=seed,
                test_command=run_test_command,
                test_timeout=test_timeout,
                use_planning=use_planning,
                max_tool_calls=max_tool_calls,
                max_tool_calls_per_step=max_tool_calls_per_step,
                verbose=False,
            )

            if exit_code == 0:
                runs_completed += 1
                runs_succeeded += 1
            elif exit_code == 1:
                runs_completed += 1
                runs_agent_failed += 1
            else:
                runs_setup_failed += 1
                if stop_on_setup_error:
                    return GenerationReport(
                        tasks_loaded=len(tasks),
                        tasks_run=len(selected_tasks),
                        runs_requested=runs_requested,
                        runs_completed=runs_completed,
                        runs_succeeded=runs_succeeded,
                        runs_agent_failed=runs_agent_failed,
                        runs_setup_failed=runs_setup_failed,
                        seeds=list(run_seeds),
                        trajectory_jsonl=trajectory_jsonl,
                    )

    return GenerationReport(
        tasks_loaded=len(tasks),
        tasks_run=len(selected_tasks),
        runs_requested=runs_requested,
        runs_completed=runs_completed,
        runs_succeeded=runs_succeeded,
        runs_agent_failed=runs_agent_failed,
        runs_setup_failed=runs_setup_failed,
        seeds=list(run_seeds),
        trajectory_jsonl=trajectory_jsonl,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks_jsonl(Path(args.tasks_jsonl))
        seeds = parse_seed_list(args.seeds)

        # Resolve --tier2-python to an absolute path if given. tier2 runs
        # with cwd=workdir (e.g. data/swebench_workdirs/<instance>), so a
        # relative path like `.venv-sympy/Scripts/python.exe` — which
        # resolves against the CLI invocation directory — breaks the moment
        # we hand it to a subprocess with a different cwd.
        #
        # Use absolute() not resolve(): resolve() follows symlinks, and a
        # venv's bin/python IS a symlink (→ /usr/bin/python3.12 on Linux).
        # Resolving it defeats the whole venv mechanism — sys.prefix logic
        # keys on sys.argv[0] being inside the venv dir, so the subprocess
        # silently runs the *system* interpreter without the venv's packages
        # (pytest, mpmath). Tier2 then fails with ModuleNotFoundError.
        tier2_python = (
            str(Path(args.tier2_python).absolute())
            if args.tier2_python else None
        )

        report = asyncio.run(
            generate_trajectories(
                tasks,
                trajectory_jsonl=Path(args.trajectory_jsonl),
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                default_cwd=Path(args.cwd).resolve() if args.cwd else None,
                seeds=seeds,
                max_instances=args.max_instances,
                test_command=args.test_command,
                test_timeout=args.test_timeout,
                tier2_python=tier2_python,
                use_planning=args.use_planning,
                max_tool_calls=args.max_tool_calls,
                max_tool_calls_per_step=args.max_tool_calls_per_step,
                stop_on_setup_error=args.stop_on_setup_error,
            )
        )
    except Exception as e:
        print(f"[generate-failed] {e}", file=sys.stderr)
        return 1

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    status = "generate-ok" if report.runs_setup_failed == 0 else "generate-partial"
    print(
        f"[{status}] "
        f"tasks={report.tasks_run}/{report.tasks_loaded} "
        f"runs={report.runs_completed}/{report.runs_requested} "
        f"success={report.runs_succeeded} "
        f"agent_failed={report.runs_agent_failed} "
        f"setup_failed={report.runs_setup_failed} "
        f"trajectory={report.trajectory_jsonl}"
    )

    return 0 if report.runs_setup_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
