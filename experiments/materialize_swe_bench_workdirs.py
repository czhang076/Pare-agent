"""Materialize SWE-bench tasks into local per-instance git workdirs.

Input tasks JSONL rows must include: instance_id, repo, base_commit, task.
Output JSONL adds cwd for each instance and can be consumed by
experiments.generate_trajectories.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


class MaterializeError(ValueError):
    """Raised when local workdir materialization fails."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_swe_bench_workdirs",
        description="Create local git workdirs for SWE-bench tasks and write cwd-mapped JSONL.",
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--repos-root", required=True)
    parser.add_argument("--workdirs-root", required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fetch-all", action="store_true")
    return parser


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        cmd = " ".join(args)
        stderr = (completed.stderr or "").strip()
        raise MaterializeError(f"Git command failed: {cmd}\n{stderr}")
    return completed


def _repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise MaterializeError(f"tasks JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as e:
                raise MaterializeError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(item, dict):
                raise MaterializeError(f"{path}:{line_no}: row must be object")
            rows.append(item)

    if not rows:
        raise MaterializeError("tasks JSONL has no rows")
    return rows


def _validate_task(task: dict[str, Any], idx: int) -> None:
    for key in ("instance_id", "repo", "base_commit", "task"):
        val = task.get(key)
        if not isinstance(val, str) or not val.strip():
            raise MaterializeError(f"task[{idx}] missing required field: {key}")


def ensure_repo_checkout(repo: str, repos_root: Path, *, fetch_all: bool = False) -> Path:
    repo_dir = repos_root / _repo_dir_name(repo)
    remote = f"https://github.com/{repo}.git"

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["git", "clone", "--filter=blob:none", "--no-checkout", remote, str(repo_dir)])

    if fetch_all:
        _run_git(["git", "-C", str(repo_dir), "fetch", "--all", "--prune"])

    return repo_dir


def ensure_commit_available(repo_dir: Path, commit: str) -> None:
    probe = subprocess.run(
        ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    _run_git(["git", "-C", str(repo_dir), "fetch", "origin", commit, "--depth=1"])
    _run_git(["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"])


def ensure_instance_workdir(
    *,
    repo_dir: Path,
    instance_id: str,
    commit: str,
    workdirs_root: Path,
    overwrite: bool = False,
) -> Path:
    instance_dir = workdirs_root / instance_id
    if overwrite:
        # Remove registered worktree first to avoid stale metadata conflicts.
        subprocess.run(
            ["git", "-C", str(repo_dir), "worktree", "remove", "--force", str(instance_dir)],
            capture_output=True,
            text=True,
        )

    if instance_dir.exists():
        if not overwrite:
            return instance_dir
        shutil.rmtree(instance_dir)

    instance_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        [
            "git", "-C", str(repo_dir),
            "worktree", "add", "--detach", str(instance_dir), commit,
        ]
    )
    return instance_dir


def materialize_tasks(
    tasks: list[dict[str, Any]],
    *,
    repos_root: Path,
    workdirs_root: Path,
    max_instances: int | None = None,
    overwrite: bool = False,
    fetch_all: bool = False,
) -> list[dict[str, Any]]:
    selected = tasks[:max_instances] if max_instances is not None else list(tasks)
    mapped: list[dict[str, Any]] = []
    repo_cache: dict[str, Path] = {}

    for idx, task in enumerate(selected):
        _validate_task(task, idx)
        repo = task["repo"]
        base_commit = task["base_commit"]
        instance_id = task["instance_id"]

        repo_dir = repo_cache.get(repo)
        if repo_dir is None:
            repo_dir = ensure_repo_checkout(repo, repos_root, fetch_all=fetch_all)
            repo_cache[repo] = repo_dir

        ensure_commit_available(repo_dir, base_commit)
        instance_dir = ensure_instance_workdir(
            repo_dir=repo_dir,
            instance_id=instance_id,
            commit=base_commit,
            workdirs_root=workdirs_root,
            overwrite=overwrite,
        )

        row = dict(task)
        row["cwd"] = str(instance_dir.resolve())
        mapped.append(row)

    return mapped


def _default_output_path(tasks_path: Path) -> Path:
    return tasks_path.with_suffix(".with_cwd.jsonl")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks_path = Path(args.tasks_jsonl)
    output_path = Path(args.output_jsonl) if args.output_jsonl else _default_output_path(tasks_path)
    repos_root = Path(args.repos_root).resolve()
    workdirs_root = Path(args.workdirs_root).resolve()

    try:
        tasks = _load_tasks(tasks_path)
        mapped = materialize_tasks(
            tasks,
            repos_root=repos_root,
            workdirs_root=workdirs_root,
            max_instances=args.max_instances,
            overwrite=args.overwrite,
            fetch_all=args.fetch_all,
        )
        _write_jsonl(output_path, mapped)
    except Exception as e:
        print(f"[materialize-failed] {e}", file=sys.stderr)
        return 1

    print(
        "[materialize-ok] "
        f"tasks={len(mapped)} "
        f"repos_root={repos_root} "
        f"workdirs_root={workdirs_root} "
        f"output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
