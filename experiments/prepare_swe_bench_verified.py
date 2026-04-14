"""Prepare tasks JSONL from official SWE-bench Verified dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any


class PrepareDatasetError(ValueError):
    """Raised when dataset preparation fails."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_swe_bench_verified",
        description="Convert SWE-bench Verified split into tasks JSONL.",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--repos-root", default=None)
    parser.add_argument("--repo-map-json", default=None)
    parser.add_argument("--include-hints", action="store_true")
    return parser


def _load_dataset_records(split: str, cache_dir: Path | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise PrepareDatasetError(
            "Missing dependency 'datasets'. Install via: .venv\\Scripts\\python.exe -m pip install datasets"
        ) from e

    dataset = load_dataset(
        "princeton-nlp/SWE-bench_Verified",
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    rows = [item for item in dataset if isinstance(item, dict)]
    if not rows:
        raise PrepareDatasetError("Dataset split is empty")
    return rows


def _load_repo_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise PrepareDatasetError(f"repo map file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise PrepareDatasetError(f"Invalid repo map JSON: {e}") from e
    if not isinstance(payload, dict):
        raise PrepareDatasetError("repo map JSON must be an object")

    out: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise PrepareDatasetError("repo map key must be non-empty str")
        if not isinstance(value, str) or not value.strip():
            raise PrepareDatasetError("repo map value must be non-empty str")
        out[key] = value
    return out


def _sample_records(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise PrepareDatasetError("sample_size must be > 0")
    if sample_size >= len(records):
        return list(records)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), sample_size))
    return [records[i] for i in indices]


def _default_cwd(repo: str, repos_root: Path | None) -> str | None:
    if repos_root is None:
        return None
    return str((repos_root / repo.replace("/", "__")).resolve())


def _task_text(record: dict[str, Any], include_hints: bool) -> str:
    problem = record.get("problem_statement")
    if not isinstance(problem, str) or not problem.strip():
        raise PrepareDatasetError("record missing problem_statement")
    text = problem.strip()
    if include_hints:
        hints = record.get("hints_text")
        if isinstance(hints, str) and hints.strip():
            text = f"{text}\n\nAdditional hints:\n{hints.strip()}"
    return text


def prepare_tasks_jsonl(
    records: list[dict[str, Any]],
    *,
    output_jsonl: Path,
    sample_size: int,
    seed: int,
    repos_root: Path | None = None,
    repo_map: dict[str, str] | None = None,
    include_hints: bool = False,
) -> int:
    mapping = repo_map or {}
    selected = _sample_records(records, sample_size=sample_size, seed=seed)

    rows: list[dict[str, Any]] = []
    for record in selected:
        instance_id = record.get("instance_id")
        repo = record.get("repo")
        base_commit = record.get("base_commit")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise PrepareDatasetError("record missing instance_id")
        if not isinstance(repo, str) or not repo.strip():
            raise PrepareDatasetError("record missing repo")

        cwd = mapping.get(repo) or _default_cwd(repo, repos_root)
        row: dict[str, Any] = {
            "instance_id": instance_id,
            "task": _task_text(record, include_hints=include_hints),
            "repo": repo,
            "base_commit": base_commit if isinstance(base_commit, str) else "",
        }
        if cwd:
            row["cwd"] = cwd
        rows.append(row)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_path = Path(args.output_jsonl)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    repos_root = Path(args.repos_root) if args.repos_root else None
    repo_map_path = Path(args.repo_map_json) if args.repo_map_json else None

    try:
        records = _load_dataset_records(args.split, cache_dir=cache_dir)
        repo_map = _load_repo_map(repo_map_path)
        written = prepare_tasks_jsonl(
            records,
            output_jsonl=output_path,
            sample_size=args.sample_size,
            seed=args.seed,
            repos_root=repos_root,
            repo_map=repo_map,
            include_hints=args.include_hints,
        )
    except Exception as e:
        print(f"[prepare-failed] {e}", file=sys.stderr)
        return 1

    print(
        "[prepare-ok] "
        f"split={args.split} sample_size={args.sample_size} written={written} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
