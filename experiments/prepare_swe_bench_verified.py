"""Prepare tasks JSONL from official SWE-bench Verified dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shlex
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
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated instance_id whitelist. Filters records before sampling.",
    )
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


def _parse_test_list(value: Any) -> list[str]:
    """Parse SWE-bench FAIL_TO_PASS / PASS_TO_PASS field into list[str].

    The HF dataset stores it as a JSON-encoded string; some mirrors store
    it already as a list. Silently ignore non-string / empty entries.
    """
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        if not value.strip():
            return []
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if not isinstance(decoded, list):
            return []
        raw_items = decoded
    else:
        return []

    out: list[str] = []
    for item in raw_items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


_DIFF_FILE_RE = re.compile(r"^diff --git a/(?P<path>\S+) b/\S+", re.MULTILINE)


def _test_files_from_patch(test_patch: str) -> list[str]:
    """Return unique file paths that the test_patch touches."""
    if not test_patch:
        return []
    seen: list[str] = []
    for match in _DIFF_FILE_RE.finditer(test_patch):
        path = match.group("path")
        if path not in seen:
            seen.append(path)
    return seen


def _build_tier2_command(fail_to_pass: list[str], test_patch: str = "") -> str:
    """Build a pytest command that asserts the FAIL_TO_PASS set now passes.

    SWE-bench stores FAIL_TO_PASS in two shapes:
      (a) pytest node ids — `path/test_file.py::TestClass::test_method`
      (b) bare function names — `test_something` (common in sympy)

    For (a) we pass them positionally. For (b) we scope pytest to the test
    files from test_patch and use `-k` name filtering; without scoping,
    pytest would collect the whole suite (minutes of overhead).

    Quoting: each token is shlex-quoted so parametrized ids and names with
    whitespace survive shell=True on both Unix and Windows.

    The `{python}` placeholder is substituted by the runner with the active
    interpreter path so tier2 uses the same venv as the pipeline, not
    whatever `python` happens to resolve to on PATH.
    """
    if not fail_to_pass:
        return ""

    has_node_ids = any("::" in t for t in fail_to_pass)
    if has_node_ids:
        quoted = " ".join(shlex.quote(t) for t in fail_to_pass)
        return f"{{python}} -m pytest {quoted} --tb=short -x --no-header -q"

    test_files = _test_files_from_patch(test_patch)
    k_expr = " or ".join(fail_to_pass)
    quoted_k = shlex.quote(k_expr)
    if test_files:
        files = " ".join(shlex.quote(f) for f in test_files)
        return f"{{python}} -m pytest {files} -k {quoted_k} --tb=short -x --no-header -q"

    # Last-resort fallback: run -k over the whole suite.
    return f"{{python}} -m pytest -k {quoted_k} --tb=short -x --no-header -q"


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
    instance_ids: list[str] | None = None,
) -> int:
    mapping = repo_map or {}
    if instance_ids:
        whitelist = set(instance_ids)
        filtered = [r for r in records if r.get("instance_id") in whitelist]
        if not filtered:
            raise PrepareDatasetError(
                f"No records matched instance_ids filter: {sorted(whitelist)}"
            )
        selected = filtered
    else:
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

        patch = record.get("patch") if isinstance(record.get("patch"), str) else ""
        test_patch = record.get("test_patch") if isinstance(record.get("test_patch"), str) else ""
        fail_to_pass = _parse_test_list(record.get("FAIL_TO_PASS"))
        pass_to_pass = _parse_test_list(record.get("PASS_TO_PASS"))

        row: dict[str, Any] = {
            "instance_id": instance_id,
            "task": _task_text(record, include_hints=include_hints),
            "repo": repo,
            "base_commit": base_commit if isinstance(base_commit, str) else "",
            "gold_patch": patch,
            "test_patch": test_patch,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
        }
        tier2_command = _build_tier2_command(fail_to_pass, test_patch=test_patch)
        if tier2_command:
            row["tier2_command"] = tier2_command
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

    instance_ids: list[str] | None = None
    if args.instance_ids:
        instance_ids = [tok.strip() for tok in args.instance_ids.split(",") if tok.strip()]
        if not instance_ids:
            print("[prepare-failed] --instance-ids provided but empty", file=sys.stderr)
            return 1

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
            instance_ids=instance_ids,
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
