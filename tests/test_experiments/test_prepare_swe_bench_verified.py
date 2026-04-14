"""Tests for SWE-bench Verified preparation script."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from experiments.prepare_swe_bench_verified import main, prepare_tasks_jsonl


def _records(n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        out.append(
            {
                "instance_id": f"swe-{i:03d}",
                "repo": "org/repo",
                "base_commit": "abc123",
                "problem_statement": f"Fix issue {i}",
                "hints_text": f"hint {i}",
            }
        )
    return out


class TestPrepareSweBenchVerified:
    def test_prepare_tasks_jsonl_basic(self, tmp_path: Path):
        output = tmp_path / "tasks.jsonl"
        count = prepare_tasks_jsonl(
            _records(5),
            output_jsonl=output,
            sample_size=3,
            seed=7,
            repos_root=tmp_path / "repos",
        )

        assert count == 3
        lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 3
        row = json.loads(lines[0])
        assert row["repo"] == "org/repo"
        assert row["cwd"].endswith("repos\\org__repo") or row["cwd"].endswith("repos/org__repo")

    def test_prepare_with_repo_map_and_hints(self, tmp_path: Path):
        output = tmp_path / "tasks.jsonl"
        count = prepare_tasks_jsonl(
            _records(1),
            output_jsonl=output,
            sample_size=1,
            seed=0,
            repo_map={"org/repo": str(tmp_path / "mapped")},
            include_hints=True,
        )
        assert count == 1

        row = json.loads(output.read_text(encoding="utf-8").strip())
        assert row["cwd"] == str(tmp_path / "mapped")
        assert "Additional hints" in row["task"]

    def test_main_with_mocked_dataset(self, tmp_path: Path):
        output = tmp_path / "tasks.jsonl"
        with patch("experiments.prepare_swe_bench_verified._load_dataset_records", return_value=_records(4)):
            code = main([
                "--output-jsonl", str(output),
                "--sample-size", "2",
                "--seed", "1",
            ])

        assert code == 0
        lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2
