"""Tests for SWE-bench workdir materialization script."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import pytest

from experiments.materialize_swe_bench_workdirs import (
    apply_test_patch,
    main,
    materialize_tasks,
)


def _tasks() -> list[dict]:
    return [
        {
            "instance_id": "repo__1",
            "repo": "org/repo",
            "base_commit": "abc123",
            "task": "Fix bug A",
        },
        {
            "instance_id": "repo__2",
            "repo": "org/repo",
            "base_commit": "def456",
            "task": "Fix bug B",
        },
    ]


class TestMaterializeSweBenchWorkdirs:
    def test_materialize_tasks_with_mocked_git(self, tmp_path: Path):
        repos_root = tmp_path / "repos"
        workdirs_root = tmp_path / "workdirs"

        def fake_repo(repo: str, repos_root: Path, fetch_all: bool = False) -> Path:
            repo_dir = repos_root / repo.replace("/", "__")
            repo_dir.mkdir(parents=True, exist_ok=True)
            return repo_dir

        def fake_commit(repo_dir: Path, commit: str) -> None:
            _ = repo_dir
            _ = commit

        def fake_workdir(
            *,
            repo_dir: Path,
            instance_id: str,
            commit: str,
            workdirs_root: Path,
            overwrite: bool = False,
        ) -> Path:
            _ = repo_dir
            _ = commit
            _ = overwrite
            instance_dir = workdirs_root / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            return instance_dir

        with patch("experiments.materialize_swe_bench_workdirs.ensure_repo_checkout", side_effect=fake_repo), patch(
            "experiments.materialize_swe_bench_workdirs.ensure_commit_available", side_effect=fake_commit
        ), patch(
            "experiments.materialize_swe_bench_workdirs.ensure_instance_workdir", side_effect=fake_workdir
        ):
            out = materialize_tasks(
                _tasks(),
                repos_root=repos_root,
                workdirs_root=workdirs_root,
            )

        assert len(out) == 2
        assert "cwd" in out[0]
        assert (workdirs_root / "repo__1").exists()
        assert (workdirs_root / "repo__2").exists()

    def test_main_writes_output_jsonl(self, tmp_path: Path):
        input_path = tmp_path / "tasks.jsonl"
        output_path = tmp_path / "tasks_with_cwd.jsonl"
        repos_root = tmp_path / "repos"
        workdirs_root = tmp_path / "workdirs"

        with open(input_path, "w", encoding="utf-8") as f:
            for row in _tasks():
                f.write(json.dumps(row) + "\n")

        def fake_repo(repo: str, repos_root: Path, fetch_all: bool = False) -> Path:
            repo_dir = repos_root / repo.replace("/", "__")
            repo_dir.mkdir(parents=True, exist_ok=True)
            return repo_dir

        def fake_commit(repo_dir: Path, commit: str) -> None:
            _ = repo_dir
            _ = commit

        def fake_workdir(
            *,
            repo_dir: Path,
            instance_id: str,
            commit: str,
            workdirs_root: Path,
            overwrite: bool = False,
        ) -> Path:
            _ = repo_dir
            _ = commit
            _ = overwrite
            instance_dir = workdirs_root / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            return instance_dir

        with patch("experiments.materialize_swe_bench_workdirs.ensure_repo_checkout", side_effect=fake_repo), patch(
            "experiments.materialize_swe_bench_workdirs.ensure_commit_available", side_effect=fake_commit
        ), patch(
            "experiments.materialize_swe_bench_workdirs.ensure_instance_workdir", side_effect=fake_workdir
        ):
            code = main([
                "--tasks-jsonl",
                str(input_path),
                "--output-jsonl",
                str(output_path),
                "--repos-root",
                str(repos_root),
                "--workdirs-root",
                str(workdirs_root),
            ])

        assert code == 0
        lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2
        payload = json.loads(lines[0])
        assert "cwd" in payload


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
class TestApplyTestPatch:
    def _init_repo(self, workdir: Path, filename: str, content: str) -> str:
        workdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "-b", "main", str(workdir)], check=True)
        subprocess.run(
            ["git", "-C", str(workdir), "config", "user.email", "t@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(workdir), "config", "user.name", "test"],
            check=True,
        )
        (workdir / filename).write_text(content, encoding="utf-8", newline="\n")
        subprocess.run(["git", "-C", str(workdir), "add", filename], check=True)
        subprocess.run(
            ["git", "-C", str(workdir), "commit", "-q", "-m", "init"], check=True,
        )
        rev = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return rev.stdout.strip()

    def test_apply_test_patch_creates_commit(self, tmp_path: Path):
        workdir = tmp_path / "wd"
        base_rev = self._init_repo(workdir, "a.py", "x = 1\n")

        patch_text = (
            "diff --git a/test_new.py b/test_new.py\n"
            "new file mode 100644\n"
            "index 0000000..e69de29\n"
            "--- /dev/null\n"
            "+++ b/test_new.py\n"
            "@@ -0,0 +1 @@\n"
            "+assert True\n"
        )
        apply_test_patch(workdir, patch_text)

        assert (workdir / "test_new.py").exists()
        new_rev = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert new_rev != base_rev

        log = subprocess.run(
            ["git", "-C", str(workdir), "log", "-1", "--pretty=%s"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert "test_patch" in log

    def test_apply_empty_patch_noop(self, tmp_path: Path):
        workdir = tmp_path / "wd"
        base_rev = self._init_repo(workdir, "a.py", "x = 1\n")
        apply_test_patch(workdir, "")
        new_rev = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert new_rev == base_rev
