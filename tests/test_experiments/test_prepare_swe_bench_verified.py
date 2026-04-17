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
                "patch": f"gold-patch-{i}",
                "test_patch": f"test-patch-{i}",
                "FAIL_TO_PASS": json.dumps([f"tests/test_mod.py::test_{i}"]),
                "PASS_TO_PASS": json.dumps(["tests/test_mod.py::test_existing"]),
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

    def test_prepare_emits_tier2_fields(self, tmp_path: Path):
        """Rows must carry gold_patch, test_patch, fail_to_pass, tier2_command."""
        output = tmp_path / "tasks.jsonl"
        count = prepare_tasks_jsonl(
            _records(1),
            output_jsonl=output,
            sample_size=1,
            seed=0,
            repos_root=tmp_path / "repos",
        )
        assert count == 1
        row = json.loads(output.read_text(encoding="utf-8").strip())
        assert row["gold_patch"] == "gold-patch-0"
        assert row["test_patch"] == "test-patch-0"
        assert row["fail_to_pass"] == ["tests/test_mod.py::test_0"]
        assert row["pass_to_pass"] == ["tests/test_mod.py::test_existing"]
        assert "{python} -m pytest" in row["tier2_command"]
        assert "tests/test_mod.py::test_0" in row["tier2_command"]

    def test_prepare_omits_tier2_when_fail_to_pass_empty(self, tmp_path: Path):
        records = [
            {
                "instance_id": "swe-x",
                "repo": "org/repo",
                "base_commit": "abc",
                "problem_statement": "p",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }
        ]
        output = tmp_path / "tasks.jsonl"
        prepare_tasks_jsonl(
            records,
            output_jsonl=output,
            sample_size=1,
            seed=0,
            repos_root=tmp_path / "repos",
        )
        row = json.loads(output.read_text(encoding="utf-8").strip())
        assert row["fail_to_pass"] == []
        assert "tier2_command" not in row

    def test_tier2_command_uses_k_filter_for_bare_names(self, tmp_path: Path):
        """FAIL_TO_PASS as bare function names (sympy convention) must use -k + scoped files."""
        test_patch = (
            "diff --git a/sympy/polys/tests/test_rings.py b/sympy/polys/tests/test_rings.py\n"
            "--- a/sympy/polys/tests/test_rings.py\n"
            "+++ b/sympy/polys/tests/test_rings.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        records = [
            {
                "instance_id": "sympy-1",
                "repo": "sympy/sympy",
                "base_commit": "abc",
                "problem_statement": "p",
                "test_patch": test_patch,
                "FAIL_TO_PASS": json.dumps(["test_foo", "test_bar"]),
            }
        ]
        output = tmp_path / "tasks.jsonl"
        prepare_tasks_jsonl(
            records,
            output_jsonl=output,
            sample_size=1,
            seed=0,
            repos_root=tmp_path / "repos",
        )
        row = json.loads(output.read_text(encoding="utf-8").strip())
        cmd = row["tier2_command"]
        assert "sympy/polys/tests/test_rings.py" in cmd
        assert "-k" in cmd
        assert "test_foo or test_bar" in cmd

    def test_tier2_command_handles_django_unittest_form(self, tmp_path: Path):
        """Django FAIL_TO_PASS `test_x (dotted.module.Class)` → pytest node ids.

        When the test_patch reveals the real on-disk prefix
        (e.g. `tests/user_commands/tests.py`), the node id must carry that
        prefix — pytest runs from the repo root, not from `tests/`.
        """
        test_patch = (
            "diff --git a/tests/user_commands/tests.py b/tests/user_commands/tests.py\n"
            "--- a/tests/user_commands/tests.py\n"
            "+++ b/tests/user_commands/tests.py\n"
            "@@ -1,1 +1,1 @@\n-old\n+new\n"
            "diff --git a/tests/admin_filters/tests.py b/tests/admin_filters/tests.py\n"
            "--- a/tests/admin_filters/tests.py\n"
            "+++ b/tests/admin_filters/tests.py\n"
            "@@ -1,1 +1,1 @@\n-old\n+new\n"
        )
        records = [
            {
                "instance_id": "django-1",
                "repo": "django/django",
                "base_commit": "abc",
                "problem_statement": "p",
                "FAIL_TO_PASS": json.dumps([
                    "test_skip_checks (user_commands.tests.CommandRunTests)",
                    "test_other (admin_filters.tests.ListFiltersTests)",
                ]),
                "test_patch": test_patch,
            }
        ]
        output = tmp_path / "tasks.jsonl"
        prepare_tasks_jsonl(
            records,
            output_jsonl=output,
            sample_size=1,
            seed=0,
            repos_root=tmp_path / "repos",
        )
        row = json.loads(output.read_text(encoding="utf-8").strip())
        cmd = row["tier2_command"]
        assert "tests/user_commands/tests.py::CommandRunTests::test_skip_checks" in cmd
        assert "tests/admin_filters/tests.py::ListFiltersTests::test_other" in cmd
        # Node-id branch: must not use the -k filter path.
        assert " -k " not in cmd
        # Parentheses from the original form must be gone (they break cmd.exe).
        assert "(" not in cmd and ")" not in cmd

    def test_django_node_id_without_test_patch_falls_back_to_dotted_tail(self):
        """With no test_patch context, `_django_name_to_node_id` uses the
        dotted suffix as the file path. Callers must provide `test_files`
        (from the patch) to get the real on-disk prefix."""
        from experiments.prepare_swe_bench_verified import _django_name_to_node_id

        node = _django_name_to_node_id(
            "test_skip_checks (user_commands.tests.CommandRunTests)",
            test_files=None,
        )
        assert node == "user_commands/tests.py::CommandRunTests::test_skip_checks"

    def test_django_node_id_matches_suffix_even_with_other_prefix(self):
        """Module `a.b.Class` whose file lives at `nested/a/b.py` should
        pick up the prefix."""
        from experiments.prepare_swe_bench_verified import _django_name_to_node_id

        node = _django_name_to_node_id(
            "test_x (a.b.MyTest)",
            test_files=["nested/a/b.py"],
        )
        assert node == "nested/a/b.py::MyTest::test_x"

    def test_prepare_parses_list_fail_to_pass(self, tmp_path: Path):
        """Some dataset mirrors store FAIL_TO_PASS as an already-decoded list."""
        records = [
            {
                "instance_id": "swe-y",
                "repo": "org/repo",
                "base_commit": "abc",
                "problem_statement": "p",
                "FAIL_TO_PASS": ["tests/foo.py::test_a", "tests/foo.py::test_b"],
            }
        ]
        output = tmp_path / "tasks.jsonl"
        prepare_tasks_jsonl(
            records,
            output_jsonl=output,
            sample_size=1,
            seed=0,
            repos_root=tmp_path / "repos",
        )
        row = json.loads(output.read_text(encoding="utf-8").strip())
        assert row["fail_to_pass"] == ["tests/foo.py::test_a", "tests/foo.py::test_b"]
        assert "tests/foo.py::test_a" in row["tier2_command"]
        assert "tests/foo.py::test_b" in row["tier2_command"]
