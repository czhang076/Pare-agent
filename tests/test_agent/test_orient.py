"""Tests for the Orient phase — repo scanning."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pare.agent.orient import (
    FileSignature,
    GitInfo,
    RepoContext,
    RepoScanner,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Create a minimal project structure for scanning."""
    # Python files
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(
        "class App:\n    pass\n\ndef run():\n    pass\n\ndef _private():\n    pass\n"
    )
    (src / "utils.py").write_text(
        "def helper_one():\n    pass\n\ndef helper_two():\n    pass\n"
    )

    # Subdirectory
    sub = src / "api"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "routes.py").write_text(
        "async def get_users():\n    pass\n\nclass Router:\n    pass\n"
    )

    # Tests
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_run():\n    assert True\n")

    # Key files
    (tmp_path / "README.md").write_text("# Test Project\n\nA test project.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')

    # Noise directories (should be ignored)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"fake")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".git").mkdir()

    return tmp_path


# ---------------------------------------------------------------------------
# Tree scanning
# ---------------------------------------------------------------------------


class TestScanTree:
    @pytest.mark.asyncio
    async def test_builds_tree(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert ctx.tree != ""
        assert "src/" in ctx.tree
        assert "tests/" in ctx.tree

    @pytest.mark.asyncio
    async def test_ignores_noise_dirs(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert "__pycache__" not in ctx.tree
        assert "node_modules" not in ctx.tree
        assert ".git" not in ctx.tree

    @pytest.mark.asyncio
    async def test_counts_files_and_dirs(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert ctx.total_files > 0
        assert ctx.total_dirs > 0

    @pytest.mark.asyncio
    async def test_depth_limit(self, repo: Path):
        # Create deep nesting
        deep = repo / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("x = 1\n")

        scanner = RepoScanner(repo, max_depth=2)
        ctx = await scanner.scan()
        # "e/" should not appear at depth 5
        assert "deep.py" not in ctx.tree


# ---------------------------------------------------------------------------
# Signature scanning
# ---------------------------------------------------------------------------


class TestScanSignatures:
    @pytest.mark.asyncio
    async def test_finds_python_signatures(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()

        # Find src/main.py signatures
        main_sigs = [s for s in ctx.signatures if s.path == "src/main.py"]
        assert len(main_sigs) == 1
        assert "App" in main_sigs[0].signatures
        assert "run" in main_sigs[0].signatures

    @pytest.mark.asyncio
    async def test_skips_private_functions(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()

        main_sigs = [s for s in ctx.signatures if s.path == "src/main.py"]
        assert "_private" not in main_sigs[0].signatures

    @pytest.mark.asyncio
    async def test_finds_async_functions(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()

        route_sigs = [s for s in ctx.signatures if "routes.py" in s.path]
        assert len(route_sigs) == 1
        assert "get_users" in route_sigs[0].signatures
        assert "Router" in route_sigs[0].signatures

    @pytest.mark.asyncio
    async def test_respects_max_files(self, repo: Path):
        scanner = RepoScanner(repo, max_files_for_signatures=1)
        ctx = await scanner.scan()
        assert len(ctx.signatures) <= 1


# ---------------------------------------------------------------------------
# Key files
# ---------------------------------------------------------------------------


class TestScanKeyFiles:
    @pytest.mark.asyncio
    async def test_reads_readme(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert "README.md" in ctx.key_file_previews
        assert "Test Project" in ctx.key_file_previews["README.md"]

    @pytest.mark.asyncio
    async def test_reads_pyproject(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert "pyproject.toml" in ctx.key_file_previews

    @pytest.mark.asyncio
    async def test_skips_missing_files(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert "Cargo.toml" not in ctx.key_file_previews


# ---------------------------------------------------------------------------
# Git status
# ---------------------------------------------------------------------------


class TestScanGit:
    @pytest.mark.asyncio
    async def test_detects_non_git_repo(self, tmp_path: Path):
        scanner = RepoScanner(tmp_path)
        ctx = await scanner.scan()
        assert not ctx.git.is_git_repo

    @pytest.mark.asyncio
    async def test_detects_git_repo(self, repo: Path):
        # repo fixture has .git dir but it's fake — create a real one
        import shutil
        shutil.rmtree(repo / ".git")

        async def _run(*args):
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=str(repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode

        await _run("git", "init")
        await _run("git", "config", "user.email", "test@test.com")
        await _run("git", "config", "user.name", "Test")
        await _run("git", "add", ".")
        rc = await _run("git", "commit", "-m", "initial")
        assert rc == 0

        scanner = RepoScanner(repo)
        ctx = await scanner.scan()
        assert ctx.git.is_git_repo
        assert ctx.git.branch in ("main", "master")
        assert len(ctx.git.recent_commits) >= 1


# ---------------------------------------------------------------------------
# RepoContext rendering
# ---------------------------------------------------------------------------


class TestRepoContext:
    def test_to_markdown(self):
        ctx = RepoContext(
            tree="project/\n├── src/\n└── tests/",
            total_files=10,
            total_dirs=3,
            total_lines=500,
            signatures=[
                FileSignature(path="src/main.py", signatures=["App", "run"]),
            ],
            git=GitInfo(
                is_git_repo=True,
                branch="main",
                recent_commits=["abc1234 initial commit"],
                uncommitted_changes=["src/main.py"],
            ),
        )
        md = ctx.to_markdown()
        assert "## Structure" in md
        assert "10 files" in md
        assert "## Key Signatures" in md
        assert "App, run" in md
        assert "## Git" in md
        assert "main" in md

    def test_to_markdown_minimal(self):
        ctx = RepoContext()
        md = ctx.to_markdown()
        assert md == ""  # Empty context produces empty output


# ---------------------------------------------------------------------------
# Full integration
# ---------------------------------------------------------------------------


class TestFullScan:
    @pytest.mark.asyncio
    async def test_scan_produces_complete_context(self, repo: Path):
        scanner = RepoScanner(repo)
        ctx = await scanner.scan()

        # All four scans should have produced data
        assert ctx.tree != ""
        assert len(ctx.signatures) > 0
        assert len(ctx.key_file_previews) > 0
        assert ctx.total_files > 0

        # Markdown rendering should work
        md = ctx.to_markdown()
        assert len(md) > 100

    @pytest.mark.asyncio
    async def test_scan_empty_directory(self, tmp_path: Path):
        scanner = RepoScanner(tmp_path)
        ctx = await scanner.scan()
        # Should not crash
        assert ctx.total_files == 0
