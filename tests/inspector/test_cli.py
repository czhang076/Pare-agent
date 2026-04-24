"""End-to-end smoke tests for the ``pare inspect`` CLI."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from pare.inspector.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.jsonl"


def test_pare_inspect_help_works() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "Inspect one or two agent trajectories" in result.output


def test_pare_top_level_help_works() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Pare 2.0" in result.output


def test_classify_only_prints_outcome_and_liu() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["inspect", str(FIXTURE), "--classify-only"],
    )
    assert result.exit_code == 0, result.output
    assert "traj-001" in result.output
    assert "traj-002" in result.output
    assert "VERIFIED_ONE_SHOT" in result.output
    assert "TOXIC" in result.output
    assert "B2.2" in result.output


def test_classify_only_requires_one_path() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", "--classify-only"])
    assert result.exit_code != 0
    assert "expects exactly one JSONL path" in result.output
