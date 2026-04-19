"""End-to-end smoke tests for the ``pare inspect`` CLI.

R0: only --help is checked (proves entry point wires up cleanly).
Real e2e tests use the fixture trajectories under ``tests/inspector/fixtures/``,
land in W2 Day 4-5.
"""

from __future__ import annotations

from click.testing import CliRunner

from pare.inspector.cli import main


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
