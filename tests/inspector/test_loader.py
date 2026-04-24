"""Tests for ``pare.inspector.loader.load_jsonl``.

The fixture :file:`tests/inspector/fixtures/minimal.jsonl` is regenerated
by :file:`scripts/gen_minimal_fixture.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pare.inspector.loader import load_jsonl
from pare.trajectory.schema import SchemaValidationError, TrajectoryRecord

FIXTURE = Path(__file__).parent / "fixtures" / "minimal.jsonl"


def test_load_jsonl_minimal() -> None:
    records = load_jsonl(FIXTURE)
    assert len(records) == 2
    assert all(isinstance(r, TrajectoryRecord) for r in records)
    assert [r.trajectory_id for r in records] == ["traj-001", "traj-002"]


def test_load_jsonl_skips_blank_and_comment(tmp_path: Path) -> None:
    valid_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    f = tmp_path / "with_noise.jsonl"
    f.write_text(f"\n# a hand-written header comment\n{valid_line}\n", encoding="utf-8")

    records = load_jsonl(f)
    assert len(records) == 1
    assert records[0].trajectory_id == "traj-001"


def test_load_jsonl_propagates_schema_error_with_lineno(tmp_path: Path) -> None:
    valid_line = FIXTURE.read_text(encoding="utf-8").splitlines()[0]
    f = tmp_path / "broken.jsonl"
    f.write_text(f"{valid_line}\nnot valid json\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError) as excinfo:
        load_jsonl(f)
    assert ":2:" in str(excinfo.value)
