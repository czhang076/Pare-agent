"""Tests for ``scripts.migrate_v0_sft_metadata``.

The script rewrites legacy ``tier1_pass`` keys to ``has_diff`` across
two JSONL shapes (trajectory + SFT metadata). The invariants we pin:

1. A v0.1.0 trajectory JSONL migrates to v0.1.1 byte-exact semantics.
2. A v0.1.0 SFT metadata JSONL (row-level ``metadata`` dict) migrates.
3. Idempotent: running twice produces the same content as running once.
4. Conflict (both keys, different values) halts with exit code 2.
5. Backup sidecar is created unless ``--no-backup`` is passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.migrate_v0_sft_metadata import main, migrate_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestMigrateTrajectoryJsonl:
    def test_renames_verification_tier1_pass(self, tmp_path: Path):
        path = tmp_path / "arm.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "trajectory_id": "t1",
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "tier2_pass": True,
                    },
                },
                {
                    "trajectory_id": "t2",
                    "verification": {
                        "final_passed": False,
                        "tier1_pass": False,
                        "tier2_pass": False,
                    },
                },
            ],
        )
        counts = migrate_file(path)
        assert counts == {"rewritten": 2, "unchanged": 0, "total": 2}

        rows = _read_jsonl(path)
        for row in rows:
            assert "tier1_pass" not in row["verification"]
            assert "has_diff" in row["verification"]
        assert rows[0]["verification"]["has_diff"] is True
        assert rows[1]["verification"]["has_diff"] is False


class TestMigrateSftMetadataJsonl:
    def test_renames_metadata_tier1_pass(self, tmp_path: Path):
        path = tmp_path / "sft.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {
                        "trajectory_id": "t1",
                        "tier1_pass": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        counts = migrate_file(path)
        assert counts["rewritten"] == 1

        row = _read_jsonl(path)[0]
        assert "tier1_pass" not in row["metadata"]
        assert row["metadata"]["has_diff"] is True


class TestIdempotency:
    def test_second_run_is_noop(self, tmp_path: Path):
        path = tmp_path / "arm.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        migrate_file(path)
        first_content = path.read_text(encoding="utf-8")

        # Running again on the already-migrated file must not change it.
        counts = migrate_file(path)
        assert counts == {"rewritten": 0, "unchanged": 1, "total": 1}
        assert path.read_text(encoding="utf-8") == first_content

    def test_already_migrated_file_has_no_side_effects(self, tmp_path: Path):
        """No backup written, content byte-identical."""
        path = tmp_path / "already_v011.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "has_diff": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        counts = migrate_file(path)
        assert counts["rewritten"] == 0
        backup = path.with_suffix(path.suffix + ".v0_1_0.bak")
        assert not backup.exists()


class TestConflictHandling:
    def test_same_value_duplicate_drops_legacy(self, tmp_path: Path):
        path = tmp_path / "both_same.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "has_diff": True,  # same value; idempotent upgrade
                        "tier2_pass": True,
                    },
                }
            ],
        )
        counts = migrate_file(path)
        assert counts["rewritten"] == 1
        row = _read_jsonl(path)[0]
        assert "tier1_pass" not in row["verification"]
        assert row["verification"]["has_diff"] is True

    def test_conflicting_values_exits_with_code_2(
        self, tmp_path: Path, capsys
    ):
        path = tmp_path / "conflict.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "has_diff": False,  # conflict
                        "tier2_pass": True,
                    },
                }
            ],
        )
        rc = main([str(path)])
        assert rc == 2
        err = capsys.readouterr().err
        assert "[migrate-failed]" in err


class TestBackupBehaviour:
    def test_backup_created_on_rewrite(self, tmp_path: Path):
        path = tmp_path / "arm.jsonl"
        original_bytes = b'{"verification": {"final_passed": true, "tier1_pass": true, "tier2_pass": true}}\n'
        path.write_bytes(original_bytes)

        migrate_file(path)

        backup = path.with_suffix(path.suffix + ".v0_1_0.bak")
        assert backup.exists()
        assert backup.read_bytes() == original_bytes

    def test_no_backup_flag_skips_sidecar(self, tmp_path: Path):
        path = tmp_path / "arm.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        rc = main([str(path), "--no-backup"])
        assert rc == 0
        backup = path.with_suffix(path.suffix + ".v0_1_0.bak")
        assert not backup.exists()


class TestCliSurface:
    def test_multiple_paths_reported_individually(
        self, tmp_path: Path, capsys
    ):
        """The CLI prints one line per input file so users can tell
        which files were rewritten vs. already-migrated."""
        p1 = tmp_path / "legacy.jsonl"
        p2 = tmp_path / "already.jsonl"
        _write_jsonl(
            p1,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        _write_jsonl(
            p2,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "has_diff": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        rc = main([str(p1), str(p2), "--no-backup"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "rewrote" in out  # p1
        assert "already-migrated" in out  # p2

    def test_missing_file_skipped_with_warning(
        self, tmp_path: Path, capsys
    ):
        """Missing path → warn on stderr, continue with the rest, exit 0."""
        p1 = tmp_path / "exists.jsonl"
        missing = tmp_path / "does_not_exist.jsonl"
        _write_jsonl(
            p1,
            [
                {
                    "verification": {
                        "final_passed": True,
                        "tier1_pass": True,
                        "tier2_pass": True,
                    },
                }
            ],
        )
        rc = main([str(missing), str(p1), "--no-backup"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "[migrate-skipped]" in captured.err
