"""Migrate v0.1.0 JSONL artifacts to v0.1.1 naming (tier1_pass → has_diff).

Rationale
---------

In v0.1.0, ``VerificationResult.tier1_pass`` was bound to
``bool(final_diff and final_diff.strip())`` — literally "the agent
wrote a non-empty diff", not any form of tier-1 verifier. v0.1.1
renames the field to ``has_diff`` end-to-end.

The schema reader accepts either key for back-compat, so already-written
JSONL *loads* fine without migration. But any downstream consumer that
reads the metadata dict directly (composition samplers, dataset-card
queries, ad-hoc ``jq`` filters, paper-table scripts) still sees the
legacy key name. This script rewrites the files in place so every
artifact on disk uses the new name.

Scope
-----

Two file shapes are handled:

1. **Trajectory JSONL** (one ``TrajectoryRecord.to_dict()`` per line)
   — rewrites ``row["verification"]["tier1_pass"]`` → ``has_diff``.

2. **SFT metadata JSONL** (one OpenAI chat row per line, with a
   ``metadata`` dict) — rewrites ``row["metadata"]["tier1_pass"]`` →
   ``has_diff``.

The script is idempotent: rows that already use ``has_diff`` pass
through untouched. Rows that have both keys with the same value drop
the legacy one. Rows with conflicting values raise loudly — this is
the same invariant enforced by ``VerificationResult.from_dict``.

Usage
-----

::

    python -m scripts.migrate_v0_sft_metadata \\
        data/pilot/arm1_baseline.jsonl \\
        data/pilot/arm2_prepasses.jsonl \\
        data/pilot/arm3_full.jsonl \\
        data/sft/recovery_only.jsonl \\
        data/sft/all_verified.jsonl

Backup: the original file is written to ``<path>.v0_1_0.bak`` before
rewrite. Deletion of backups is left to the operator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


LEGACY_KEY = "tier1_pass"
NEW_KEY = "has_diff"


class MigrationConflict(ValueError):
    """Both legacy and new keys present with conflicting values."""


def _migrate_dict_in_place(d: dict[str, Any], context: str) -> bool:
    """Rewrite ``d[LEGACY_KEY]`` -> ``d[NEW_KEY]``; return True iff changed.

    Invariants:
    - both-present-same-value  → drop legacy, keep new, changed=True
    - both-present-diff-value  → raise MigrationConflict
    - legacy-only              → rename, changed=True
    - new-only or neither      → no-op, changed=False
    """
    has_legacy = LEGACY_KEY in d
    has_new = NEW_KEY in d
    if has_legacy and has_new:
        if d[LEGACY_KEY] != d[NEW_KEY]:
            raise MigrationConflict(
                f"{context}: both {LEGACY_KEY!r} and {NEW_KEY!r} present "
                f"with different values ({d[LEGACY_KEY]!r} vs {d[NEW_KEY]!r})"
            )
        del d[LEGACY_KEY]
        return True
    if has_legacy:
        d[NEW_KEY] = d.pop(LEGACY_KEY)
        return True
    return False


def _migrate_row(row: dict[str, Any], context: str) -> bool:
    """Return True if any key was migrated in this row.

    Looks in both shapes so we can run one script against all files.
    """
    changed = False
    if isinstance(row.get("verification"), dict):
        changed |= _migrate_dict_in_place(
            row["verification"], f"{context}.verification"
        )
    if isinstance(row.get("metadata"), dict):
        changed |= _migrate_dict_in_place(
            row["metadata"], f"{context}.metadata"
        )
    return changed


def migrate_file(path: Path, *, backup: bool = True) -> dict[str, int]:
    """Migrate one JSONL file in place. Return counts for reporting.

    If ``backup`` is True, a ``<path>.v0_1_0.bak`` sidecar is written
    before the rewrite. Callers that run this from inside a git
    worktree may pass ``backup=False`` and rely on ``git`` for history.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    rewritten = 0
    unchanged = 0
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if _migrate_row(row, f"{path.name}:line{lineno}"):
                rewritten += 1
            else:
                unchanged += 1
            rows.append(row)

    if rewritten == 0:
        return {"rewritten": 0, "unchanged": unchanged, "total": len(rows)}

    if backup:
        backup_path = path.with_suffix(path.suffix + ".v0_1_0.bak")
        shutil.copy2(path, backup_path)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)

    return {"rewritten": rewritten, "unchanged": unchanged, "total": len(rows)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_v0_sft_metadata",
        description=(
            "Rewrite ``tier1_pass`` -> ``has_diff`` in trajectory and "
            "SFT metadata JSONL files. See module docstring for scope."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSONL files to migrate in place.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Skip the ``<path>.v0_1_0.bak`` sidecar. Only safe if you're "
            "running inside a git worktree with clean history."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    total_rewritten = 0
    total_unchanged = 0
    for path in args.paths:
        try:
            counts = migrate_file(path, backup=not args.no_backup)
        except MigrationConflict as e:
            print(f"[migrate-failed] {path}: {e}", file=sys.stderr)
            return 2
        except FileNotFoundError:
            print(f"[migrate-skipped] {path}: file not found", file=sys.stderr)
            continue
        tag = "rewrote" if counts["rewritten"] else "already-migrated"
        print(
            f"[migrate-{tag}] {path} "
            f"rewritten={counts['rewritten']} "
            f"unchanged={counts['unchanged']} "
            f"total={counts['total']}"
        )
        total_rewritten += counts["rewritten"]
        total_unchanged += counts["unchanged"]

    print(
        f"[migrate-done] total_rewritten={total_rewritten} "
        f"total_unchanged={total_unchanged}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
