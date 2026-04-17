"""Batch classify trajectories using the v2 pipeline.

Pipeline order (plan.md §3.3):
1. Load trajectories with ToolCallEvent sequences (schema v2)
2. Extract error_signal for each ToolCallEvent
3. Classify Liu et al. core 4 categories (B2.1, B2.2, C1, C2)
4. Detect recovery events (L1/L2/L3) via recovery_detector_v2
5. Assign trajectory-level outcome label

Outputs:
- ``<input>.labels.jsonl`` — per-trajectory classification details
- ``<input>.non_toxic.jsonl`` — filtered trajectories (toxic removed)
- ``<input>.summary.json`` — aggregate statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pare.trajectory.classifier_liu import (
    LiuClassification,
    OutcomeLabel,
    assign_outcome_label,
    classify_liu_from_record,
)
from pare.trajectory.error_signal_extractor import classify_trajectory_signals
from pare.trajectory.recovery_detector_v2 import (
    RecoveryDetectionResult,
    detect_recovery_events,
)
from pare.trajectory.schema import (
    TrajectoryRecord,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)
from pare.trajectory.schema_v2 import ErrorSignal


class ClassificationError(ValueError):
    """Raised when trajectory classification input/output is invalid."""


# ---------------------------------------------------------------------------
# Per-trajectory classification result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrajectoryClassificationV2:
    """Full classification result for one trajectory."""

    trajectory_id: str
    instance_id: str
    seed: int

    # Liu et al. categories
    liu: LiuClassification

    # Recovery detection
    recovery: RecoveryDetectionResult

    # Outcome label
    outcome: OutcomeLabel

    # Error signal summary
    error_signal_counts: dict[str, int]

    def to_label_row(self) -> dict:
        """Serialize to a JSONL row for labels output."""
        return {
            "trajectory_id": self.trajectory_id,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "outcome": self.outcome.value,
            "liu_categories": self.liu.categories,
            "is_toxic": self.liu.is_toxic,
            "contains_recovery": self.recovery.contains_recovery,
            "highest_recovery_level": (
                self.recovery.highest_level.value
                if self.recovery.highest_level
                else None
            ),
            "recovery_event_count": len(self.recovery.recovery_events),
            "recovery_events": [r.to_dict() for r in self.recovery.recovery_events],
            "error_signal_counts": self.error_signal_counts,
            "liu_detail": self.liu.to_dict(),
        }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    total: int
    outcome_counts: dict[str, int]
    liu_category_counts: dict[str, int]
    recovery_level_counts: dict[str, int]
    error_signal_totals: dict[str, int]
    toxic_count: int
    non_toxic_count: int
    recovery_count: int
    labels_jsonl: str
    non_toxic_jsonl: str

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "outcome_counts": self.outcome_counts,
            "liu_category_counts": self.liu_category_counts,
            "recovery_level_counts": self.recovery_level_counts,
            "error_signal_totals": self.error_signal_totals,
            "toxic_count": self.toxic_count,
            "non_toxic_count": self.non_toxic_count,
            "recovery_count": self.recovery_count,
            "labels_jsonl": self.labels_jsonl,
            "non_toxic_jsonl": self.non_toxic_jsonl,
        }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def classify_one(record: TrajectoryRecord, *, gold_patch: str = "") -> TrajectoryClassificationV2:
    """Run the full v2 classification pipeline on one trajectory.

    Steps:
    1. Extract error signals from ToolCallEvents
    2. Classify Liu et al. core 4 categories
    3. Detect recovery events
    4. Assign outcome label
    """
    events = list(record.tool_call_events)

    # Step 1: Error signal extraction
    if events:
        signals = classify_trajectory_signals(events)
    else:
        signals = []

    # Final diff: prefer the real unified diff captured at agent-finalize
    # time from the git checkpoint (stored in metadata). This is the ground
    # truth for "what did the agent actually change."
    #
    # Legacy records (pre-final_diff plumbing) don't have this field. For
    # those we leave final_diff empty rather than fall back to concatenating
    # file_edit tool result_content — those results are confirmation
    # strings, not unified diffs, so they always yield 0 files/0 hunks and
    # make detect_b11_incomplete_fix return True for every non-empty
    # gold_patch, which is a degenerate signal (see pilot20 data).
    final_diff = record.metadata.get("final_diff", "") if record.metadata else ""

    # Step 2: Liu et al. classification
    liu = classify_liu_from_record(
        record,
        signals,
        final_diff=final_diff,
        gold_patch=gold_patch,
    )

    # Step 3: Recovery detection
    recovery = detect_recovery_events(events, signals)

    # Step 4: Outcome label
    outcome = assign_outcome_label(liu, record.verification, recovery.contains_recovery)

    # Error signal summary
    signal_counts: Counter[str] = Counter()
    for sig in signals:
        signal_counts[sig.value] += 1

    return TrajectoryClassificationV2(
        trajectory_id=record.trajectory_id,
        instance_id=record.instance_id,
        seed=record.seed,
        liu=liu,
        recovery=recovery,
        outcome=outcome,
        error_signal_counts=dict(signal_counts),
    )


def classify_trajectories(
    trajectory_jsonl: Path,
    *,
    tasks_jsonl: Path | None = None,
    labels_jsonl: Path,
    non_toxic_jsonl: Path,
) -> ClassificationSummary:
    """Run full pipeline on all trajectories in a JSONL file."""
    trajectories = load_trajectory_jsonl(trajectory_jsonl)
    if not trajectories:
        raise ClassificationError("No trajectories found.")

    gold_patches: dict[str, str] = {}
    if tasks_jsonl and tasks_jsonl.exists():
        with open(tasks_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    task_data = json.loads(line)
                    if "instance_id" in task_data and "gold_patch" in task_data:
                        gold_patches[task_data["instance_id"]] = task_data["gold_patch"]
                except json.JSONDecodeError:
                    pass

    results: list[TrajectoryClassificationV2] = []
    for record in trajectories:
        gp = gold_patches.get(record.instance_id, "")
        results.append(classify_one(record, gold_patch=gp))

    # Aggregate counts
    outcome_counts: Counter[str] = Counter()
    liu_category_counts: Counter[str] = Counter()
    recovery_level_counts: Counter[str] = Counter()
    error_signal_totals: Counter[str] = Counter()

    rows: list[dict] = []
    non_toxic_records: list[TrajectoryRecord] = []

    for record, result in zip(trajectories, results):
        outcome_counts[result.outcome.value] += 1

        for cat in result.liu.categories:
            liu_category_counts[cat] += 1

        if result.recovery.highest_level:
            recovery_level_counts[result.recovery.highest_level.value] += 1
        else:
            recovery_level_counts["none"] += 1

        for sig_name, count in result.error_signal_counts.items():
            error_signal_totals[sig_name] += count

        if not result.liu.is_toxic:
            non_toxic_records.append(record)

        rows.append(result.to_label_row())

    # Write labels JSONL
    labels_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    # Write non-toxic trajectories
    write_trajectory_jsonl(non_toxic_jsonl, non_toxic_records)

    toxic_count = sum(1 for r in results if r.liu.is_toxic)
    recovery_count = sum(1 for r in results if r.recovery.contains_recovery)

    return ClassificationSummary(
        total=len(trajectories),
        outcome_counts=dict(outcome_counts),
        liu_category_counts=dict(liu_category_counts),
        recovery_level_counts=dict(recovery_level_counts),
        error_signal_totals=dict(error_signal_totals),
        toxic_count=toxic_count,
        non_toxic_count=len(trajectories) - toxic_count,
        recovery_count=recovery_count,
        labels_jsonl=str(labels_jsonl),
        non_toxic_jsonl=str(non_toxic_jsonl),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classify_trajectories",
        description="Classify trajectory JSONL using v2 pipeline (Liu et al. + recovery detection).",
    )
    parser.add_argument("--trajectory-jsonl", required=True, help="Input trajectory JSONL path.")
    parser.add_argument(
        "--tasks-jsonl", default=None,
        help="Optional input tasks JSONL path (to provide gold_patch for B1.1 detection).",
    )
    parser.add_argument(
        "--labels-jsonl", default=None,
        help="Output labels JSONL path (default: <input>.labels.jsonl).",
    )
    parser.add_argument(
        "--summary-json", default=None,
        help="Output summary JSON path (default: <input>.summary.json).",
    )
    parser.add_argument(
        "--non-toxic-jsonl", default=None,
        help="Output non-toxic trajectory JSONL path (default: <input>.non_toxic.jsonl).",
    )
    return parser


def _default_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_suffix(suffix)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trajectory_path = Path(args.trajectory_jsonl)
    tasks_path = Path(args.tasks_jsonl) if args.tasks_jsonl else None
    labels_path = Path(args.labels_jsonl) if args.labels_jsonl else _default_path(trajectory_path, ".labels.jsonl")
    summary_path = Path(args.summary_json) if args.summary_json else _default_path(trajectory_path, ".summary.json")
    non_toxic_path = Path(args.non_toxic_jsonl) if args.non_toxic_jsonl else _default_path(trajectory_path, ".non_toxic.jsonl")

    try:
        summary = classify_trajectories(
            trajectory_path,
            tasks_jsonl=tasks_path,
            labels_jsonl=labels_path,
            non_toxic_jsonl=non_toxic_path,
        )
    except Exception as e:
        print(f"[classify-failed] {e}", file=sys.stderr)
        return 1

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[classify-ok] "
        f"total={summary.total} "
        f"toxic={summary.toxic_count} "
        f"non_toxic={summary.non_toxic_count} "
        f"recovery={summary.recovery_count} "
        f"outcomes={summary.outcome_counts}"
    )
    print(f"  labels: {labels_path}")
    print(f"  summary: {summary_path}")
    print(f"  non_toxic: {non_toxic_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
