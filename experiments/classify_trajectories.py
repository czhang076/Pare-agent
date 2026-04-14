"""Batch classify trajectories and emit label statistics.

This script applies deterministic trajectory labels, writes per-trajectory
classification JSONL, optional non-toxic filtered trajectories, and summary JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pare.trajectory.classifier import TrajectoryClassifier, TrajectoryLabel
from pare.trajectory.schema import TrajectoryRecord, load_trajectory_jsonl, write_trajectory_jsonl


class ClassificationError(ValueError):
    """Raised when trajectory classification input/output is invalid."""


@dataclass(frozen=True, slots=True)
class ClassificationSummary:
    total: int
    primary_counts: dict[str, int]
    verification_counts: dict[str, int]
    recovery_level_counts: dict[str, int]
    toxic_count: int
    non_toxic_count: int
    labels_jsonl: Path
    non_toxic_jsonl: Path

    def to_dict(self) -> dict:
        data = asdict(self)
        data["labels_jsonl"] = str(self.labels_jsonl)
        data["non_toxic_jsonl"] = str(self.non_toxic_jsonl)
        return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classify_trajectories",
        description="Classify trajectory JSONL using deterministic trajectory labels.",
    )
    parser.add_argument("--trajectory-jsonl", required=True, help="Input trajectory JSONL path.")
    parser.add_argument(
        "--labels-jsonl",
        default=None,
        help="Output labels JSONL path (default: <input>.labels.jsonl).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Output summary JSON path (default: <input>.summary.json).",
    )
    parser.add_argument(
        "--non-toxic-jsonl",
        default=None,
        help="Output non-toxic trajectory JSONL path (default: <input>.non_toxic.jsonl).",
    )
    return parser


def _default_labels_path(input_path: Path) -> Path:
    return input_path.with_suffix(".labels.jsonl")


def _default_summary_path(input_path: Path) -> Path:
    return input_path.with_suffix(".summary.json")


def _default_non_toxic_path(input_path: Path) -> Path:
    return input_path.with_suffix(".non_toxic.jsonl")


def classify_trajectories(
    trajectory_jsonl: Path,
    *,
    labels_jsonl: Path,
    non_toxic_jsonl: Path,
) -> ClassificationSummary:
    trajectories = load_trajectory_jsonl(trajectory_jsonl)
    if not trajectories:
        raise ClassificationError("No trajectories found.")

    classifier = TrajectoryClassifier()
    results = classifier.classify_many(trajectories)

    primary_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    recovery_level_counts: Counter[str] = Counter()

    rows: list[dict] = []
    non_toxic_records: list[TrajectoryRecord] = []

    for trajectory, result in zip(trajectories, results):
        primary = result.primary_label.value
        verification = result.verification_label.value
        recovery_level = result.recovery_level.value if result.recovery_level else ""

        primary_counts[primary] += 1
        verification_counts[verification] += 1
        recovery_level_counts[recovery_level or "none"] += 1

        if result.primary_label != TrajectoryLabel.TOXIC:
            non_toxic_records.append(trajectory)

        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "instance_id": trajectory.instance_id,
                "seed": trajectory.seed,
                "primary_label": primary,
                "verification_label": verification,
                "recovery_level": recovery_level,
                "recovery_event_count": len(result.recovery_events),
                "reasons": list(result.reasons),
            }
        )

    labels_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    write_trajectory_jsonl(non_toxic_jsonl, non_toxic_records)

    toxic_count = int(primary_counts.get(TrajectoryLabel.TOXIC.value, 0))
    summary = ClassificationSummary(
        total=len(trajectories),
        primary_counts=dict(primary_counts),
        verification_counts=dict(verification_counts),
        recovery_level_counts=dict(recovery_level_counts),
        toxic_count=toxic_count,
        non_toxic_count=len(trajectories) - toxic_count,
        labels_jsonl=labels_jsonl,
        non_toxic_jsonl=non_toxic_jsonl,
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trajectory_path = Path(args.trajectory_jsonl)
    labels_path = Path(args.labels_jsonl) if args.labels_jsonl else _default_labels_path(trajectory_path)
    summary_path = Path(args.summary_json) if args.summary_json else _default_summary_path(trajectory_path)
    non_toxic_path = Path(args.non_toxic_jsonl) if args.non_toxic_jsonl else _default_non_toxic_path(trajectory_path)

    try:
        summary = classify_trajectories(
            trajectory_path,
            labels_jsonl=labels_path,
            non_toxic_jsonl=non_toxic_path,
        )
    except Exception as e:
        print(f"[classify-failed] {e}", file=sys.stderr)
        return 1

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "[classify-ok] "
        f"total={summary.total} "
        f"toxic={summary.toxic_count} "
        f"non_toxic={summary.non_toxic_count} "
        f"labels={summary.labels_jsonl} "
        f"summary={summary_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
