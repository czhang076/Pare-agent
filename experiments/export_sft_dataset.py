"""CLI wrapper around ``pare.trajectory.sft_export``.

Converts a classifier-labeled trajectory JSONL into an OpenAI-format SFT
dataset ready for ``openai.fine_tuning.jobs.create`` (or an equivalent
HF trainer wrapper that consumes the same chat schema).

Typical invocations::

    # Recovery-only (the "self-correction demonstrations" bucket)
    python -m experiments.export_sft_dataset \\
        --trajectory-jsonl data/sympy20/arm2_prepasses.jsonl \\
        --output-jsonl     data/sft/recovery_only.jsonl \\
        --include-recovery-only

    # Successful runs (both one-shot and with-recovery) for a baseline mix
    python -m experiments.export_sft_dataset \\
        --trajectory-jsonl data/sympy20/arm3_full.jsonl \\
        --output-jsonl     data/sft/all_verified.jsonl \\
        --include-outcome verified_one_shot \\
        --include-outcome verified_with_recovery

A report JSON next to the output records drop counts and filter provenance,
which is the artefact to attach to any dataset-card / paper-table claim
about "what's in the training mix."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pare.trajectory.sft_export import export_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_sft_dataset",
        description=(
            "Convert a labeled trajectory JSONL into an OpenAI-format "
            "SFT dataset JSONL."
        ),
    )
    parser.add_argument(
        "--trajectory-jsonl",
        required=True,
        help="Input trajectory JSONL (from experiments.generate_trajectories).",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output SFT JSONL path. Parent directories are created.",
    )
    parser.add_argument(
        "--labels-jsonl",
        default=None,
        help=(
            "Classifier labels JSONL. Defaults to "
            "<trajectory-jsonl>.labels.jsonl if present."
        ),
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help=(
            "Optional path for the export report (counts + filter "
            "provenance). Defaults to <output-jsonl>.report.json."
        ),
    )
    parser.add_argument(
        "--include-outcome",
        dest="include_outcomes",
        action="append",
        default=None,
        help=(
            "Only include trajectories with this classifier outcome "
            "(repeatable, e.g. --include-outcome verified_one_shot "
            "--include-outcome verified_with_recovery)."
        ),
    )
    parser.add_argument(
        "--include-recovery-only",
        action="store_true",
        help="Keep only trajectories with contains_recovery=True.",
    )
    parser.add_argument(
        "--keep-toxic",
        action="store_true",
        help=(
            "Do NOT drop toxic-labeled rows. Default is to drop them; "
            "use this flag only for sanity / diagnostic exports."
        ),
    )
    parser.add_argument(
        "--keep-empty-events",
        action="store_true",
        help=(
            "Do NOT drop rows with zero tool_call_events. Default is "
            "to drop them because they're nothing to train on."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help=(
            "System prompt text to prepend to every conversation. "
            "Pass the same text used at trajectory generation time for "
            "format parity with the student."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help=(
            "Read --system-prompt from this file instead of passing it "
            "inline. Takes precedence over --system-prompt if both set."
        ),
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional cap on output rows (applied after filtering).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    traj_path = Path(args.trajectory_jsonl)
    out_path = Path(args.output_jsonl)
    labels_path = Path(args.labels_jsonl) if args.labels_jsonl else None
    report_path = (
        Path(args.report_json)
        if args.report_json
        else Path(str(out_path) + ".report.json")
    )

    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    try:
        report = export_dataset(
            traj_path,
            out_path,
            labels_jsonl=labels_path,
            include_outcomes=args.include_outcomes,
            include_recovery_only=args.include_recovery_only,
            drop_toxic=not args.keep_toxic,
            drop_empty_events=not args.keep_empty_events,
            system_prompt=system_prompt,
            max_trajectories=args.max_trajectories,
        )
    except Exception as e:
        print(f"[sft-export-failed] {e}", file=sys.stderr)
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[sft-export-ok] "
        f"loaded={report.trajectories_loaded} "
        f"written={report.rows_written} "
        f"drops={report.drop_reasons}"
    )
    print(f"  output: {out_path}")
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
