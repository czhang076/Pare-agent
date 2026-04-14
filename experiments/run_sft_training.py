"""LoRA pre-training smoke script.

This script is intentionally lightweight and deterministic. It does not
train a model; it verifies that exported SFT data can be loaded and
batched by a training pipeline.

One-click usage:
    .venv\\Scripts\\python.exe experiments/run_sft_training.py \\
        --trajectory-jsonl data/trajectories.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pare.export.sft_exporter import export_trajectory_jsonl_to_sft


class SFTSmokeError(ValueError):
    """Raised when SFT data fails smoke validation."""


@dataclass(frozen=True, slots=True)
class SFTSmokeReport:
    """Summary of one smoke run."""

    exported_samples: int
    loaded_samples: int
    batch_count: int
    batch_size: int
    min_chars: int
    max_chars: int
    avg_chars: float
    output_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sft_training",
        description=(
            "Smoke-check that trajectory export output can be read by "
            "a LoRA training data pipeline."
        ),
    )
    parser.add_argument(
        "--trajectory-jsonl",
        required=True,
        help="Input trajectory JSONL path.",
    )
    parser.add_argument(
        "--sft-jsonl",
        default=None,
        help="Output SFT JSONL path (default: <trajectory>.sft.jsonl).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for smoke collation.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum expected sample count after export.",
    )
    return parser


def load_and_validate_sft_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load SFT JSONL and validate OpenAI message structure."""
    if not path.exists():
        raise SFTSmokeError(f"SFT file does not exist: {path}")

    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                sample = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SFTSmokeError(f"{path}:{line_no}: invalid JSON: {e}") from e

            if not isinstance(sample, dict):
                raise SFTSmokeError(f"{path}:{line_no}: sample must be an object")

            messages = sample.get("messages")
            if not isinstance(messages, list) or not messages:
                raise SFTSmokeError(f"{path}:{line_no}: messages must be a non-empty list")

            for idx, msg in enumerate(messages):
                _validate_message(msg, path=path, line_no=line_no, index=idx)

            samples.append(sample)

    return samples


def run_lora_smoke(
    trajectory_jsonl: Path,
    *,
    sft_jsonl: Path | None = None,
    batch_size: int = 4,
    min_samples: int = 1,
) -> SFTSmokeReport:
    """Export trajectories and validate data-loading readiness for LoRA."""
    if batch_size <= 0:
        raise SFTSmokeError("batch_size must be > 0")
    if min_samples <= 0:
        raise SFTSmokeError("min_samples must be > 0")

    if sft_jsonl is None:
        sft_jsonl = trajectory_jsonl.with_suffix(".sft.jsonl")

    exported_samples = export_trajectory_jsonl_to_sft(trajectory_jsonl, sft_jsonl)
    samples = load_and_validate_sft_jsonl(sft_jsonl)

    if exported_samples < min_samples:
        raise SFTSmokeError(
            f"Exported samples {exported_samples} below minimum {min_samples}."
        )
    if len(samples) < min_samples:
        raise SFTSmokeError(
            f"Loaded samples {len(samples)} below minimum {min_samples}."
        )

    training_texts = [sample_to_training_text(sample) for sample in samples]
    lengths = [len(text) for text in training_texts]
    if not lengths or min(lengths) <= 0:
        raise SFTSmokeError("At least one training text is empty.")

    batches = [
        training_texts[i:i + batch_size]
        for i in range(0, len(training_texts), batch_size)
    ]

    return SFTSmokeReport(
        exported_samples=exported_samples,
        loaded_samples=len(samples),
        batch_count=len(batches),
        batch_size=batch_size,
        min_chars=min(lengths),
        max_chars=max(lengths),
        avg_chars=sum(lengths) / len(lengths),
        output_path=sft_jsonl,
    )


def sample_to_training_text(sample: dict[str, Any]) -> str:
    """Convert one OpenAI-message sample to a plain training text block."""
    messages = sample.get("messages")
    assert isinstance(messages, list)

    chunks: list[str] = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        chunks.append(f"[{role}]\n{content}")

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        name = fn.get("name", "")
                        args = fn.get("arguments", "")
                        chunks.append(f"[TOOL_CALL:{name}]\n{args}")

        tool_call_id = msg.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            chunks.append(f"[TOOL_CALL_ID]\n{tool_call_id}")

    return "\n\n".join(chunks)


def _validate_message(
    msg: Any,
    *,
    path: Path,
    line_no: int,
    index: int,
) -> None:
    if not isinstance(msg, dict):
        raise SFTSmokeError(f"{path}:{line_no}: messages[{index}] must be an object")

    role = msg.get("role")
    content = msg.get("content")
    allowed_roles = {"system", "user", "assistant", "tool"}

    if role not in allowed_roles:
        raise SFTSmokeError(
            f"{path}:{line_no}: messages[{index}].role invalid: {role}"
        )
    if not isinstance(content, str):
        raise SFTSmokeError(
            f"{path}:{line_no}: messages[{index}].content must be str"
        )

    tool_call_id = msg.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise SFTSmokeError(
            f"{path}:{line_no}: messages[{index}].tool_call_id must be str"
        )

    tool_calls = msg.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise SFTSmokeError(
                f"{path}:{line_no}: messages[{index}].tool_calls must be list"
            )
        for tc_idx, tc in enumerate(tool_calls):
            _validate_tool_call(tc, path=path, line_no=line_no, index=index, tc_idx=tc_idx)


def _validate_tool_call(
    tool_call: Any,
    *,
    path: Path,
    line_no: int,
    index: int,
    tc_idx: int,
) -> None:
    prefix = f"{path}:{line_no}: messages[{index}].tool_calls[{tc_idx}]"
    if not isinstance(tool_call, dict):
        raise SFTSmokeError(f"{prefix} must be object")

    if tool_call.get("type") != "function":
        raise SFTSmokeError(f"{prefix}.type must be 'function'")

    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise SFTSmokeError(f"{prefix}.id must be non-empty str")

    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        raise SFTSmokeError(f"{prefix}.function must be object")

    name = fn.get("name")
    arguments = fn.get("arguments")
    if not isinstance(name, str) or not name:
        raise SFTSmokeError(f"{prefix}.function.name must be non-empty str")
    if not isinstance(arguments, str):
        raise SFTSmokeError(f"{prefix}.function.arguments must be str")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trajectory_jsonl = Path(args.trajectory_jsonl)
    sft_jsonl = Path(args.sft_jsonl) if args.sft_jsonl else None

    try:
        report = run_lora_smoke(
            trajectory_jsonl,
            sft_jsonl=sft_jsonl,
            batch_size=args.batch_size,
            min_samples=args.min_samples,
        )
    except Exception as e:
        print(f"[smoke-failed] {e}", file=sys.stderr)
        return 1

    print(
        "[smoke-ok] "
        f"exported={report.exported_samples} "
        f"loaded={report.loaded_samples} "
        f"batches={report.batch_count} "
        f"batch_size={report.batch_size} "
        f"chars(min/avg/max)={report.min_chars}/{report.avg_chars:.1f}/{report.max_chars} "
        f"sft={report.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
