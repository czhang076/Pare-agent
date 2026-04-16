"""SFT exporter: trajectory records -> OpenAI messages JSONL.

This exporter supports two sources for the final conversation:
1) Raw OpenAI messages embedded in trajectory metadata (preferred)
2) Deterministic reconstruction from step attempts (fallback)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pare.trajectory.schema import TrajectoryRecord, load_trajectory_jsonl

_DEFAULT_SYSTEM_PROMPT = "You are Pare, an expert coding agent."


class SFTExportError(ValueError):
    """Raised when a trajectory cannot be exported to SFT format."""


@dataclass(frozen=True, slots=True)
class SFTExporterConfig:
    """Configuration for trajectory -> OpenAI message export."""

    include_metadata: bool = True
    require_raw_messages: bool = False
    default_system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    raw_messages_metadata_keys: tuple[str, ...] = (
        "openai_messages_json",
        "messages_json",
        "raw_messages_json",
    )


class SFTExporter:
    """Export trajectory records into OpenAI-compatible message examples."""

    def __init__(self, config: SFTExporterConfig | None = None) -> None:
        self.config = config or SFTExporterConfig()

    def export_record(self, trajectory: TrajectoryRecord) -> dict[str, Any]:
        """Export one trajectory to a JSON-serializable SFT example."""
        raw = self._extract_raw_messages(trajectory)
        if raw is not None:
            messages = raw
            source = "metadata_raw_messages"
        else:
            if self.config.require_raw_messages:
                raise SFTExportError(
                    "Raw messages required but none found in trajectory metadata."
                )
            messages = self._reconstruct_messages(trajectory)
            source = "reconstructed_attempts"

        sample: dict[str, Any] = {
            "messages": messages,
        }

        if self.config.include_metadata:
            sample["metadata"] = {
                "trajectory_id": trajectory.trajectory_id,
                "instance_id": trajectory.instance_id,
                "model": trajectory.model,
                "seed": trajectory.seed,
                "llm_claimed_success": trajectory.llm_claimed_success,
                "final_passed": trajectory.verification.final_passed,
                "tier1_pass": trajectory.verification.tier1_pass,
                "tier2_pass": trajectory.verification.tier2_pass,
                "token_usage": trajectory.token_usage.to_dict(),
                "export_source": source,
            }

        return sample

    def export_many(self, trajectories: Iterable[TrajectoryRecord]) -> list[dict[str, Any]]:
        """Export multiple trajectories."""
        return [self.export_record(record) for record in trajectories]

    def _extract_raw_messages(self, trajectory: TrajectoryRecord) -> list[dict[str, Any]] | None:
        for key in self.config.raw_messages_metadata_keys:
            payload = trajectory.metadata.get(key)
            if not payload:
                continue

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as e:
                raise SFTExportError(
                    f"Invalid JSON in metadata field '{key}': {e}"
                ) from e

            if not isinstance(parsed, list):
                raise SFTExportError(
                    f"Metadata field '{key}' must be a JSON list of messages."
                )

            return self._validate_openai_messages(parsed, context=f"metadata.{key}")

        return None

    def _reconstruct_messages(self, trajectory: TrajectoryRecord) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": trajectory.metadata.get(
                    "system_prompt",
                    self.config.default_system_prompt,
                ),
            },
            {
                "role": "user",
                "content": trajectory.task,
            },
        ]

        for attempt in trajectory.attempts:
            intro = (
                f"Step {attempt.step_number}, attempt {attempt.attempt_number}: "
                f"{attempt.goal}"
            )

            tool_calls: list[dict[str, Any]] = []
            for idx, tool_name in enumerate(attempt.tool_names, start=1):
                tool_calls.append(
                    {
                        "id": self._tool_call_id(attempt.step_number, attempt.attempt_number, idx),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": "{}",
                        },
                    }
                )

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": intro,
                        "tool_calls": tool_calls,
                    }
                )

                for idx, tool_name in enumerate(attempt.tool_names, start=1):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": self._tool_call_id(
                                attempt.step_number,
                                attempt.attempt_number,
                                idx,
                            ),
                            "content": (
                                f"Tool '{tool_name}' output is unavailable in the "
                                "trajectory schema; this is a deterministic placeholder."
                            ),
                        }
                    )
            else:
                messages.append({"role": "assistant", "content": intro})

            status_msg = f"Attempt status: {attempt.status}."
            if attempt.failure_reason:
                status_msg += f" Failure reason: {attempt.failure_reason}."
            if attempt.target_files:
                status_msg += f" Target files: {', '.join(attempt.target_files)}."

            messages.append({"role": "assistant", "content": status_msg})

        messages.append(
            {
                "role": "assistant",
                "content": self._build_final_summary(trajectory),
            }
        )

        return self._validate_openai_messages(messages, context="reconstructed")

    @staticmethod
    def _tool_call_id(step_number: int, attempt_number: int, index: int) -> str:
        return f"tc_s{step_number}_a{attempt_number}_{index}"

    @staticmethod
    def _build_final_summary(trajectory: TrajectoryRecord) -> str:
        verification = trajectory.verification
        usage = trajectory.token_usage
        return (
            "Final summary: "
            f"llm_claimed_success={trajectory.llm_claimed_success}; "
            f"final_passed={verification.final_passed}; "
            f"tier1_pass={verification.tier1_pass}; "
            f"tier2_pass={verification.tier2_pass}; "
            f"input_tokens={usage.input_tokens}; "
            f"output_tokens={usage.output_tokens}; "
            f"total_tokens={usage.total_tokens}."
        )

    def _validate_openai_messages(
        self,
        messages: list[Any],
        *,
        context: str,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        allowed_roles = {"system", "user", "assistant", "tool"}

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise SFTExportError(f"{context}[{i}] must be an object")

            role = msg.get("role")
            content = msg.get("content")

            if role not in allowed_roles:
                raise SFTExportError(
                    f"{context}[{i}].role must be one of {sorted(allowed_roles)}"
                )
            if not isinstance(content, str):
                raise SFTExportError(f"{context}[{i}].content must be a string")

            out: dict[str, Any] = {
                "role": role,
                "content": content,
            }

            tool_call_id = msg.get("tool_call_id")
            if tool_call_id is not None:
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise SFTExportError(f"{context}[{i}].tool_call_id must be non-empty str")
                out["tool_call_id"] = tool_call_id

            tool_calls = msg.get("tool_calls")
            if tool_calls is not None:
                out["tool_calls"] = self._validate_tool_calls(
                    tool_calls,
                    context=f"{context}[{i}].tool_calls",
                )

            normalized.append(out)

        return normalized

    def _validate_tool_calls(self, tool_calls: Any, *, context: str) -> list[dict[str, Any]]:
        if not isinstance(tool_calls, list):
            raise SFTExportError(f"{context} must be a list")

        normalized: list[dict[str, Any]] = []
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                raise SFTExportError(f"{context}[{i}] must be an object")

            tc_id = tc.get("id")
            tc_type = tc.get("type")
            function = tc.get("function")

            if not isinstance(tc_id, str) or not tc_id:
                raise SFTExportError(f"{context}[{i}].id must be non-empty str")
            if tc_type != "function":
                raise SFTExportError(f"{context}[{i}].type must be 'function'")
            if not isinstance(function, dict):
                raise SFTExportError(f"{context}[{i}].function must be an object")

            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name:
                raise SFTExportError(f"{context}[{i}].function.name must be non-empty str")
            if not isinstance(arguments, str):
                raise SFTExportError(f"{context}[{i}].function.arguments must be str JSON")

            normalized.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

        return normalized


def write_sft_jsonl(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    """Write exported SFT samples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")


def export_trajectory_jsonl_to_sft(
    input_path: Path,
    output_path: Path,
    *,
    config: SFTExporterConfig | None = None,
) -> int:
    """Load trajectory JSONL and export to OpenAI-message SFT JSONL.

    Returns:
        Number of exported samples.
    """
    trajectories = load_trajectory_jsonl(input_path)
    exporter = SFTExporter(config=config)
    samples = exporter.export_many(trajectories)
    write_sft_jsonl(output_path, samples)
    return len(samples)
