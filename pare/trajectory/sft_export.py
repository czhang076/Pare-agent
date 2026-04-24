"""Export TrajectoryRecords to OpenAI-format SFT training rows.

Pare's research question is: *does SFT on trajectories containing tool-call-
level error-correction patterns transfer self-correction capability to a
student model?* That question reduces to a very concrete artefact: a
``dataset.jsonl`` where each line is an OpenAI-format chat conversation
ready to be fed into ``openai.fine_tuning.jobs.create`` or an equivalent
HF trainer wrapper.

This module is that last mile. It does two things and nothing else:

1. ``export_trajectory_to_sft(record, label=None)`` — convert one
   ``TrajectoryRecord`` into a single ``{"messages": [...], "metadata": {...}}``
   row. Pure function, no I/O.
2. ``export_dataset(trajectory_jsonl, labels_jsonl, output_jsonl, ...)`` —
   batch driver with outcome/recovery filtering. Writes a report dict so
   we can see at a glance how many rows were kept vs dropped and why.

Message-shape reconstruction
----------------------------

``TrajectoryRecord`` does **not** persist the assistant's free-text
reasoning between tool calls — the flat-ReAct loop stores only the
structured ``tool_call_events`` sequence (turn_id, call_index_in_turn,
tool_name, params, result_content). That's fine for SFT on tool use:
we emit assistant messages with empty ``content`` and a populated
``tool_calls`` array, which is exactly the format OpenAI's fine-tuning
API expects for "train on the tool-call decisions, not on a paraphrase
of the reasoning."

The synthesized message sequence per trajectory is::

    [
      {"role": "system",    "content": <task / orient / plan>},
      {"role": "user",      "content": <original task>},
      # turn 0 (one assistant response with k tool_calls):
      {"role": "assistant", "content": "", "tool_calls": [
          {"id": "call_<turn>_<idx>", "type": "function",
           "function": {"name": ..., "arguments": json_str}}, ...
      ]},
      {"role": "tool", "tool_call_id": "call_<turn>_<idx>", "content": ...},
      ...  # one tool message per call in that turn, in order
      # turn 1, turn 2, ...
    ]

Because the schema lacks a wire-format ``tool_call_id``, we synthesize
stable deterministic ids from ``(turn_id, call_index_in_turn)``. Anything
reading the output can assume ids are unique *within* a conversation
(which is all OpenAI's format requires).

Filtering
---------

The B2.1 / recovery hypothesis is that trajectories containing L2+
recovery events are the ones that teach self-correction. ``export_dataset``
therefore accepts:

- ``include_outcomes``   — restrict to e.g. ``{"verified_with_recovery"}``
  (the canonical "good recovery" bucket) or
  ``{"verified_one_shot", "verified_with_recovery"}`` for "successful".
- ``include_recovery_only`` — filter to ``contains_recovery=True`` rows.
- ``drop_toxic`` — default True; toxic rows (classifier flagged C1/C2
  signatures) are always excluded from student training corpora.

Callers combining flags get the AND of all constraints.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pare.trajectory.schema import TrajectoryRecord, load_trajectory_jsonl


# ---------------------------------------------------------------------------
# Public return types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SFTRow:
    """One exported row. ``to_jsonl_dict`` is the canonical on-disk shape."""

    messages: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_jsonl_dict(self) -> dict[str, Any]:
        return {"messages": list(self.messages), "metadata": dict(self.metadata)}


@dataclass(frozen=True, slots=True)
class ExportReport:
    """Outcome of a batch export, intended to be written alongside the JSONL.

    ``drop_reasons`` gives a loud per-bucket account so we never wonder
    "where did those 40 rows go?" after a filter tweak.
    """

    trajectories_loaded: int
    rows_written: int
    drop_reasons: dict[str, int] = field(default_factory=dict)
    output_path: str = ""
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories_loaded": self.trajectories_loaded,
            "rows_written": self.rows_written,
            "drop_reasons": dict(self.drop_reasons),
            "output_path": self.output_path,
            "filters": dict(self.filters),
        }


# ---------------------------------------------------------------------------
# Single-trajectory conversion (pure)
# ---------------------------------------------------------------------------


def _synthesize_tool_call_id(turn_id: int, call_index_in_turn: int) -> str:
    """Deterministic id unique within a conversation.

    The wire-format id isn't persisted in ``TrajectoryRecord`` (the loop
    uses the provider's id at runtime, but it's not worth retaining post
    hoc), so we rebuild one from the temporal key. Stable across reruns
    of the exporter — that matters for dataset-diff reproducibility.
    """
    return f"call_{turn_id}_{call_index_in_turn}"


def _format_tool_arguments(params: dict[str, Any]) -> str:
    """OpenAI tool_calls.arguments is a JSON *string*, not an object."""
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def _group_events_by_turn(
    events: Iterable[Any],
) -> list[list[Any]]:
    """Return events grouped by ``turn_id``, preserving turn order.

    We sort by ``global_index`` defensively even though the recording
    loop today guarantees this — any future resampler / deduper / hand
    edit of the JSONL that reorders events would otherwise split one
    assistant turn into two SFT rows with partial tool_calls, training
    the student on a malformed conversation shape. Sort is cheap
    (events per trajectory are O(100)) and makes the invariant
    machine-enforced.
    """
    sorted_events = sorted(events, key=lambda e: e.global_index)
    groups: list[list[Any]] = []
    current_turn: int | None = None
    for evt in sorted_events:
        if current_turn is None or evt.turn_id != current_turn:
            groups.append([evt])
            current_turn = evt.turn_id
        else:
            groups[-1].append(evt)
    return groups


def export_trajectory_to_sft(
    record: TrajectoryRecord,
    *,
    label: dict[str, Any] | None = None,
    system_prompt: str = "",
) -> SFTRow:
    """Convert one trajectory into a single OpenAI-format SFT row.

    Args:
        record: The trajectory to convert.
        label: Optional classifier label row (as produced by
            ``classify_trajectories``). If provided, its ``outcome``,
            ``contains_recovery``, ``highest_recovery_level``, and
            ``liu_categories`` are attached to ``metadata``, which the
            sampler downstream can key on when building training mixes.
        system_prompt: System prompt to prepend. Defaults to empty —
            caller is expected to pass the same system text used at
            generation time for format parity (we intentionally don't
            hard-code a template because the orient/planner pre-passes
            mutate it at runtime, and the record doesn't persist that).

    Returns:
        ``SFTRow`` with OpenAI chat-format messages + metadata.
    """
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": record.task})

    # Turn-grouped tool-call events → (assistant w/ tool_calls, tool, tool, ...)
    for turn in _group_events_by_turn(record.tool_call_events):
        tool_calls = [
            {
                "id": _synthesize_tool_call_id(evt.turn_id, evt.call_index_in_turn),
                "type": "function",
                "function": {
                    "name": evt.tool_name,
                    "arguments": _format_tool_arguments(evt.params),
                },
            }
            for evt in turn
        ]
        messages.append(
            {
                "role": "assistant",
                # Free-text reasoning between tool calls is not persisted
                # in TrajectoryRecord. Empty content is valid OpenAI
                # format when tool_calls is non-empty.
                "content": "",
                "tool_calls": tool_calls,
            }
        )
        for evt in turn:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _synthesize_tool_call_id(
                        evt.turn_id, evt.call_index_in_turn
                    ),
                    "content": evt.result_content,
                }
            )

    metadata: dict[str, Any] = {
        "trajectory_id": record.trajectory_id,
        "instance_id": record.instance_id,
        "seed": record.seed,
        "model": record.model,
        "final_passed": record.verification.final_passed,
        "has_diff": record.verification.has_diff,
        "tier2_pass": record.verification.tier2_pass,
        "input_tokens": record.token_usage.input_tokens,
        "output_tokens": record.token_usage.output_tokens,
        "tool_call_count": len(record.tool_call_events),
    }

    if label is not None:
        for k in (
            "outcome",
            "contains_recovery",
            "highest_recovery_level",
            "recovery_event_count",
            "liu_categories",
            "is_toxic",
        ):
            if k in label:
                metadata[k] = label[k]

    return SFTRow(messages=messages, metadata=metadata)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


def _load_labels_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load classifier labels keyed by ``trajectory_id``.

    Labels files come from ``experiments.classify_trajectories`` and have
    one row per trajectory with ``trajectory_id`` as the stable join key.
    """
    labels: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("trajectory_id")
            if not isinstance(tid, str) or not tid:
                raise ValueError(
                    f"{path}: label row missing trajectory_id: {row!r}"
                )
            labels[tid] = row
    return labels


def _validate_labels_schema(
    labels: dict[str, dict[str, Any]],
    *,
    require_is_toxic: bool,
    require_outcome: bool,
    require_contains_recovery: bool,
    source: Path | str,
) -> None:
    """Loud-fail on missing keys that an active filter depends on.

    Counter-pattern to ``dict.get(key)``'s silent falsy-default: if
    ``drop_toxic=True`` and a label row has no ``is_toxic`` key, the
    previous code treated that as ``False`` and the row passed. A mixed
    batch of label-file versions would silently ship toxic rows into
    the student corpus. We prefer a loud error the moment we detect it.
    """
    required: list[str] = []
    if require_is_toxic:
        required.append("is_toxic")
    if require_outcome:
        required.append("outcome")
    if require_contains_recovery:
        required.append("contains_recovery")
    if not required:
        return

    missing: dict[str, list[str]] = {}
    for tid, row in labels.items():
        absent = [k for k in required if k not in row]
        if absent:
            missing[tid] = absent
    if missing:
        sample = dict(list(missing.items())[:3])
        raise ValueError(
            f"{source}: {len(missing)} label row(s) missing required keys "
            f"{required} (active filters depend on them). "
            f"Sample of offenders: {sample}. "
            "Re-run the classifier or disable the filter."
        )


def _should_drop(
    record: TrajectoryRecord,
    label: dict[str, Any] | None,
    *,
    include_outcomes: set[str] | None,
    include_recovery_only: bool,
    drop_toxic: bool,
    drop_empty_events: bool,
) -> str | None:
    """Return a reason string if the row should be dropped, else None."""
    if drop_empty_events and not record.tool_call_events:
        return "empty_tool_call_events"

    # If filters reference labels but the label is missing, that's a
    # drop — we want the loud signal, not silent inclusion.
    needs_label = (
        include_outcomes is not None or include_recovery_only or drop_toxic
    )
    if needs_label and label is None:
        return "missing_label"

    if drop_toxic and label and label.get("is_toxic"):
        return "toxic"

    if include_outcomes is not None:
        outcome = label.get("outcome") if label else None
        if outcome not in include_outcomes:
            return f"outcome_not_in_{sorted(include_outcomes)}"

    if include_recovery_only and label and not label.get("contains_recovery"):
        return "no_recovery"

    return None


def export_dataset(
    trajectory_jsonl: Path,
    output_jsonl: Path,
    *,
    labels_jsonl: Path | None = None,
    include_outcomes: Iterable[str] | None = None,
    include_recovery_only: bool = False,
    drop_toxic: bool = True,
    drop_empty_events: bool = True,
    system_prompt: str = "",
    max_trajectories: int | None = None,
) -> ExportReport:
    """Batch-export a trajectory JSONL to an OpenAI-format SFT JSONL.

    The sibling labels JSONL is optional but required whenever any
    outcome / recovery / toxicity filter is active (otherwise a drop
    reason of ``missing_label`` is counted loudly).

    Args:
        trajectory_jsonl: Path to input ``<arm>.jsonl``.
        output_jsonl: Path to write SFT rows. Parent dirs are created.
        labels_jsonl: Path to sibling classifier output. Defaults to
            ``<trajectory_jsonl>.labels.jsonl`` if it exists.
        include_outcomes: Iterable of outcome strings to keep (e.g.
            ``{"verified_with_recovery"}``). ``None`` = no outcome filter.
        include_recovery_only: Keep only rows with ``contains_recovery``.
        drop_toxic: Skip rows flagged toxic by the Liu classifier.
        drop_empty_events: Skip rows with zero tool_call_events (these
            are agent bail-outs; nothing to train on).
        system_prompt: System text to prepend to each conversation.
        max_trajectories: Optional cap on output rows (applied after
            filtering). Useful for smoke runs.

    Returns:
        ``ExportReport`` with counts + filter provenance, ready to serialize.
    """
    include_outcomes_set = (
        set(include_outcomes) if include_outcomes is not None else None
    )

    # Auto-locate sibling labels if not provided.
    if labels_jsonl is None:
        candidate = trajectory_jsonl.with_suffix(".labels.jsonl")
        if candidate.exists():
            labels_jsonl = candidate

    labels = _load_labels_jsonl(labels_jsonl) if labels_jsonl else {}
    if labels:
        _validate_labels_schema(
            labels,
            require_is_toxic=drop_toxic,
            require_outcome=include_outcomes_set is not None,
            require_contains_recovery=include_recovery_only,
            source=labels_jsonl if labels_jsonl else "<labels>",
        )

    records = load_trajectory_jsonl(trajectory_jsonl)

    drop_reasons: dict[str, int] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            if max_trajectories is not None and rows_written >= max_trajectories:
                break
            label = labels.get(record.trajectory_id)
            reason = _should_drop(
                record,
                label,
                include_outcomes=include_outcomes_set,
                include_recovery_only=include_recovery_only,
                drop_toxic=drop_toxic,
                drop_empty_events=drop_empty_events,
            )
            if reason:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                continue

            row = export_trajectory_to_sft(
                record, label=label, system_prompt=system_prompt
            )
            f.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False) + "\n")
            rows_written += 1

    return ExportReport(
        trajectories_loaded=len(records),
        rows_written=rows_written,
        drop_reasons=drop_reasons,
        output_path=str(output_jsonl),
        filters={
            "include_outcomes": (
                sorted(include_outcomes_set) if include_outcomes_set else None
            ),
            "include_recovery_only": include_recovery_only,
            "drop_toxic": drop_toxic,
            "drop_empty_events": drop_empty_events,
            "max_trajectories": max_trajectories,
            "labels_jsonl": str(labels_jsonl) if labels_jsonl else None,
        },
    )


__all__ = [
    "ExportReport",
    "SFTRow",
    "export_dataset",
    "export_trajectory_to_sft",
]
