"""Tests for ``pare.trajectory.sft_export``.

The module is the last mile between classifier-labeled trajectories and
a student-trainable JSONL. Bugs here silently corrupt the training
corpus, so we cover: single-record shape, turn grouping, filter
semantics, label join, and the ``export_dataset`` report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pare.trajectory import (
    SCHEMA_VERSION,
    TrajectoryRecord,
    export_dataset,
    export_trajectory_to_sft,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    turn_id: int,
    call_index_in_turn: int,
    global_index: int,
    tool_name: str = "bash",
    params: dict | None = None,
    result_content: str = "ok",
    result_success: bool = True,
) -> dict:
    return {
        "turn_id": turn_id,
        "call_index_in_turn": call_index_in_turn,
        "global_index": global_index,
        "tool_name": tool_name,
        "params": params or {"command": "true"},
        "result_success": result_success,
        "result_content": result_content,
        "error_signal": "NONE",
        "timestamp": 0.0,
    }


def _record(
    *,
    trajectory_id: str = "t1",
    instance_id: str = "swe-1",
    task: str = "Fix the bug",
    events: list[dict] | None = None,
    final_passed: bool = True,
    input_tokens: int = 1000,
) -> TrajectoryRecord:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "instance_id": instance_id,
        "task": task,
        "model": "minimax",
        "seed": 0,
        "created_at": 1710000000.0,
        "llm_claimed_success": final_passed,
        "verification": {
            "final_passed": final_passed,
            "has_diff": final_passed,
            "tier2_pass": final_passed,
            "tier2_command": "",
        },
        "attempts": [],
        "tool_call_events": events if events is not None else [],
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": 200,
            "cache_read_tokens": 0,
            "cache_create_tokens": 0,
        },
        "metadata": {},
    }
    return TrajectoryRecord.from_dict(payload)


def _label(
    *,
    trajectory_id: str,
    outcome: str,
    contains_recovery: bool = False,
    is_toxic: bool = False,
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "instance_id": "swe-1",
        "seed": 0,
        "outcome": outcome,
        "liu_categories": [],
        "is_toxic": is_toxic,
        "contains_recovery": contains_recovery,
        "highest_recovery_level": "L2" if contains_recovery else None,
        "recovery_event_count": 1 if contains_recovery else 0,
        "recovery_events": [],
        "error_signal_counts": {},
        "liu_detail": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# export_trajectory_to_sft — shape
# ---------------------------------------------------------------------------


class TestExportTrajectoryShape:
    def test_single_turn_single_call_shape(self):
        """One tool call → assistant(tool_calls=[one]) + tool reply.

        The OpenAI fine-tuning API refuses rows where a tool message has
        no matching tool_call_id upstream, so this minimal shape is the
        contract we guarantee.
        """
        record = _record(events=[
            _event(turn_id=0, call_index_in_turn=0, global_index=0,
                   tool_name="bash", params={"command": "ls"},
                   result_content="file.py"),
        ])
        row = export_trajectory_to_sft(record, system_prompt="SYS")

        # [system, user, assistant, tool]
        assert [m["role"] for m in row.messages] == [
            "system", "user", "assistant", "tool",
        ]
        assert row.messages[0]["content"] == "SYS"
        assert row.messages[1]["content"] == "Fix the bug"
        assert row.messages[2]["content"] == ""
        assert len(row.messages[2]["tool_calls"]) == 1

        tc = row.messages[2]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "bash"
        # Arguments MUST be a JSON string (OpenAI schema), not a dict.
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

        # tool message tool_call_id references the same synthetic id.
        assert row.messages[3]["tool_call_id"] == tc["id"]
        assert row.messages[3]["content"] == "file.py"

    def test_omits_system_when_empty(self):
        """Empty system_prompt → no system message (matches runtime loop
        behaviour in ``_build_initial_messages``)."""
        record = _record(events=[
            _event(turn_id=0, call_index_in_turn=0, global_index=0),
        ])
        row = export_trajectory_to_sft(record)
        assert row.messages[0]["role"] == "user"

    def test_groups_multiple_calls_in_same_turn(self):
        """Two tool calls with the same turn_id → one assistant message
        carrying both tool_calls, then two tool replies in order.

        This is the ReAct parallel-call pattern; flattening it into
        two separate assistant messages would diverge from how the
        trajectory was actually recorded and train the student on a
        fake cadence.
        """
        record = _record(events=[
            _event(turn_id=0, call_index_in_turn=0, global_index=0,
                   tool_name="file_read", params={"file_path": "a.py"},
                   result_content="A"),
            _event(turn_id=0, call_index_in_turn=1, global_index=1,
                   tool_name="file_read", params={"file_path": "b.py"},
                   result_content="B"),
        ])
        row = export_trajectory_to_sft(record)
        roles = [m["role"] for m in row.messages]
        # user, assistant(2 calls), tool, tool
        assert roles == ["user", "assistant", "tool", "tool"]
        assert len(row.messages[1]["tool_calls"]) == 2
        assert row.messages[2]["content"] == "A"
        assert row.messages[3]["content"] == "B"

    def test_separates_turns(self):
        """Events with different turn_ids → separate assistant messages."""
        record = _record(events=[
            _event(turn_id=0, call_index_in_turn=0, global_index=0,
                   tool_name="bash", result_content="out1"),
            _event(turn_id=1, call_index_in_turn=0, global_index=1,
                   tool_name="bash", result_content="out2"),
        ])
        row = export_trajectory_to_sft(record)
        roles = [m["role"] for m in row.messages]
        # user, assistant, tool, assistant, tool
        assert roles == ["user", "assistant", "tool", "assistant", "tool"]

    def test_tool_call_ids_are_unique_within_conversation(self):
        """The OpenAI format hard-requires unique tool_call_ids within
        a single conversation. Synthesized ids must respect that across
        turns and calls."""
        record = _record(events=[
            _event(turn_id=0, call_index_in_turn=0, global_index=0),
            _event(turn_id=0, call_index_in_turn=1, global_index=1),
            _event(turn_id=1, call_index_in_turn=0, global_index=2),
            _event(turn_id=2, call_index_in_turn=0, global_index=3),
        ])
        row = export_trajectory_to_sft(record)
        ids = [
            tc["id"]
            for m in row.messages
            if m["role"] == "assistant"
            for tc in m["tool_calls"]
        ]
        assert len(ids) == len(set(ids)) == 4

    def test_empty_events_still_produces_minimal_conversation(self):
        """Zero tool_call_events → just [user] (or [system, user]).

        This shape is never written by export_dataset (drop_empty_events
        is on by default), but the pure function must not crash on it —
        callers that disable the filter deserve a sensible row.
        """
        record = _record(events=[])
        row = export_trajectory_to_sft(record, system_prompt="SYS")
        assert [m["role"] for m in row.messages] == ["system", "user"]

    def test_metadata_includes_verification_and_tokens(self):
        record = _record(
            events=[_event(turn_id=0, call_index_in_turn=0, global_index=0)],
            final_passed=True,
            input_tokens=5000,
        )
        row = export_trajectory_to_sft(record)
        assert row.metadata["final_passed"] is True
        assert row.metadata["tier2_pass"] is True
        assert row.metadata["input_tokens"] == 5000
        assert row.metadata["output_tokens"] == 200
        assert row.metadata["tool_call_count"] == 1

    def test_metadata_merges_label_fields(self):
        record = _record(events=[
            _event(turn_id=0, call_index_in_turn=0, global_index=0),
        ])
        label = _label(
            trajectory_id="t1",
            outcome="verified_with_recovery",
            contains_recovery=True,
        )
        row = export_trajectory_to_sft(record, label=label)
        assert row.metadata["outcome"] == "verified_with_recovery"
        assert row.metadata["contains_recovery"] is True
        assert row.metadata["highest_recovery_level"] == "L2"
        assert row.metadata["is_toxic"] is False


# ---------------------------------------------------------------------------
# export_dataset — filter semantics + report
# ---------------------------------------------------------------------------


class TestExportDatasetFilters:
    def _write_two_arm_fixture(
        self, tmp_path: Path
    ) -> tuple[Path, Path, Path]:
        """Three trajectories: recovery-success, one-shot, toxic."""
        traj_path = tmp_path / "arm.jsonl"
        label_path = tmp_path / "arm.labels.jsonl"
        out_path = tmp_path / "sft.jsonl"

        records = [
            _record(
                trajectory_id="t_recover",
                events=[
                    _event(turn_id=0, call_index_in_turn=0, global_index=0),
                    _event(turn_id=1, call_index_in_turn=0, global_index=1),
                ],
            ),
            _record(
                trajectory_id="t_oneshot",
                events=[
                    _event(turn_id=0, call_index_in_turn=0, global_index=0),
                ],
            ),
            _record(
                trajectory_id="t_toxic",
                events=[
                    _event(turn_id=0, call_index_in_turn=0, global_index=0),
                ],
                final_passed=False,
            ),
        ]
        _write_jsonl(traj_path, [r.to_dict() for r in records])
        _write_jsonl(label_path, [
            _label(trajectory_id="t_recover",
                   outcome="verified_with_recovery",
                   contains_recovery=True),
            _label(trajectory_id="t_oneshot",
                   outcome="verified_one_shot"),
            _label(trajectory_id="t_toxic",
                   outcome="failed",
                   is_toxic=True),
        ])
        return traj_path, label_path, out_path

    def test_round_trip_no_filter(self, tmp_path: Path):
        """Without filters (drop_toxic still on), get 2 rows (toxic
        dropped) + report round-trips."""
        traj, _labels, out = self._write_two_arm_fixture(tmp_path)
        report = export_dataset(
            traj, out, include_outcomes=None, include_recovery_only=False
        )
        assert report.trajectories_loaded == 3
        assert report.rows_written == 2
        assert report.drop_reasons == {"toxic": 1}

        # Output file has exactly 2 well-formed JSON lines.
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj
            assert "metadata" in obj

    def test_include_recovery_only(self, tmp_path: Path):
        """``include_recovery_only=True`` → only t_recover survives."""
        traj, _labels, out = self._write_two_arm_fixture(tmp_path)
        report = export_dataset(traj, out, include_recovery_only=True)
        assert report.rows_written == 1

        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["metadata"]["trajectory_id"] == "t_recover"
        assert "no_recovery" in report.drop_reasons

    def test_include_outcomes_filter(self, tmp_path: Path):
        """Restrict to explicit outcome whitelist."""
        traj, _labels, out = self._write_two_arm_fixture(tmp_path)
        report = export_dataset(
            traj,
            out,
            include_outcomes={"verified_one_shot"},
        )
        assert report.rows_written == 1
        row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert row["metadata"]["trajectory_id"] == "t_oneshot"

    def test_missing_label_is_loud(self, tmp_path: Path):
        """Filter requires labels → missing label counts as a drop
        (not a silent include). This is the anti-footgun.

        Ablation dataset with a filter but no labels file is a bug
        in the caller's invocation — we prefer to hand them an empty
        output + a clear ``missing_label: N`` count rather than
        trainable rows of unknown provenance.
        """
        traj_path = tmp_path / "arm.jsonl"
        out_path = tmp_path / "sft.jsonl"
        _write_jsonl(traj_path, [
            _record(trajectory_id="t1",
                    events=[_event(turn_id=0, call_index_in_turn=0,
                                   global_index=0)]).to_dict(),
        ])
        # No sibling labels file.
        report = export_dataset(
            traj_path,
            out_path,
            include_outcomes={"verified_with_recovery"},
        )
        assert report.rows_written == 0
        assert report.drop_reasons.get("missing_label") == 1

    def test_drop_empty_events(self, tmp_path: Path):
        """Trajectory with zero tool_call_events → dropped with reason.

        The flat-ReAct loop occasionally records these when the LLM
        emits zero tool calls on turn 0 (provider hiccup / empty reply).
        Training on them teaches the student to output nothing, which
        is exactly the behaviour we don't want.
        """
        traj_path = tmp_path / "arm.jsonl"
        out_path = tmp_path / "sft.jsonl"
        _write_jsonl(traj_path, [
            _record(trajectory_id="t1", events=[]).to_dict(),
            _record(trajectory_id="t2",
                    events=[_event(turn_id=0, call_index_in_turn=0,
                                   global_index=0)]).to_dict(),
        ])
        report = export_dataset(
            traj_path, out_path, drop_toxic=False
        )
        assert report.rows_written == 1
        assert report.drop_reasons.get("empty_tool_call_events") == 1

    def test_max_trajectories_cap(self, tmp_path: Path):
        traj, _labels, out = self._write_two_arm_fixture(tmp_path)
        report = export_dataset(
            traj, out, drop_toxic=False, max_trajectories=1
        )
        assert report.rows_written == 1

    def test_report_filters_are_serializable(self, tmp_path: Path):
        """``ExportReport.to_dict`` must be json-dumpable — the CLI
        writes it next to the JSONL for audit."""
        traj, _labels, out = self._write_two_arm_fixture(tmp_path)
        report = export_dataset(
            traj,
            out,
            include_outcomes={"verified_with_recovery"},
            include_recovery_only=True,
        )
        blob = json.dumps(report.to_dict())
        assert "include_outcomes" in blob
        assert "drop_reasons" in blob

    def test_auto_locates_sibling_labels(self, tmp_path: Path):
        """``<traj>.labels.jsonl`` is picked up automatically — this is
        the classifier's canonical output path, so the happy path should
        require zero filter arguments."""
        traj, _labels, out = self._write_two_arm_fixture(tmp_path)
        # Don't pass labels_jsonl explicitly.
        report = export_dataset(
            traj, out, include_recovery_only=True
        )
        assert report.rows_written == 1
        assert report.filters["labels_jsonl"] is not None


# ---------------------------------------------------------------------------
# Label schema validation (counter-pattern to silent ``dict.get`` defaults)
# ---------------------------------------------------------------------------


class TestLabelSchemaValidation:
    """Anti-footgun: if the caller activates a filter that depends on a
    label key and the labels file is missing that key, fail loudly at
    load time rather than silently treating the key as ``False`` and
    leaking toxic / non-recovery rows into the corpus."""

    def _minimal_record_and_traj(self, tmp_path: Path) -> tuple[Path, Path]:
        traj_path = tmp_path / "arm.jsonl"
        out_path = tmp_path / "sft.jsonl"
        _write_jsonl(
            traj_path,
            [
                _record(
                    trajectory_id="t1",
                    events=[
                        _event(turn_id=0, call_index_in_turn=0, global_index=0)
                    ],
                ).to_dict()
            ],
        )
        return traj_path, out_path

    def test_drop_toxic_requires_is_toxic_key(self, tmp_path: Path):
        """Labels that predate the toxic field silently pass the filter
        under ``.get()``. Require the key instead."""
        traj, out = self._minimal_record_and_traj(tmp_path)
        labels_path = tmp_path / "arm.labels.jsonl"
        _write_jsonl(
            labels_path,
            [
                {
                    "trajectory_id": "t1",
                    "outcome": "verified_one_shot",
                    # is_toxic missing on purpose
                }
            ],
        )
        with pytest.raises(ValueError, match="is_toxic"):
            export_dataset(
                traj, out, labels_jsonl=labels_path, drop_toxic=True
            )

    def test_include_recovery_only_requires_contains_recovery_key(
        self, tmp_path: Path
    ):
        traj, out = self._minimal_record_and_traj(tmp_path)
        labels_path = tmp_path / "arm.labels.jsonl"
        _write_jsonl(
            labels_path,
            [
                {
                    "trajectory_id": "t1",
                    "outcome": "verified_one_shot",
                    "is_toxic": False,
                    # contains_recovery missing on purpose
                }
            ],
        )
        with pytest.raises(ValueError, match="contains_recovery"):
            export_dataset(
                traj,
                out,
                labels_jsonl=labels_path,
                drop_toxic=True,
                include_recovery_only=True,
            )

    def test_include_outcomes_requires_outcome_key(self, tmp_path: Path):
        traj, out = self._minimal_record_and_traj(tmp_path)
        labels_path = tmp_path / "arm.labels.jsonl"
        _write_jsonl(
            labels_path,
            [
                {
                    "trajectory_id": "t1",
                    "is_toxic": False,
                    # outcome missing on purpose
                }
            ],
        )
        with pytest.raises(ValueError, match="outcome"):
            export_dataset(
                traj,
                out,
                labels_jsonl=labels_path,
                include_outcomes={"verified_one_shot"},
            )

    def test_all_filters_off_accepts_minimal_labels(self, tmp_path: Path):
        """No filters active → no keys required. Purely informational
        label rows (e.g. ``{trajectory_id, notes}``) must still load."""
        traj, out = self._minimal_record_and_traj(tmp_path)
        labels_path = tmp_path / "arm.labels.jsonl"
        _write_jsonl(
            labels_path,
            [{"trajectory_id": "t1", "notes": "informational only"}],
        )
        # drop_toxic=False, no outcome/recovery filters → zero requirements.
        report = export_dataset(
            traj,
            out,
            labels_jsonl=labels_path,
            drop_toxic=False,
        )
        assert report.rows_written == 1


# ---------------------------------------------------------------------------
# Event grouping — defend against reordered input
# ---------------------------------------------------------------------------


class TestEventGroupingSortsByGlobalIndex:
    """The loop guarantees append order today, but the grouper must not
    depend on that — any future resampler / deduper / hand edit of the
    JSONL that reorders events would otherwise split one assistant turn
    into multiple SFT rows with partial tool_calls."""

    def test_shuffled_input_still_groups_correctly(self):
        """The grouper sorts by ``global_index`` before scanning, so
        shuffled input produces the same grouping as in-order input.

        We use a tiny duck-typed stand-in rather than the real
        ``ToolCallEvent`` to keep this test focused on the grouper's
        ordering invariant — the grouper only reads ``.turn_id`` and
        ``.global_index``."""
        from pare.trajectory.sft_export import _group_events_by_turn

        class _E:  # minimal event stand-in
            __slots__ = ("turn_id", "global_index", "tag")

            def __init__(self, turn_id: int, global_index: int, tag: str):
                self.turn_id = turn_id
                self.global_index = global_index
                self.tag = tag

        # Turn 0 has two calls; turn 1 has one call. Feed them mixed.
        shuffled = [
            _E(turn_id=1, global_index=2, tag="X"),
            _E(turn_id=0, global_index=1, tag="B"),
            _E(turn_id=0, global_index=0, tag="A"),
        ]
        groups = _group_events_by_turn(shuffled)
        # Two groups: turn 0 (two events), turn 1 (one event), in order.
        assert [e.turn_id for e in groups[0]] == [0, 0]
        assert [e.turn_id for e in groups[1]] == [1]
        # Within turn 0, events are sorted by global_index (A then B).
        assert [e.tag for e in groups[0]] == ["A", "B"]
