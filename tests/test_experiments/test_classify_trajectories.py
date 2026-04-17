"""Tests for trajectory classification script (v2 pipeline)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.classify_trajectories import classify_one, classify_trajectories, main
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    load_trajectory_jsonl,
    write_trajectory_jsonl,
)
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


def _record(
    trajectory_id: str,
    *,
    llm_claimed_success: bool,
    final_passed: bool,
    tier1_pass: bool,
    tier2_pass: bool,
    tier2_command: str = "",
    attempts: list[StepAttempt] | None = None,
    tool_call_events: list[ToolCallEvent] | None = None,
    metadata: dict[str, str] | None = None,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=trajectory_id,
        instance_id=f"inst-{trajectory_id}",
        task="Fix bug",
        model="deepseek/deepseek-chat",
        seed=0,
        created_at=1710000000.0,
        llm_claimed_success=llm_claimed_success,
        verification=VerificationResult(
            final_passed=final_passed,
            tier1_pass=tier1_pass,
            tier2_pass=tier2_pass,
            tier2_command=tier2_command,
        ),
        attempts=attempts or [],
        tool_call_events=tool_call_events or [],
        token_usage=TokenUsageSummary(input_tokens=100, output_tokens=20),
        metadata=metadata or {},
    )


def _attempt(
    step: int,
    attempt_number: int,
    status: str,
) -> StepAttempt:
    return StepAttempt(
        step_number=step,
        attempt_number=attempt_number,
        goal="Fix parser",
        status=status,
        target_files=["a.py"],
        tool_names=["file_edit"],
        failure_reason="" if status == "success" else "error",
    )


def _evt(
    gi: int,
    turn: int,
    tool: str,
    *,
    ok: bool = True,
    target: str = "",
    params_hash: str = "",
    content: str = "",
) -> ToolCallEvent:
    return ToolCallEvent(
        turn_id=turn,
        call_index_in_turn=0,
        global_index=gi,
        tool_name=tool,
        params={},
        params_hash=params_hash or f"h{gi}",
        target_file=target,
        result_success=ok,
        result_content=content,
        error_signal=ErrorSignal.NONE,
        timestamp=float(gi),
    )


class TestClassifyOne:
    def test_toxic_c2(self):
        """Agent claims success, tier1 fails → C2 → toxic."""
        rec = _record("t", llm_claimed_success=True,
                       final_passed=False, tier1_pass=False, tier2_pass=False)
        result = classify_one(rec)
        assert result.outcome.value == "toxic"
        assert result.liu.c2_premature_success is True

    def test_weakly_verified(self):
        """Tier1 passes, no tier2 configured → weakly_verified."""
        rec = _record("w", llm_claimed_success=True,
                       final_passed=True, tier1_pass=True, tier2_pass=False)
        result = classify_one(rec)
        assert result.outcome.value == "weakly_verified"

    def test_verified_one_shot(self):
        """Both tiers pass, no recovery → verified_one_shot."""
        rec = _record("o", llm_claimed_success=True,
                       final_passed=True, tier1_pass=True, tier2_pass=True,
                       tier2_command="pytest tests/")
        result = classify_one(rec)
        assert result.outcome.value == "verified_one_shot"

    def test_with_recovery_events(self):
        """Error→correction in tool calls → contains_recovery."""
        events = [
            _evt(0, 0, "bash", ok=False, content="command not found"),
            _evt(1, 1, "bash", ok=True, content="ok", params_hash="fix"),
        ]
        rec = _record("r", llm_claimed_success=True,
                       final_passed=True, tier1_pass=True, tier2_pass=True,
                       tier2_command="pytest tests/",
                       tool_call_events=events)
        result = classify_one(rec)
        assert result.recovery.contains_recovery is True
        assert result.outcome.value == "verified_with_recovery"

    def test_b11_uses_metadata_final_diff_over_file_edit_reconstruction(self):
        """B1.1 must compare the gold_patch against the real unified diff stored in
        metadata['final_diff'], not synthesize one from file_edit tool results.

        Scenario: the agent fully matches the gold_patch (same file + same hunk
        count). file_edit tool result_content is a confirmation string, which
        would yield 0 hunks under the old reconstruction path and wrongly flag
        B1.1=True. The real diff in metadata should override that.
        """
        gold = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        # Real diff matches gold exactly — B1.1 should be False.
        real_diff = gold
        events = [
            _evt(0, 0, "file_edit", ok=True, content="Edited a.py: 1 line changed"),
        ]
        rec = _record(
            "b11-real",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            tier2_command="pytest tests/",
            tool_call_events=events,
            metadata={"final_diff": real_diff},
        )
        result = classify_one(rec, gold_patch=gold)
        assert result.liu.b11_incomplete_fix is False

    def test_b11_skipped_when_no_final_diff_in_metadata(self):
        """Legacy records without metadata['final_diff'] must not reconstruct
        from file_edit results (that would always yield a degenerate 0-hunk
        diff and wrongly fire B1.1 for every non-empty gold_patch)."""
        gold = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        events = [
            _evt(0, 0, "file_edit", ok=True, content="Edited a.py: 1 line changed"),
        ]
        rec = _record(
            "b11-legacy",
            llm_claimed_success=True,
            final_passed=False,
            tier1_pass=True,
            tier2_pass=False,
            tier2_command="pytest tests/",
            tool_call_events=events,
            metadata={},
        )
        # With empty final_diff and non-empty gold, detect_b11_incomplete_fix
        # returns True (documented behaviour: "no fix at all"). That's fine —
        # it's a principled signal based on the absence of a real diff, not a
        # false positive driven by misreconstruction.
        result = classify_one(rec, gold_patch=gold)
        assert result.liu.b11_incomplete_fix is True

    def test_no_events_backward_compat(self):
        """v1 trajectory with 0 events classifies without error."""
        rec = _record("v1", llm_claimed_success=True,
                       final_passed=True, tier1_pass=True, tier2_pass=False)
        result = classify_one(rec)
        assert result.outcome.value == "weakly_verified"
        assert result.recovery.contains_recovery is False
        assert result.liu.categories == []


class TestClassifyTrajectoriesScript:
    def test_classify_and_write_outputs(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        non_toxic_path = tmp_path / "non_toxic.jsonl"

        records = [
            _record(
                "toxic",
                llm_claimed_success=True,
                final_passed=False,
                tier1_pass=False,
                tier2_pass=False,
            ),
            _record(
                "weak",
                llm_claimed_success=True,
                final_passed=True,
                tier1_pass=True,
                tier2_pass=False,
            ),
            _record(
                "one",
                llm_claimed_success=True,
                final_passed=True,
                tier1_pass=True,
                tier2_pass=True,
                tier2_command="pytest tests/",
            ),
        ]
        write_trajectory_jsonl(trajectory_path, records)

        summary = classify_trajectories(
            trajectory_path,
            labels_jsonl=labels_path,
            non_toxic_jsonl=non_toxic_path,
        )

        assert summary.total == 3
        assert summary.toxic_count == 1
        assert summary.non_toxic_count == 2
        assert labels_path.exists()
        assert non_toxic_path.exists()

        loaded_non_toxic = load_trajectory_jsonl(non_toxic_path)
        assert len(loaded_non_toxic) == 2

        lines = [l for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

        # Check v2 label format
        payload = json.loads(lines[0])
        assert "outcome" in payload
        assert "liu_categories" in payload
        assert "contains_recovery" in payload
        assert payload["outcome"] == "toxic"
        assert payload["is_toxic"] is True

    def test_summary_counts(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        records = [
            _record("a", llm_claimed_success=True,
                    final_passed=True, tier1_pass=True, tier2_pass=True,
                    tier2_command="pytest"),
            _record("b", llm_claimed_success=True,
                    final_passed=True, tier1_pass=True, tier2_pass=True,
                    tier2_command="pytest"),
        ]
        write_trajectory_jsonl(trajectory_path, records)

        summary = classify_trajectories(
            trajectory_path,
            labels_jsonl=tmp_path / "labels.jsonl",
            non_toxic_jsonl=tmp_path / "non_toxic.jsonl",
        )
        assert summary.outcome_counts == {"verified_one_shot": 2}
        assert summary.toxic_count == 0
        assert summary.recovery_count == 0

    def test_main_writes_default_outputs(self, tmp_path: Path):
        trajectory_path = tmp_path / "traj.jsonl"
        write_trajectory_jsonl(
            trajectory_path,
            [
                _record(
                    "one",
                    llm_claimed_success=True,
                    final_passed=True,
                    tier1_pass=True,
                    tier2_pass=True,
                    tier2_command="pytest",
                )
            ],
        )

        code = main([
            "--trajectory-jsonl",
            str(trajectory_path),
        ])
        assert code == 0

        assert trajectory_path.with_suffix(".labels.jsonl").exists()
        assert trajectory_path.with_suffix(".summary.json").exists()
        assert trajectory_path.with_suffix(".non_toxic.jsonl").exists()

    def test_empty_file_raises(self, tmp_path: Path):
        trajectory_path = tmp_path / "empty.jsonl"
        trajectory_path.write_text("")

        code = main(["--trajectory-jsonl", str(trajectory_path)])
        assert code == 1
