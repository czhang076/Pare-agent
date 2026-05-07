"""Schema v2 compatibility regression guard.

Post-R3/R4, the agent layer was rewritten (flat ReAct + InstanceContainer)
but ``TrajectoryRecord`` stays the master JSONL schema: v1 attempts +
optional v2 ``tool_call_events`` on the same record.

This test is a structural canary: if any of ``ToolCallEvent``,
``ErrorSignal``, ``TrajectoryRecord``, or the downstream classifier
pipeline (``classify_trajectory_signals`` → ``classify_liu_from_record``
→ ``detect_recovery_events`` → ``assign_outcome_label``) drifts in a way
that silently corrupts classification output, this file should fail.

The risk it guards against (plan §R3/R4):
> Classifier pipeline because schema drift silently corrupts data —
> classifier_liu / recovery_detector_v2 / error_signal_extractor all read
> TrajectoryRecord.tool_call_events; if the new loop constructs events
> with slightly-off field semantics, whole batches misclassify.

Deliberately hand-constructs both v1 ("no events") and v2 ("with events")
records so the suite is independent of any pilot JSONL on disk.
"""

from __future__ import annotations

import json

import pytest

from pare.trajectory.classifier_liu import (
    OutcomeLabel,
    assign_outcome_label,
    classify_liu_from_record,
)
from pare.trajectory.error_signal_extractor import (
    classify_trajectory_signals,
    extract_error_signal,
)
from pare.trajectory.recovery_detector_v2 import detect_recovery_events
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
)
from pare.trajectory.schema_v2 import ErrorSignal, ToolCallEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    global_index: int = 0,
    tool_name: str = "bash",
    params: dict | None = None,
    result_success: bool = True,
    result_content: str = "",
    error_signal: ErrorSignal = ErrorSignal.NONE,
    timestamp: float = 0.0,
) -> ToolCallEvent:
    """Build a ToolCallEvent via the factory (same path as loop.py)."""
    return ToolCallEvent.create(
        turn_id=global_index,
        call_index_in_turn=0,
        global_index=global_index,
        tool_name=tool_name,
        params=params or {},
        result_success=result_success,
        result_content=result_content,
        timestamp=timestamp,
        error_signal=error_signal,
    )


def _record(
    *,
    tool_call_events: list[ToolCallEvent] | None = None,
    has_diff: bool = False,
    tier2_pass: bool = False,
    tier2_command: str = "pytest",
    llm_claimed_success: bool = False,
    instance_id: str = "test__test-0001",
) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=f"traj-{instance_id}",
        instance_id=instance_id,
        task="synthetic task",
        model="synthetic-model",
        seed=0,
        created_at=0.0,
        llm_claimed_success=llm_claimed_success,
        verification=VerificationResult(
            final_passed=has_diff and tier2_pass,
            has_diff=has_diff,
            tier2_pass=tier2_pass,
            tier2_command=tier2_command,
        ),
        attempts=[],
        tool_call_events=tool_call_events or [],
        token_usage=TokenUsageSummary(),
        metadata={},
    )


def _run_full_pipeline(record: TrajectoryRecord) -> OutcomeLabel:
    """The same call graph sampler.py uses post-R4."""
    signals = classify_trajectory_signals(record.tool_call_events)
    liu = classify_liu_from_record(record, signals)
    recovery = detect_recovery_events(record.tool_call_events, signals)
    return assign_outcome_label(liu, record.verification, recovery.contains_recovery)


# ---------------------------------------------------------------------------
# 1. Empty tool_call_events (v1-shaped) must still classify cleanly
# ---------------------------------------------------------------------------


class TestV1SchemaCompat:
    """Records authored before v2 events were added must still classify."""

    def test_empty_events_with_tier2_fail_classifies_as_failed(self):
        record = _record(tier2_pass=False, tier2_command="pytest x.py")
        assert _run_full_pipeline(record) == OutcomeLabel.FAILED

    def test_empty_events_with_tier2_pass_classifies_as_one_shot(self):
        record = _record(
            has_diff=True,
            tier2_pass=True,
            tier2_command="pytest x.py",
        )
        # No events → no recovery → ONE_SHOT rather than WITH_RECOVERY
        assert _run_full_pipeline(record) == OutcomeLabel.VERIFIED_ONE_SHOT

    def test_empty_events_no_tier2_command_weakly_verified(self):
        record = _record(has_diff=True, tier2_command="")
        assert _run_full_pipeline(record) == OutcomeLabel.WEAKLY_VERIFIED

    def test_empty_events_no_tier2_and_no_tier1_fails(self):
        record = _record(has_diff=False, tier2_command="")
        assert _run_full_pipeline(record) == OutcomeLabel.FAILED


# ---------------------------------------------------------------------------
# 2. v2 events round-trip JSON without corrupting classification
# ---------------------------------------------------------------------------


class TestV2SchemaRoundTrip:
    """Every TrajectoryRecord persisted via to_json_line must read back
    into an object that produces identical classifier output.

    This is the core silent-corruption guard: it catches any field whose
    serialization differs from deserialization (enum values, float
    precision, default fills)."""

    def test_round_trip_preserves_outcome_label(self):
        events = [
            _event(global_index=0, tool_name="file_read",
                   params={"file_path": "a.py"}),
            _event(global_index=1, tool_name="file_edit",
                   params={"file_path": "a.py", "old": "x", "new": "y"},
                   result_success=False, result_content="IndentationError: bad",
                   error_signal=ErrorSignal.SYNTAX_ERROR),
            _event(global_index=2, tool_name="file_edit",
                   params={"file_path": "a.py", "old": "x", "new": "y fixed"}),
        ]
        original = _record(tool_call_events=events, tier2_command="pytest x.py")

        serialized = original.to_json_line()
        restored = TrajectoryRecord.from_json_line(serialized)

        assert _run_full_pipeline(original) == _run_full_pipeline(restored)

    def test_round_trip_preserves_recovery_detection(self):
        events = [
            _event(global_index=0, tool_name="file_edit",
                   params={"file_path": "foo.py", "old": "a", "new": "b"},
                   result_success=False,
                   result_content="RuntimeError: bad",
                   error_signal=ErrorSignal.RUNTIME_ERROR),
            _event(global_index=1, tool_name="file_edit",
                   params={"file_path": "foo.py", "old": "a", "new": "c"}),
        ]
        original = _record(tool_call_events=events)

        signals_orig = classify_trajectory_signals(original.tool_call_events)
        recovery_orig = detect_recovery_events(original.tool_call_events, signals_orig)

        restored = TrajectoryRecord.from_json_line(original.to_json_line())
        signals_rest = classify_trajectory_signals(restored.tool_call_events)
        recovery_rest = detect_recovery_events(restored.tool_call_events, signals_rest)

        assert recovery_orig.contains_recovery == recovery_rest.contains_recovery
        assert len(recovery_orig.recovery_events) == len(recovery_rest.recovery_events)

    def test_round_trip_preserves_error_signal_enum(self):
        """Every ErrorSignal value must survive JSON serialization."""
        events = [
            _event(global_index=i, tool_name="bash",
                   error_signal=sig,
                   result_success=(sig == ErrorSignal.NONE))
            for i, sig in enumerate(ErrorSignal)
        ]
        original = _record(tool_call_events=events)
        restored = TrajectoryRecord.from_json_line(original.to_json_line())

        restored_sigs = [e.error_signal for e in restored.tool_call_events]
        original_sigs = [e.error_signal for e in original.tool_call_events]
        assert restored_sigs == original_sigs


# ---------------------------------------------------------------------------
# 3. Coverage guard — every ErrorSignal must be classifiable
# ---------------------------------------------------------------------------


class TestErrorSignalCoverage:
    """If a new ErrorSignal enum value is added, the pipeline must still
    run without raising. This catches the "enum added, classifier branch
    missing" failure mode that produced silent mis-categorization in v1."""

    @pytest.mark.parametrize("sig", list(ErrorSignal))
    def test_every_error_signal_survives_pipeline(self, sig: ErrorSignal):
        events = [_event(global_index=0, tool_name="bash", error_signal=sig,
                         result_success=(sig == ErrorSignal.NONE))]
        record = _record(tool_call_events=events, tier2_command="pytest x.py")

        label = _run_full_pipeline(record)
        # Must be a valid OutcomeLabel — not None, not raising.
        assert isinstance(label, OutcomeLabel)


# ---------------------------------------------------------------------------
# 4. ToolCallEvent.create is the sole event factory — params_hash
# / target_file must be computed, not defaulted to empty.
# ---------------------------------------------------------------------------


class TestFactoryInvariants:
    """loop.py relies on ToolCallEvent.create populating derived fields.
    If those fields get silently lost (someone refactors to plain ctor
    calls), recovery detection's same-file / param-differ logic breaks."""

    def test_factory_sets_params_hash(self):
        e = _event(tool_name="bash", params={"command": "ls"})
        assert len(e.params_hash) == 16  # 16-char sha prefix

    def test_factory_sets_target_file_for_file_tools(self):
        e = _event(tool_name="file_edit",
                   params={"file_path": "a/b.py", "old": "x", "new": "y"})
        assert e.target_file == "a/b.py"

    def test_factory_sets_target_file_for_search_tool(self):
        e = _event(tool_name="search", params={"pattern": "foo", "path": "src"})
        assert e.target_file == "src"

    def test_factory_leaves_bash_target_file_empty(self):
        e = _event(tool_name="bash", params={"command": "pytest x.py"})
        assert e.target_file == ""

    def test_params_hash_is_deterministic_across_param_key_order(self):
        a = _event(tool_name="file_edit", params={"file_path": "x", "new": "n", "old": "o"})
        b = _event(tool_name="file_edit", params={"old": "o", "file_path": "x", "new": "n"})
        assert a.params_hash == b.params_hash


# ---------------------------------------------------------------------------
# 5. Full pipeline stability on a realistic v2 trajectory
# ---------------------------------------------------------------------------


class TestRealisticV2Pipeline:
    """Mirror the v6_refactor_smoke shape: read → fail-edit → fix-edit →
    declare_done. Exercises recovery detection + Liu categorization +
    outcome labeling jointly."""

    def test_recovery_trajectory_classifies_consistently(self):
        events = [
            _event(global_index=0, tool_name="file_read",
                   params={"file_path": "mod.py"}),
            _event(global_index=1, tool_name="file_edit",
                   params={"file_path": "mod.py", "old": "a = 1", "new": "a = "},
                   result_success=False,
                   result_content="SyntaxError: invalid syntax",
                   error_signal=ErrorSignal.SYNTAX_ERROR),
            _event(global_index=2, tool_name="file_edit",
                   params={"file_path": "mod.py", "old": "a = ", "new": "a = 2"}),
            _event(global_index=3, tool_name="declare_done",
                   params={"status": "fixed", "summary": "ok"}),
        ]
        # Note: llm_claimed_success=False so this scenario tests
        # "recovery happened but Tier 2 still failed" → FAILED.
        # If llm_claimed_success were True with tier1=False, C2 would
        # fire and the outcome would be TOXIC — that's a separate case.
        record = _record(
            tool_call_events=events,
            has_diff=False,
            tier2_pass=False,
            tier2_command="pytest mod.py",
            llm_claimed_success=False,
        )

        signals = classify_trajectory_signals(record.tool_call_events)
        assert signals[1] == ErrorSignal.SYNTAX_ERROR

        liu = classify_liu_from_record(record, signals)
        # B2.2 (syntax error in final state) would be toxic. Here the agent
        # corrected it, so b22_syntax_error should NOT fire as final state.
        # This asserts the classifier reads the *sequence*, not just a
        # snapshot — if that logic regresses, this test flips.
        assert liu.b22_syntax_error is False
        assert liu.c2_premature_success is False  # agent didn't claim success

        recovery = detect_recovery_events(record.tool_call_events, signals)
        assert recovery.contains_recovery is True

        outcome = assign_outcome_label(liu, record.verification, recovery.contains_recovery)
        # Tier 2 configured and failed → FAILED (regardless of recovery).
        assert outcome == OutcomeLabel.FAILED


# ---------------------------------------------------------------------------
# 6. JSONL on-the-wire format — required top-level keys frozen
# ---------------------------------------------------------------------------


class TestJSONLWireFormat:
    """Downstream research tools (sampler, sft_exporter, classify_trajectories)
    all read JSONL line-by-line via ``TrajectoryRecord.from_json_line``.
    These assertions freeze the minimum required key set on the wire;
    breaking them means downstream consumers break, silently or loudly."""

    def test_record_serializes_required_top_level_keys(self):
        record = _record()
        payload = json.loads(record.to_json_line())
        for key in (
            "schema_version", "trajectory_id", "instance_id", "task",
            "model", "seed", "created_at", "llm_claimed_success",
            "verification", "tool_call_events",
        ):
            assert key in payload, f"missing required key: {key}"

    def test_tool_call_event_serializes_required_subkeys(self):
        events = [_event(global_index=0, tool_name="bash",
                         params={"command": "ls"})]
        record = _record(tool_call_events=events)
        payload = json.loads(record.to_json_line())
        event_payload = payload["tool_call_events"][0]
        for key in (
            "turn_id", "call_index_in_turn", "global_index",
            "tool_name", "result_success", "timestamp",
        ):
            assert key in event_payload, f"missing event key: {key}"


# ---------------------------------------------------------------------------
# 7. extract_error_signal honors the event's pre-populated signal slot
# ---------------------------------------------------------------------------


class TestExtractorRespectsPrepopulatedSignal:
    """If loop.py pre-populates ``error_signal`` on the event, the extractor
    must not re-run and silently downgrade it to NONE. This is the v1 bug
    path that let error signals vanish between tool execution and classifier."""

    def test_extractor_returns_prepopulated_non_none(self):
        e = _event(
            tool_name="bash",
            result_success=False,
            result_content="ModuleNotFoundError: no such module",
            error_signal=ErrorSignal.RUNTIME_ERROR,
        )
        assert extract_error_signal(e) == ErrorSignal.RUNTIME_ERROR
