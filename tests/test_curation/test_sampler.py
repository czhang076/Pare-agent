"""Tests for token-budget-matched sampler.

Fixtures are tuned to the Liu v2 ``assign_outcome_label`` contract (post-R4
sampler port):

- ``VERIFIED_ONE_SHOT`` / ``VERIFIED_WITH_RECOVERY`` both require a
  non-empty ``tier2_command`` AND ``tier1_pass=True``/``tier2_pass=True``;
  that's why the default ``_record(tier2_command="pytest")`` is used for
  the ``os-*`` / ``fr-*`` pools.
- ``WEAKLY_VERIFIED`` is the Tier-2-not-configured branch, so ``wv-1``
  explicitly sets ``tier2_command=""``.
- The ``fr-*`` pool needs actually-detectable recovery in
  ``tool_call_events`` — the sampler now delegates to
  ``recovery_detector_v2``, which walks the event list rather than
  ``StepAttempt`` retries.

Deprecation filters on the removed v1 classifier modules are kept defensive
until R5 deletes them.
"""

from __future__ import annotations

import pytest

from pare.curation.sampler import (
    TokenBudgetSampler,
    TokenBudgetSamplerConfig,
    TokenBudgetSamplingError,
)
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
)
from pare.trajectory.schema_v2 import ToolCallEvent

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::DeprecationWarning:pare.trajectory.classifier"
    ),
    pytest.mark.filterwarnings(
        "ignore::DeprecationWarning:pare.trajectory.recovery_detector"
    ),
]


def _record(
    trajectory_id: str,
    *,
    task: str = "Fix bug",
    llm_claimed_success: bool,
    final_passed: bool,
    tier1_pass: bool,
    tier2_pass: bool,
    attempts: list[StepAttempt],
    tokens: int,
    tier2_command: str = "pytest",
    tool_call_events: list[ToolCallEvent] | None = None,
) -> TrajectoryRecord:
    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=trajectory_id,
        instance_id=f"inst-{trajectory_id}",
        task=task,
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
        attempts=attempts,
        tool_call_events=list(tool_call_events or []),
        token_usage=TokenUsageSummary(input_tokens=tokens - 1, output_tokens=1),
        metadata={},
    )


def _recovery_events() -> list[ToolCallEvent]:
    """Build a minimal (error, correction) event pair.

    Shape:
      0: file_edit on a.py fails with a SyntaxError → error_signal=SYNTAX_ERROR
      1: file_edit on a.py succeeds with different params → correction
    Recovery detector returns L1 (same tool + same target, params differ).
    """
    err = ToolCallEvent.create(
        turn_id=0,
        call_index_in_turn=0,
        global_index=0,
        tool_name="file_edit",
        params={"file_path": "a.py", "old_string": "x = 1", "new_string": "x ="},
        result_success=False,
        result_content="SyntaxError: invalid syntax (a.py, line 1)",
        timestamp=1.0,
    )
    fix = ToolCallEvent.create(
        turn_id=1,
        call_index_in_turn=0,
        global_index=1,
        tool_name="file_edit",
        params={"file_path": "a.py", "old_string": "x =", "new_string": "x = 2"},
        result_success=True,
        result_content="edit applied",
        timestamp=2.0,
    )
    return [err, fix]


def _one_shot_attempt(step: int = 1) -> list[StepAttempt]:
    return [
        StepAttempt(
            step_number=step,
            attempt_number=1,
            goal="Implement fix",
            status="success",
            target_files=["a.py"],
            tool_names=["file_edit"],
        )
    ]


def _recovery_attempts(step: int = 1) -> list[StepAttempt]:
    return [
        StepAttempt(
            step_number=step,
            attempt_number=1,
            goal="Implement fix",
            status="failed",
            target_files=["a.py"],
            tool_names=["file_edit"],
            failure_reason="test failed",
        ),
        StepAttempt(
            step_number=step,
            attempt_number=2,
            goal="Implement fix",
            status="success",
            target_files=["a.py"],
            tool_names=["file_edit"],
        ),
    ]


def _dataset() -> list[TrajectoryRecord]:
    return [
        # one_shot_success pool
        _record(
            "os-1",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_one_shot_attempt(),
            tokens=100,
        ),
        _record(
            "os-2",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_one_shot_attempt(),
            tokens=120,
        ),
        _record(
            "os-3",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_one_shot_attempt(),
            tokens=80,
        ),
        # failure_recovery pool — Tier 2 pass + real recovery event pair
        _record(
            "fr-1",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_recovery_attempts(),
            tokens=90,
            tool_call_events=_recovery_events(),
        ),
        _record(
            "fr-2",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_recovery_attempts(),
            tokens=110,
            tool_call_events=_recovery_events(),
        ),
        # weakly_verified + failed for unfiltered
        #   wv-1: Tier 2 not configured → WEAKLY_VERIFIED branch
        _record(
            "wv-1",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=False,
            attempts=_one_shot_attempt(),
            tokens=70,
            tier2_command="",
        ),
        _record(
            "fd-1",
            llm_claimed_success=False,
            final_passed=False,
            tier1_pass=True,
            tier2_pass=False,
            attempts=[
                StepAttempt(
                    step_number=1,
                    attempt_number=1,
                    goal="Implement fix",
                    status="failed",
                    target_files=["b.py"],
                    tool_names=["bash"],
                    failure_reason="budget exhausted",
                )
            ],
            tokens=60,
        ),
        # toxic (should be excluded from unfiltered)
        _record(
            "tx-1",
            llm_claimed_success=True,
            final_passed=False,
            tier1_pass=False,
            tier2_pass=False,
            attempts=[],
            tokens=50,
        ),
    ]


class TestTokenBudgetSampler:
    def test_auto_target_and_group_shapes(self):
        sampler = TokenBudgetSampler(
            TokenBudgetSamplerConfig(tolerance_ratio=0.10, seed=7)
        )
        plan = sampler.sample_groups(_dataset())

        assert set(plan.groups.keys()) == {
            "clean_only",
            "mixed",
            "recovery_enriched",
            "unfiltered",
        }
        assert plan.target_tokens == 300

        clean = plan.groups["clean_only"]
        mixed = plan.groups["mixed"]
        enriched = plan.groups["recovery_enriched"]
        unfiltered = plan.groups["unfiltered"]

        assert "failure_recovery" not in clean.label_counts
        assert "toxic" not in unfiltered.label_counts

        # recovery_enriched should contain at least as much recovery proportion as mixed
        mixed_total = sum(mixed.label_counts.values())
        mixed_recovery_ratio = mixed.label_counts.get("failure_recovery", 0) / mixed_total
        enriched_total = sum(enriched.label_counts.values())
        enriched_recovery_ratio = (
            enriched.label_counts.get("failure_recovery", 0) / enriched_total
        )
        assert enriched_recovery_ratio >= mixed_recovery_ratio

    def test_explicit_target(self):
        sampler = TokenBudgetSampler(
            TokenBudgetSamplerConfig(tolerance_ratio=0.10, seed=11)
        )
        plan = sampler.sample_groups(_dataset(), target_tokens=220)

        assert plan.target_tokens == 220
        for group in plan.groups.values():
            assert group.within_tolerance

    def test_target_too_high_raises(self):
        sampler = TokenBudgetSampler(TokenBudgetSamplerConfig(tolerance_ratio=0.10))

        with pytest.raises(TokenBudgetSamplingError, match="capacity"):
            sampler.sample_groups(_dataset(), target_tokens=1_000)

    def test_no_one_shot_raises(self):
        sampler = TokenBudgetSampler()
        records = [
            _record(
                "fr-only",
                llm_claimed_success=True,
                final_passed=True,
                tier1_pass=True,
                tier2_pass=True,
                attempts=_recovery_attempts(),
                tokens=100,
                tool_call_events=_recovery_events(),
            )
        ]
        with pytest.raises(TokenBudgetSamplingError, match="clean_only"):
            sampler.sample_groups(records)

    def test_classification_length_mismatch_raises(self):
        sampler = TokenBudgetSampler()
        records = _dataset()
        with pytest.raises(TokenBudgetSamplingError, match="length"):
            sampler.sample_groups(records, classifications=[])

    def test_deterministic_with_same_seed(self):
        config = TokenBudgetSamplerConfig(tolerance_ratio=0.10, seed=123)
        sampler_a = TokenBudgetSampler(config)
        sampler_b = TokenBudgetSampler(config)

        plan_a = sampler_a.sample_groups(_dataset())
        plan_b = sampler_b.sample_groups(_dataset())

        for name in plan_a.groups:
            assert plan_a.groups[name].trajectory_ids == plan_b.groups[name].trajectory_ids
