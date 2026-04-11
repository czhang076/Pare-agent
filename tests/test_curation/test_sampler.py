"""Tests for token-budget-matched sampler."""

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
            tier2_command="",
        ),
        attempts=attempts,
        token_usage=TokenUsageSummary(input_tokens=tokens - 1, output_tokens=1),
        metadata={},
    )


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
            rolled_back=True,
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
        # failure_recovery pool
        _record(
            "fr-1",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_recovery_attempts(),
            tokens=90,
        ),
        _record(
            "fr-2",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=True,
            attempts=_recovery_attempts(),
            tokens=110,
        ),
        # weakly_verified + failed for unfiltered
        _record(
            "wv-1",
            llm_claimed_success=True,
            final_passed=True,
            tier1_pass=True,
            tier2_pass=False,
            attempts=_one_shot_attempt(),
            tokens=70,
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
                    rolled_back=False,
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
