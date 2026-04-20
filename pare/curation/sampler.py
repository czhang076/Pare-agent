"""Token-budget-matched dataset sampler.

Builds four SFT groups from classified trajectories:
- clean_only
- mixed
- recovery_enriched
- unfiltered

All groups are sampled to the same token budget (within tolerance).

R5 state: auto-classification runs through ``classifier_liu`` +
``error_signal_extractor`` + ``recovery_detector_v2`` (the Liu et al. 2025
pipeline). The v1 classifier module has been deleted; the
``TrajectoryLabel`` / ``ClassificationResult`` types that used to live
there are now defined here because this sampler is their only consumer
— they act as an output alphabet for group selection, not a pipeline
stage. ``_map_outcome_to_label`` bridges Liu's ``OutcomeLabel`` back into
that alphabet so the group-selection logic below stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Sequence

from pare.trajectory.classifier_liu import (
    OutcomeLabel,
    assign_outcome_label,
    classify_liu_from_record,
)
from pare.trajectory.error_signal_extractor import (
    classify_trajectory_signals,
)
from pare.trajectory.recovery_detector_v2 import (
    RecoveryEvent,
    RecoveryLevel,
    detect_recovery_events,
)
from pare.trajectory.schema import TrajectoryRecord


# ---------------------------------------------------------------------------
# Label vocabulary for group selection (formerly in pare.trajectory.classifier;
# moved here when that v1 module was deleted in R5). These types are *not*
# re-exported to the public API — they exist solely to describe sampler
# pools and to support precomputed-label callsites.
# ---------------------------------------------------------------------------


class TrajectoryLabel(str, Enum):
    """Sampler pool labels. Mirrors the v1 classifier's enum verbatim."""

    TOXIC = "toxic"
    FAILED = "failed"
    WEAKLY_VERIFIED = "weakly_verified"
    FULLY_VERIFIED = "fully_verified"
    ONE_SHOT_SUCCESS = "one_shot_success"
    FAILURE_RECOVERY = "failure_recovery"


@dataclass(slots=True)
class ClassificationResult:
    """Backward-compat shim for callers that precomputed v1 labels.

    Only ``primary_label`` is consulted by the sampler; the other fields
    exist so pre-R5 call sites using the old classifier keep type-checking.
    """

    primary_label: TrajectoryLabel
    verification_label: TrajectoryLabel | None = None
    recovery_level: RecoveryLevel | None = None
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# v1 OutcomeLabel → v1 TrajectoryLabel mapping. Kept as a module-level
# constant so it appears exactly once (single-source-of-truth for the
# alignment between Liu et al. outcomes and the sampler's pool vocabulary).
_OUTCOME_TO_V1_LABEL: dict[OutcomeLabel, TrajectoryLabel] = {
    OutcomeLabel.TOXIC: TrajectoryLabel.TOXIC,
    OutcomeLabel.FAILED: TrajectoryLabel.FAILED,
    OutcomeLabel.WEAKLY_VERIFIED: TrajectoryLabel.WEAKLY_VERIFIED,
    OutcomeLabel.VERIFIED_ONE_SHOT: TrajectoryLabel.ONE_SHOT_SUCCESS,
    OutcomeLabel.VERIFIED_WITH_RECOVERY: TrajectoryLabel.FAILURE_RECOVERY,
}


def _map_outcome_to_label(outcome: OutcomeLabel) -> TrajectoryLabel:
    """Translate a Liu v2 outcome into the v1 label the sampler pools on.

    Raises KeyError on unknown values so any future addition to
    ``OutcomeLabel`` forces an explicit audit here rather than silently
    miscategorising trajectories.
    """
    return _OUTCOME_TO_V1_LABEL[outcome]


class TokenBudgetSamplingError(ValueError):
    """Raised when token-matched group sampling cannot satisfy constraints."""


@dataclass(frozen=True, slots=True)
class TokenBudgetSamplerConfig:
    """Sampler behavior config."""

    tolerance_ratio: float = 0.05
    seed: int = 0
    recovery_enrichment_factor: float = 2.0
    max_trials: int = 256
    strict_tolerance: bool = True


@dataclass(slots=True)
class SampledGroup:
    """One sampled dataset group."""

    name: str
    trajectories: list[TrajectoryRecord]
    trajectory_ids: list[str]
    total_tokens: int
    label_counts: dict[str, int]
    within_tolerance: bool


@dataclass(slots=True)
class SamplingPlan:
    """All token-matched groups for one run."""

    target_tokens: int
    tolerance_ratio: float
    groups: dict[str, SampledGroup]


@dataclass(frozen=True, slots=True)
class _TrajectoryItem:
    key: str
    trajectory: TrajectoryRecord
    label: TrajectoryLabel
    tokens: int


class TokenBudgetSampler:
    """Sample token-budget-matched training groups from classified trajectories."""

    def __init__(self, config: TokenBudgetSamplerConfig | None = None) -> None:
        self.config = config or TokenBudgetSamplerConfig()

    def sample_groups(
        self,
        trajectories: Sequence[TrajectoryRecord],
        classifications: Sequence[ClassificationResult] | None = None,
        *,
        target_tokens: int | None = None,
    ) -> SamplingPlan:
        """Build four groups with matched token budgets.

        Args:
            trajectories: Input trajectory records.
            classifications: Optional precomputed classifier output.
            target_tokens: Optional explicit target budget per group.
        """
        records = list(trajectories)
        if not records:
            raise TokenBudgetSamplingError("No trajectories provided.")

        labels = self._resolve_labels(records, classifications)
        items = self._build_items(records, labels)

        one_shot = [it for it in items if it.label == TrajectoryLabel.ONE_SHOT_SUCCESS]
        recovery = [it for it in items if it.label == TrajectoryLabel.FAILURE_RECOVERY]
        non_toxic = [it for it in items if it.label != TrajectoryLabel.TOXIC]

        if not one_shot:
            raise TokenBudgetSamplingError(
                "clean_only group is empty (no one_shot_success trajectories)."
            )
        if not recovery:
            raise TokenBudgetSamplingError(
                "mixed/recovery_enriched groups need at least one failure_recovery trajectory."
            )
        if not non_toxic:
            raise TokenBudgetSamplingError("unfiltered group is empty (all trajectories are toxic).")

        base_rng = random.Random(self.config.seed)
        recovery_enriched_pool = self._make_recovery_enriched_pool(
            one_shot,
            recovery,
            base_rng,
        )

        group_pools: dict[str, list[_TrajectoryItem]] = {
            "clean_only": one_shot,
            "mixed": [*one_shot, *recovery],
            "recovery_enriched": recovery_enriched_pool,
            "unfiltered": non_toxic,
        }

        natural_recovery_ratio = len(recovery) / (len(one_shot) + len(recovery))
        desired_recovery_ratio = min(
            1.0,
            natural_recovery_ratio * self.config.recovery_enrichment_factor,
        )

        capacities = {
            name: sum(item.tokens for item in pool)
            for name, pool in group_pools.items()
        }

        if target_tokens is None:
            target_tokens = min(capacities.values())

        if target_tokens <= 0:
            raise TokenBudgetSamplingError("Target token budget must be positive.")

        for name, cap in capacities.items():
            if cap < target_tokens:
                raise TokenBudgetSamplingError(
                    f"Group '{name}' capacity {cap} is below target {target_tokens}."
                )

        lower, upper = self._token_bounds(target_tokens)
        groups: dict[str, SampledGroup] = {}

        for idx, (name, pool) in enumerate(group_pools.items()):
            group_rng = random.Random(self.config.seed + 10_000 + idx)
            selected, total = self._select_for_target(pool, target_tokens, lower, upper, group_rng)

            if name == "recovery_enriched":
                selected, total = self._enforce_recovery_ratio(
                    selected,
                    pool,
                    desired_recovery_ratio,
                    target_tokens,
                    lower,
                    upper,
                )

            within = lower <= total <= upper
            if self.config.strict_tolerance and not within:
                raise TokenBudgetSamplingError(
                    f"Group '{name}' total {total} is outside tolerance [{lower}, {upper}]."
                )

            label_counts: dict[str, int] = {}
            for item in selected:
                key = item.label.value
                label_counts[key] = label_counts.get(key, 0) + 1

            groups[name] = SampledGroup(
                name=name,
                trajectories=[item.trajectory for item in selected],
                trajectory_ids=[item.trajectory.trajectory_id for item in selected],
                total_tokens=total,
                label_counts=label_counts,
                within_tolerance=within,
            )

        return SamplingPlan(
            target_tokens=target_tokens,
            tolerance_ratio=self.config.tolerance_ratio,
            groups=groups,
        )

    def _resolve_labels(
        self,
        trajectories: list[TrajectoryRecord],
        classifications: Sequence[ClassificationResult] | None,
    ) -> list[TrajectoryLabel]:
        """Return one v1 ``TrajectoryLabel`` per trajectory.

        Two supported input modes:

        - ``classifications is None`` — auto-classify through the Liu v2
          pipeline (``error_signal_extractor`` → ``classifier_liu`` →
          ``recovery_detector_v2`` → ``assign_outcome_label``), then map
          each ``OutcomeLabel`` down to the v1 pool label via
          ``_map_outcome_to_label``. This is the path exercised by the R4+
          flat-ReAct trajectory pipeline.
        - ``classifications`` provided — legacy path. Use ``primary_label``
          verbatim. Kept for backward compat with callers that already
          computed labels against the v1 classifier; will be removed when
          the v1 classifier itself is deleted (post-R5).
        """
        if classifications is not None:
            if len(classifications) != len(trajectories):
                raise TokenBudgetSamplingError(
                    "classifications length must match trajectories length."
                )
            return [result.primary_label for result in classifications]

        labels: list[TrajectoryLabel] = []
        for record in trajectories:
            events = list(record.tool_call_events)
            # If the record was loaded from a pre-v2 JSONL (no tool_call_events),
            # classify_trajectory_signals returns an empty list; the Liu
            # classifier degrades gracefully into "no signal → not toxic"
            # which maps back onto v1 labels via verification fields alone.
            signals = classify_trajectory_signals(events)
            liu = classify_liu_from_record(record, signals)
            # detect_recovery_events emits per-event ``RecoveryEvent``s; the
            # v1 sampler only needs the boolean "contains any recovery",
            # mirroring what the plan's draft used.
            recovery_result = detect_recovery_events(events, signals)
            outcome = assign_outcome_label(
                liu, record.verification, recovery_result.contains_recovery
            )
            labels.append(_map_outcome_to_label(outcome))
        return labels

    def _build_items(
        self,
        trajectories: list[TrajectoryRecord],
        labels: list[TrajectoryLabel],
    ) -> list[_TrajectoryItem]:
        items: list[_TrajectoryItem] = []
        for idx, (trajectory, label) in enumerate(zip(trajectories, labels, strict=True)):
            tokens = max(1, trajectory.token_usage.total_tokens)
            items.append(
                _TrajectoryItem(
                    key=f"{trajectory.trajectory_id}#{idx}",
                    trajectory=trajectory,
                    label=label,
                    tokens=tokens,
                )
            )
        return items

    def _make_recovery_enriched_pool(
        self,
        one_shot: list[_TrajectoryItem],
        recovery: list[_TrajectoryItem],
        rng: random.Random,
    ) -> list[_TrajectoryItem]:
        pool = [*one_shot, *recovery]

        if self.config.recovery_enrichment_factor <= 1.0:
            return pool

        extra_count = int(round(len(recovery) * (self.config.recovery_enrichment_factor - 1.0)))
        for i in range(extra_count):
            base = rng.choice(recovery)
            pool.append(
                _TrajectoryItem(
                    key=f"{base.key}@dup{i}",
                    trajectory=base.trajectory,
                    label=base.label,
                    tokens=base.tokens,
                )
            )
        return pool

    def _select_for_target(
        self,
        pool: list[_TrajectoryItem],
        target: int,
        lower: int,
        upper: int,
        rng: random.Random,
    ) -> tuple[list[_TrajectoryItem], int]:
        if not pool:
            return [], 0

        total_pool = sum(item.tokens for item in pool)
        if lower <= total_pool <= upper:
            return list(pool), total_pool

        best_indices: list[int] = []
        best_total = 0
        best_score = (2, float("inf"), float("inf"))

        for _ in range(self.config.max_trials):
            order = list(range(len(pool)))
            rng.shuffle(order)

            selected: list[int] = []
            selected_set: set[int] = set()
            running = 0

            for idx in order:
                tok = pool[idx].tokens
                if running + tok <= upper:
                    selected.append(idx)
                    selected_set.add(idx)
                    running += tok
                if running >= lower:
                    break

            improved = True
            while improved:
                improved = False
                best_add_idx = -1
                best_add_delta = abs(target - running)
                for idx in order:
                    if idx in selected_set:
                        continue
                    new_total = running + pool[idx].tokens
                    if new_total > upper:
                        continue
                    delta = abs(target - new_total)
                    if delta < best_add_delta:
                        best_add_delta = delta
                        best_add_idx = idx
                if best_add_idx >= 0:
                    selected.append(best_add_idx)
                    selected_set.add(best_add_idx)
                    running += pool[best_add_idx].tokens
                    improved = True

            score = (
                0 if lower <= running <= upper else 1,
                abs(target - running),
                upper - running if running <= upper else running - upper,
            )
            if score < best_score:
                best_score = score
                best_indices = selected
                best_total = running
                if best_score[0] == 0 and best_score[1] == 0:
                    break

        selected_items = [pool[i] for i in best_indices]
        return selected_items, best_total

    def _enforce_recovery_ratio(
        self,
        selected: list[_TrajectoryItem],
        pool: list[_TrajectoryItem],
        desired_ratio: float,
        target: int,
        lower: int,
        upper: int,
    ) -> tuple[list[_TrajectoryItem], int]:
        if not selected:
            return selected, 0

        selected_list = list(selected)
        selected_keys = {item.key for item in selected_list}
        total_tokens = sum(item.tokens for item in selected_list)
        recovery_label = TrajectoryLabel.FAILURE_RECOVERY

        def _ratio(items: list[_TrajectoryItem]) -> float:
            if not items:
                return 0.0
            count = sum(1 for item in items if item.label == recovery_label)
            return count / len(items)

        current_ratio = _ratio(selected_list)
        if current_ratio >= desired_ratio:
            return selected_list, total_tokens

        max_swaps = len(pool) * 2
        for _ in range(max_swaps):
            if current_ratio >= desired_ratio:
                break

            non_recovery = [item for item in selected_list if item.label != recovery_label]
            recovery_outside = [
                item
                for item in pool
                if item.label == recovery_label and item.key not in selected_keys
            ]
            if not non_recovery or not recovery_outside:
                break

            best_pair: tuple[_TrajectoryItem, _TrajectoryItem] | None = None
            best_score = (2, float("inf"), float("inf"))

            for add_item in recovery_outside:
                for remove_item in non_recovery:
                    new_total = total_tokens - remove_item.tokens + add_item.tokens
                    if not (lower <= new_total <= upper):
                        continue

                    # remove non-recovery + add recovery => recovery count +1
                    new_ratio = (sum(
                        1 for item in selected_list if item.label == recovery_label
                    ) + 1) / len(selected_list)

                    score = (
                        0 if new_ratio >= desired_ratio else 1,
                        abs(desired_ratio - new_ratio),
                        abs(target - new_total),
                    )
                    if score < best_score:
                        best_score = score
                        best_pair = (remove_item, add_item)

            if best_pair is None:
                break

            remove_item, add_item = best_pair
            selected_list.remove(remove_item)
            selected_list.append(add_item)
            selected_keys.remove(remove_item.key)
            selected_keys.add(add_item.key)
            total_tokens = total_tokens - remove_item.tokens + add_item.tokens
            current_ratio = _ratio(selected_list)

        return selected_list, total_tokens

    def _token_bounds(self, target: int) -> tuple[int, int]:
        tol = self.config.tolerance_ratio
        lower = int(target * (1.0 - tol))
        upper = int(target * (1.0 + tol))
        return lower, upper


def sample_token_matched_groups(
    trajectories: Sequence[TrajectoryRecord],
    classifications: Sequence[ClassificationResult] | None = None,
    *,
    target_tokens: int | None = None,
    config: TokenBudgetSamplerConfig | None = None,
) -> SamplingPlan:
    """Convenience wrapper for token-budget group sampling."""
    sampler = TokenBudgetSampler(config=config)
    return sampler.sample_groups(
        trajectories,
        classifications,
        target_tokens=target_tokens,
    )
