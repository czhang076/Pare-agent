"""Two-trajectory alignment and divergence detection.

R0 scaffold — types and signatures only. Real implementation in W1 Day 3-5.

Algorithm (locked, do not change without RFC):

    Strict positional alignment, no LCS / DP. Walk i=j=0; at each step
    compare ``(tool_name, target_file, params_hash)`` triples and tag
    by strongest matching prefix. Advance both indices together, even
    when divergence kind drops to TOOL_DIFF.

    Why not LCS: agents don't ``insert/delete`` steps the way DNA does.
    Two trajectories that diverge at step 7 might coincidentally both
    call ``file_read("README.md")`` at step 11. LCS would re-align those
    and report a misleadingly late divergence. Strict positional
    alignment matches user intent: "where did the failed run first do
    something different from the successful run?"
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pare_research.trajectory.schema_v2 import ToolCallEvent  # type: ignore[import-not-found]

    from pare.inspector.annotator import AnnotatedTrajectory


class DivergenceKind(str, enum.Enum):
    IDENTICAL    = "identical"     # params_hash matches exactly
    INTENT_MATCH = "intent_match"  # same tool + target_file, params differ
    TARGET_DRIFT = "target_drift"  # same tool, different target_file
    TOOL_DIFF    = "tool_diff"     # different tool entirely
    LENGTH_DIFF  = "length_diff"   # one trajectory ran out of steps


@dataclass(frozen=True, slots=True)
class AlignedStep:
    index_a: int | None
    index_b: int | None
    kind: DivergenceKind


@dataclass(frozen=True, slots=True)
class DivergencePoint:
    """The first non-IDENTICAL step in the alignment, with semantics.

    ``aligned_index`` is the position in the alignment table, NOT in
    either source trajectory. To map back to source positions read
    ``step_a.global_index`` / ``step_b.global_index``.

    INTENT_MATCH alone does NOT produce a DivergencePoint unless every
    step in the alignment is INTENT_MATCH (in which case we report the
    first one as a "parametric divergence" — both trajectories did the
    same things in the same order with subtly different parameters).
    """

    aligned_index: int
    step_a: "ToolCallEvent | None"
    step_b: "ToolCallEvent | None"
    kind: DivergenceKind
    semantic_tag_a: str
    semantic_tag_b: str
    detail: str


def align(a: list["ToolCallEvent"], b: list["ToolCallEvent"]) -> list[AlignedStep]:
    """Strict positional alignment, see module docstring for rationale."""
    raise NotImplementedError("W1 Day 3")


def find_divergence(
    aligned: list[AlignedStep],
    annotated_a: "AnnotatedTrajectory",
    annotated_b: "AnnotatedTrajectory",
) -> DivergencePoint | None:
    """First non-IDENTICAL aligned step, decorated with Liu tags.

    Returns ``None`` only when both trajectories are bit-identical — vanishingly
    rare in practice but a real possibility for golden tests.
    """
    raise NotImplementedError("W1 Day 4-5")
