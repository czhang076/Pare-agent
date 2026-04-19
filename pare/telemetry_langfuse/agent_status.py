"""Compute ``metadata.agent_status`` for a Langfuse span.

R0 scaffold. Real mapping in W3 Day 3-4. This is the semantic contract
that the W4 Langfuse RFC proposes upstream — keep the value set tight
(four labels, not a free-form string) so the proposal stays defensible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pare_research.trajectory.schema_v2 import ToolCallEvent  # type: ignore[import-not-found]


AgentStatus = Literal["success", "dead_end", "retrying", "error"]


def compute_agent_status(
    event: "ToolCallEvent",
    recovery_label: Literal["L1", "L2", "L3", "none"],
    is_post_divergence_dead_path: bool = False,
) -> AgentStatus:
    """Map (event, recovery context) → ``agent_status``.

    Rules (W3 Day 3-4 to finalise; the four labels are locked):

    - L3 repeated failure        → "dead_end"
    - L1 / L2 local recovery     → "retrying"
    - tool failed, no recovery   → "error"
    - everything else            → "success"

    ``is_post_divergence_dead_path`` is set by the Inspector when it
    can prove (via comparison with a successful sibling trajectory) that
    this step is on a branch that never recovers. Optional override.
    """
    raise NotImplementedError("W3 Day 3-4")
