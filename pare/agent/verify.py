"""Tier-2 verification result type.

R5 state: the rest of this module (``syntax_check`` / ``git_diff_check`` /
``run_tier2_check``) belonged to the legacy 3-layer agent, which has been
deleted. ``Tier2CheckResult`` is retained because ``pare.sandbox.docker_eval``
still emits it as the canonical Tier-2 result shape — the trajectory
schema and downstream consumers reference this dataclass contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tier2CheckResult:
    """Result of Tier-2 verification (SWE-bench eval inside InstanceContainer)."""

    enabled: bool
    command: str = ""
    passed: bool = False
    return_code: int | None = None
    output: str = ""
    error: str = ""
