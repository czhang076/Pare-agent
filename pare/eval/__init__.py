"""Evaluation utilities: failure injection + κ helpers.

Kept separate from ``pare.trajectory`` (the passive data-model layer)
because these modules *act* on a workdir — they mutate files, run
agents, and measure outcomes. Mixing them into trajectory would blur
the "data vs. experiment" separation we care about.
"""

from pare.eval.failure_injection import (
    FaultInjectionResult,
    InjectedFault,
    apply_fault,
    revert_fault,
    run_with_fault,
    REGISTRY,
)

__all__ = [
    "FaultInjectionResult",
    "InjectedFault",
    "REGISTRY",
    "apply_fault",
    "revert_fault",
    "run_with_fault",
]
