"""Agent layer — flat ReAct loop + optional pre-pass switches.

R5 landed the flat-ReAct refactor:

- ``loop.run_agent`` is the only tool-call loop. Orchestrator / Executor /
  Orient / Planner from the 3-layer architecture have been deleted.
- ``LoopConfig`` exposes ``use_orient`` / ``use_planner`` for optional
  pre-passes (ablation switches per plan §5.4.3); pre-passes themselves
  are implemented in ``orient_v2`` / ``planner_v2`` as stubs today.
- ``Guardrails`` is retained as a pure budget / advisory checker; the
  per-step ``reset_step`` sugar that orchestrator owned is gone.
"""

from pare.agent.guardrails import GuardrailConfig, Guardrails
from pare.agent.loop import LoopConfig, LoopResult, run_agent

__all__ = [
    "GuardrailConfig",
    "Guardrails",
    "LoopConfig",
    "LoopResult",
    "run_agent",
]
