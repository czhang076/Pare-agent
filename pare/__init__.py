"""Pare 2.0 — opinionated coding-agent observability layer on top of Langfuse.

See plan.md. Modules:

- ``pare.inspector``: Trajectory Inspector CLI (``pare inspect``). The MVP
  product. Consumes SWE-bench style JSONL trajectories, classifies failures
  with the research branch's Liu et al. classifier, and renders HTML reports
  with side-by-side success-vs-failure diffs and divergence-point highlight.

- ``pare.telemetry_langfuse``: W3 deliverable. Emits Pare runtime telemetry
  to Langfuse spans with the ``metadata.agent_status`` semantics that W4
  proposes upstream.

- ``pare.ci``: Post-MVP. Eval-Gated Prompt CI webhook server backed by
  DiffVerify (bidirectional test validation).
"""

__version__ = "0.2.0"
