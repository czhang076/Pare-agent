"""R0 placeholder tests for run_agent flat ReAct loop.

Real tests land in R3. Three CI regression guards from the refactor plan:

- ``test_declared_done_exits``  — fake LLM calls declare_done(status="fixed");
  assert stop_reason == "declared_done" and success == True.
- ``test_budget_exhausted``     — fake LLM spams file_read; assert
  stop_reason == "budget_exhausted" once max_steps is hit.
- ``test_end_turn``             — fake LLM returns no tool_calls; assert
  stop_reason == "end_turn".

All three use a mocked LLMAdapter and a ``MagicMock(spec=InstanceContainer)``
so they run in-process without Docker.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(reason="R0 scaffold — run_agent not implemented yet (R3)")


def test_declared_done_exits() -> None:
    raise NotImplementedError("R3")


def test_budget_exhausted() -> None:
    raise NotImplementedError("R3")


def test_end_turn() -> None:
    raise NotImplementedError("R3")
