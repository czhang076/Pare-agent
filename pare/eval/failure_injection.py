"""Failure-injection harness for Pare agent recovery evaluation.

Design problem
--------------

We want to measure "given a known fault in the environment, can the
agent notice and recover from it?" — separately from "can the agent
solve the task from scratch?" Natural-fault recovery rates are a
*correlational* signal (which tasks happen to need recovery); injected
faults are a *controlled* signal (we choose which fault, when).

MVP contract (this file)
------------------------

1. A fault is a pair of pure functions:
   - ``apply(workdir)``  — mutate the workdir to introduce the fault
   - ``revert(workdir)`` — restore the workdir to its pre-apply state

2. Faults are registered in ``REGISTRY`` and identified by short string
   names (``stale_test_cache``, ``wrong_import``, ...). The registry is
   the CLI's source of truth; adding a new fault = one decorator call.

3. Faults are applied **pre-agent-start** in v0. Mid-trajectory
   injection is deferred to v1 — it requires a loop hook and adds the
   "when is the right moment to inject" experimental-design question.
   Pre-injection is honest: the agent sees the faulted workdir from
   turn 0 and its recorded trajectory shows whether it recovered.

4. ``run_with_fault`` is a thin orchestrator that:
      apply(fault) → run agent → revert(fault) → return result
   The agent runner is passed in as a callback so tests can stub it
   (we absolutely don't want this module to import Docker / an LLM
   provider at import time).

5. Recovery judgement is deliberately NOT baked in here. The caller
   runs ``pare.trajectory.recovery_detector_v2.detect_recovery_events``
   on the returned trajectory and joins with the fault metadata
   post-hoc. This keeps failure_injection orthogonal to the rubric —
   we can change how we score without changing how we generate.

Non-goals (explicit)
--------------------

- **Not a security tool.** We don't simulate prompt injection,
  adversarial outputs, or attacker models. This is a *capability probe*.
- **Not a sandboxing layer.** Faults run against whatever workdir
  the caller hands us. Caller is responsible for isolation.
- **Not a fuzzing harness.** Faults are hand-designed, not randomized —
  we want every injection to be legible in the results table.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Fault abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedFault:
    """A reversible mutation of a workdir that simulates a failure mode.

    Attributes:
        name: Short identifier, used in CLI + result rows. Must be unique
              within the REGISTRY.
        description: Human-readable one-liner. Surfaces in reports and
                     on ``--list-faults``.
        applies_to_liu: Which Liu taxonomy category this fault simulates
                        (e.g. ``"B2.2"`` for a fault that produces a
                        syntax-error scenario). Used post-hoc to check
                        "did the agent classify the fault correctly
                        before fixing it?"
        apply_fn: ``apply_fn(workdir: Path) -> RevertToken`` — mutate
                  the workdir in place, return anything needed to undo
                  later (typically a dict of backed-up paths + contents).
        revert_fn: ``revert_fn(workdir: Path, token) -> None`` — restore
                   the workdir to its pre-apply state using the token
                   returned by apply_fn.
    """

    name: str
    description: str
    applies_to_liu: str
    apply_fn: Callable[[Path], Any]
    revert_fn: Callable[[Path, Any], None]

    def apply(self, workdir: Path) -> Any:
        return self.apply_fn(workdir)

    def revert(self, workdir: Path, token: Any) -> None:
        self.revert_fn(workdir, token)


@dataclass(frozen=True)
class FaultInjectionResult:
    """Outcome of one (task, fault, agent_run) tuple.

    ``trajectory`` is the raw ``TrajectoryRecord.to_dict()`` output from
    the agent runner. We keep it as a dict here rather than parsed so
    this module doesn't pull in the full trajectory schema at import
    time — keeps the dependency graph shallow.
    """

    instance_id: str
    fault_name: str
    applied_at: float
    agent_duration_s: float
    agent_exit_code: int
    trajectory: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "fault_name": self.fault_name,
            "applied_at": self.applied_at,
            "agent_duration_s": self.agent_duration_s,
            "agent_exit_code": self.agent_exit_code,
            "trajectory": dict(self.trajectory),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Fault registry + decorator
# ---------------------------------------------------------------------------


REGISTRY: dict[str, InjectedFault] = {}


def _register(fault: InjectedFault) -> InjectedFault:
    """Add a fault to the global registry, rejecting duplicate names."""
    if fault.name in REGISTRY:
        raise ValueError(f"duplicate fault name: {fault.name!r}")
    REGISTRY[fault.name] = fault
    return fault


# ---------------------------------------------------------------------------
# Concrete faults
# ---------------------------------------------------------------------------


def _apply_stale_test_cache(workdir: Path) -> dict[str, Any]:
    """Create a stale .pytest_cache that claims tests passed last run.

    The scenario: a previous test run cached "nodeids with passing" and
    the agent re-reading that cache without invalidating would trust
    stale green signal. Proper agents re-run pytest rather than trust
    ``.pytest_cache`` unconditionally.

    Revert strategy: the token records whether ``.pytest_cache`` existed
    before we touched it. If we created the tree from nothing we own it
    (``revert`` nukes the whole root); if the tree pre-existed we only
    own the ``nodeids`` file and restore or delete that single file.
    """
    cache_root = workdir / ".pytest_cache"
    root_preexisted = cache_root.exists()

    cache_dir = cache_root / "v" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    nodeids_path = cache_dir / "nodeids"

    token: dict[str, Any] = {
        "root_preexisted": root_preexisted,
        "nodeids_backup": (
            nodeids_path.read_text(encoding="utf-8")
            if nodeids_path.exists()
            else None
        ),
    }

    # Fake cache content — pretend every node passed 12 runs ago.
    nodeids_path.write_text(
        json.dumps(
            [
                "tests/test_stale.py::test_that_does_not_exist",
                "tests/test_stale.py::test_wishful_thinking",
            ]
        ),
        encoding="utf-8",
    )
    return token


def _revert_stale_test_cache(workdir: Path, token: Any) -> None:
    assert isinstance(token, dict)
    cache_root = workdir / ".pytest_cache"

    if not token.get("root_preexisted", False):
        if cache_root.exists():
            shutil.rmtree(cache_root, ignore_errors=True)
        return

    nodeids_path = cache_root / "v" / "cache" / "nodeids"
    backup = token.get("nodeids_backup")
    if backup is not None:
        nodeids_path.write_text(backup, encoding="utf-8")
    elif nodeids_path.exists():
        nodeids_path.unlink()


STALE_TEST_CACHE = _register(
    InjectedFault(
        name="stale_test_cache",
        description=(
            "Plant a .pytest_cache with fake 'last-run-passed' nodeids "
            "for tests that don't exist. Probes whether the agent re-runs "
            "pytest or trusts the stale cache."
        ),
        applies_to_liu="C2",  # premature success via stale signal
        apply_fn=_apply_stale_test_cache,
        revert_fn=_revert_stale_test_cache,
    )
)


def _is_non_test_python_source(path: Path, workdir: Path) -> bool:
    """Precise filter for the wrong_import target search.

    Avoids two false-positive modes a substring match hits:
    - Windows tmp dirs named like ``test_xxx`` (host-path contamination)
    - Legit source files whose *parent directory* contains ``test`` as
      a substring but isn't a test directory (e.g. ``contest/``).

    Rules:
    - exclude anything under a ``.git``, ``test``, or ``tests`` directory
      (case-insensitive) **relative to workdir**
    - exclude files whose basename starts with ``test_`` or ends in
      ``_test.py`` (Python test-discovery conventions)
    """
    try:
        rel_parts = path.relative_to(workdir).parts
    except ValueError:
        return False
    lowered = [part.lower() for part in rel_parts]
    if any(part in (".git", "test", "tests") for part in lowered):
        return False
    name = path.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return False
    return True


def _apply_wrong_import(workdir: Path) -> dict[str, Any]:
    """Inject a bogus import at the top of a Python file in the workdir.

    We pick the first non-test ``.py`` file to avoid breaking test
    collection. The agent should see ``ModuleNotFoundError`` on any
    script execution and is expected to either remove the import or
    identify it as the root cause.
    """
    candidates = sorted(
        p for p in workdir.rglob("*.py") if _is_non_test_python_source(p, workdir)
    )
    if not candidates:
        raise RuntimeError(
            f"wrong_import: no non-test .py file found under {workdir}"
        )
    target = candidates[0]
    original = target.read_text(encoding="utf-8")
    poisoned = f"import _pare_synthetic_missing_module  # INJECTED\n" + original
    target.write_text(poisoned, encoding="utf-8")
    return {"target": str(target), "original": original}


def _revert_wrong_import(workdir: Path, token: Any) -> None:
    assert isinstance(token, dict)
    Path(token["target"]).write_text(token["original"], encoding="utf-8")


WRONG_IMPORT = _register(
    InjectedFault(
        name="wrong_import",
        description=(
            "Prepend `import _pare_synthetic_missing_module` to the first "
            "non-test .py file. Probes whether the agent identifies a "
            "ModuleNotFoundError root cause before attempting other edits."
        ),
        applies_to_liu="B2.2",  # broken import = effectively a parse/exec error
        apply_fn=_apply_wrong_import,
        revert_fn=_revert_wrong_import,
    )
)


def _apply_empty_edit_baseline(workdir: Path) -> dict[str, Any]:
    """No-op fault — the 'sanity baseline' arm.

    Exists so the runner table has a row where we expect **zero**
    injected-recovery signal. If the agent recovers here, something
    is wrong with our definition of "recovery" — i.e., this is the
    false-positive detector.
    """
    return {}


def _revert_empty_edit_baseline(workdir: Path, token: Any) -> None:
    pass


EMPTY_BASELINE = _register(
    InjectedFault(
        name="empty_baseline",
        description=(
            "No-op fault; sanity arm. A positive recovery signal here "
            "indicates a false positive in the recovery detector."
        ),
        applies_to_liu="",  # no category
        apply_fn=_apply_empty_edit_baseline,
        revert_fn=_revert_empty_edit_baseline,
    )
)


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------


def apply_fault(fault_name: str, workdir: Path) -> Any:
    """Apply the named fault to ``workdir`` and return the revert token."""
    if fault_name not in REGISTRY:
        raise KeyError(
            f"unknown fault {fault_name!r}; known: {sorted(REGISTRY)}"
        )
    return REGISTRY[fault_name].apply(workdir)


def revert_fault(fault_name: str, workdir: Path, token: Any) -> None:
    """Revert the named fault using the token returned by apply_fault."""
    REGISTRY[fault_name].revert(workdir, token)


# Type alias for the agent callback. Intentionally permissive: callers
# pass either a sync wrapper around run_headless_flat_react, or a stub
# in tests. We require just enough contract to produce a result row.
AgentRunner = Callable[[str, Path], tuple[int, dict[str, Any]]]
"""``(instance_id, workdir) -> (exit_code, trajectory_dict)``"""


def run_with_fault(
    *,
    fault_name: str,
    instance_id: str,
    workdir: Path,
    agent_runner: AgentRunner,
) -> FaultInjectionResult:
    """Apply a fault, run the agent, revert the fault, return the result.

    The revert runs even if the agent raises — workdirs often belong to
    git worktrees shared across runs and we absolutely cannot leave
    faults in place.

    The ``agent_runner`` callback is the seam where tests inject a
    stub. In production, the CLI wraps ``run_headless_flat_react`` so
    this module itself never imports Docker or LLM providers.
    """
    if fault_name not in REGISTRY:
        raise KeyError(
            f"unknown fault {fault_name!r}; known: {sorted(REGISTRY)}"
        )

    fault = REGISTRY[fault_name]
    applied_at = time.time()
    token = fault.apply(workdir)

    start = time.time()
    exit_code = -1
    trajectory: dict[str, Any] = {}
    error_msg = ""
    try:
        exit_code, trajectory = agent_runner(instance_id, workdir)
    except Exception as e:  # noqa: BLE001 — we need to revert no matter what
        error_msg = f"{type(e).__name__}: {e}"
    finally:
        # Revert always. A half-reverted workdir is worse than an
        # unreverted one, but both are better than a silent leftover.
        try:
            fault.revert(workdir, token)
        except Exception as e:  # noqa: BLE001
            error_msg = (
                (error_msg + "; " if error_msg else "")
                + f"revert_failed: {type(e).__name__}: {e}"
            )

    return FaultInjectionResult(
        instance_id=instance_id,
        fault_name=fault_name,
        applied_at=applied_at,
        agent_duration_s=time.time() - start,
        agent_exit_code=exit_code,
        trajectory=trajectory,
        error=error_msg,
    )
