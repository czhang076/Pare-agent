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
   names (``fake_test_success``, ``wrong_import``, ...). The registry is
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

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Fault abstraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedFault:
    """A reversible mutation that simulates a failure mode.

    Two parallel API surfaces — host-mode (sync, Path-based) and
    container-mode (async, container-like). Both produce the same kind
    of revert token; both must round-trip byte-perfect when applied to
    a fresh workdir. The orchestrator (``run_with_fault`` or
    ``run_with_fault_in_container``) picks one path per call site.

    Why two surfaces:
    - host-mode is fast, no Docker, used by unit tests and host-side
      smoke runs
    - container-mode is what the actual SWE-bench failure-injection
      sweep needs — the agent runs inside an ``InstanceContainer`` and
      its filesystem view is entirely inside that container

    A fault must implement at least one. ``apply_in_container_fn=None``
    means "not yet ported to container mode"; the orchestrator will
    raise loudly rather than silently fall back to host-mode (which
    would produce a fault that's invisible to the agent).

    Attributes:
        name: Short identifier, used in CLI + result rows. Must be unique
              within the REGISTRY.
        description: Human-readable one-liner.
        applies_to_liu: Liu taxonomy category this fault simulates
                        (e.g. ``"B2.2"`` for a missing-import fault).
        apply_fn: ``(workdir: Path) -> RevertToken`` — host-mode apply.
        revert_fn: ``(workdir: Path, token) -> None`` — host-mode revert.
        apply_in_container_fn: ``async (container) -> RevertToken``;
            ``container`` is duck-typed: ``exec/read_file/write_file/
            workdir``. ``None`` = not ported. The container's
            ``workdir`` is the path inside the container (typically
            ``/testbed``) — apply must operate relative to it.
        revert_in_container_fn: ``async (container, token) -> None``.
    """

    name: str
    description: str
    applies_to_liu: str
    apply_fn: Callable[[Path], Any]
    revert_fn: Callable[[Path, Any], None]
    apply_in_container_fn: Callable[[Any], Awaitable[Any]] | None = None
    revert_in_container_fn: Callable[[Any, Any], Awaitable[None]] | None = None

    def apply(self, workdir: Path) -> Any:
        return self.apply_fn(workdir)

    def revert(self, workdir: Path, token: Any) -> None:
        self.revert_fn(workdir, token)

    async def apply_in_container(self, container: Any) -> Any:
        if self.apply_in_container_fn is None:
            raise NotImplementedError(
                f"fault {self.name!r} has no container-mode apply. "
                "Run in host mode, or port the fault to container mode."
            )
        return await self.apply_in_container_fn(container)

    async def revert_in_container(self, container: Any, token: Any) -> None:
        if self.revert_in_container_fn is None:
            raise NotImplementedError(
                f"fault {self.name!r} has no container-mode revert."
            )
        await self.revert_in_container_fn(container, token)


@dataclass(frozen=True)
class FaultInjectionResult:
    """Outcome of one (task, fault, agent_run) tuple.

    ``trajectory`` is the raw ``TrajectoryRecord.to_dict()`` output from
    the agent runner. We keep it as a dict here rather than parsed so
    this module doesn't pull in the full trajectory schema at import
    time — keeps the dependency graph shallow.

    ``agent_exit_code`` sentinel contract:
        - ``int``  — the agent_runner returned this exit code (any
                    integer, including negative numbers like ``-1`` from
                    subprocess wrappers)
        - ``None`` — the agent_runner raised an exception before it
                    could return. ``error`` carries the traceback
                    summary in that case.

        Use ``agent_exit_code is None`` (not ``== -1``) to detect
        runner failures. The old ``-1`` sentinel was ambiguous because
        real subprocesses can legitimately exit ``-1``.
    """

    instance_id: str
    fault_name: str
    applied_at: float
    agent_duration_s: float
    agent_exit_code: int | None
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


_TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def _iter_test_files(workdir: Path):
    """Yield real test files under workdir, skipping .git / venv / build dirs."""
    seen: set[Path] = set()
    for pattern in _TEST_FILE_PATTERNS:
        for p in workdir.rglob(pattern):
            if p in seen or not p.is_file():
                continue
            try:
                rel_parts = p.relative_to(workdir).parts
            except ValueError:
                continue
            lowered = [part.lower() for part in rel_parts]
            if any(
                part in (".git", "node_modules", ".venv", "venv", "build", "dist")
                for part in lowered
            ):
                continue
            seen.add(p)
            yield p


def _apply_fake_test_success(workdir: Path) -> dict[str, Any]:
    """Rewrite every ``test_*`` / ``*_test`` function body to ``pass``.

    A direct C2 Premature Success probe: after this fault, running
    ``pytest`` reports all tests as PASSED even though no real
    assertions execute. A naive agent will run pytest, see green, and
    declare victory — the trajectory then carries a forged signal that
    classifier_liu's C2 detector ought to catch.

    This replaces the earlier ``stale_test_cache`` fault, which had a
    design flaw under Pare's prompts: ``.pytest_cache/nodeids`` is a
    collection cache, only consulted for ``--last-failed`` / ``-ff``,
    which Pare's agent doesn't use. The fault was effectively a no-op
    against the actual agent. ``fake_test_success`` is observable on
    every ``pytest`` invocation.

    Why AST-rewrite rather than monkey-patch via ``conftest.py``:
    - A planted conftest.py with ``pytest_collection_modifyitems`` is
      easy for a careful agent to ``cat`` and notice; rewriting test
      bodies in-place looks like the test files themselves were
      weakened — closer to the natural C2 pattern (agent edits tests
      to make them pass) we're probing for.
    - AST round-trip preserves docstrings and module-level imports, so
      the token is just ``{path: original_content}`` per file. Revert
      is a straight string write-back — byte-perfect when paired with
      ``ast.unparse`` deterministic output.

    Raises ``RuntimeError`` if no test files are found — a workdir
    with no tests can't carry a meaningful C2 signal, and we prefer
    loud failure over silent no-op (the same posture wrong_import
    takes when no non-test .py file exists).
    """
    import ast

    backups: dict[str, str] = {}
    for p in _iter_test_files(workdir):
        try:
            original = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(original)
        except SyntaxError:
            # Pre-existing syntax error — leave file untouched.
            continue
        rewrote = False
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                # Body becomes a single Pass. ``lineno`` on the new node
                # is intentionally not set — ast.unparse handles that.
                node.body = [ast.Pass()]
                rewrote = True
        if not rewrote:
            continue
        new_src = ast.unparse(tree)
        # Preserve trailing newline if original had one — small but
        # makes the revert byte-comparison test friendlier.
        if original.endswith("\n") and not new_src.endswith("\n"):
            new_src += "\n"
        p.write_text(new_src, encoding="utf-8")
        backups[str(p)] = original

    if not backups:
        raise RuntimeError(
            f"fake_test_success: no test_*.py / *_test.py file found under "
            f"{workdir} (or all candidates failed to parse)"
        )
    return {"backups": backups}


def _revert_fake_test_success(workdir: Path, token: Any) -> None:
    assert isinstance(token, dict)
    for path, content in token.get("backups", {}).items():
        Path(path).write_text(content, encoding="utf-8")


# --- container-mode parallel implementations -------------------------------


_CONTAINER_EXCLUDE_DIR_PARTS = (
    "/.git/", "/node_modules/", "/.venv/", "/venv/", "/build/", "/dist/",
)


def _container_path_excluded(path: str) -> bool:
    """True if any path part matches the universal exclude list."""
    p = path.replace("\\", "/")
    return any(part in p for part in _CONTAINER_EXCLUDE_DIR_PARTS)


async def _list_test_files_in_container(container: Any) -> list[str]:
    """Enumerate test files inside the container under container.workdir."""
    wd = container.workdir
    # find … -path '*/.git/*' -prune is the standard exclusion idiom,
    # but easier to filter in Python afterwards — keeps the find
    # invocation simple and skip rules in one place.
    r = await container.exec(
        f"find {wd} -type f \\( -name 'test_*.py' -o -name '*_test.py' \\)",
        timeout=60.0,
    )
    if r.exit_code != 0:
        # Empty repo / find unavailable — surface as no targets.
        return []
    paths = [
        line.strip()
        for line in r.stdout.splitlines()
        if line.strip() and not _container_path_excluded(line)
    ]
    return sorted(paths)


async def _apply_fake_test_success_in_container(
    container: Any,
) -> dict[str, Any]:
    """Container-mode parallel of ``_apply_fake_test_success``.

    Walks every test_*.py / *_test.py under ``container.workdir``,
    AST-rewrites each ``def test_*`` body to ``pass``, and writes the
    file back inside the container. Returns ``{"backups": {path:
    original_content}}`` for byte-perfect revert.

    Implementation notes:
    - We do the AST parse + unparse on the **host** Python (which lives
      in the calling process); this is fine because Python's grammar is
      the same regardless of where the source originated. The risk is
      that container-side Python may have version-specific syntax our
      host AST doesn't recognize — for those files we leave the source
      untouched (matching the host-mode "skip on SyntaxError" rule).
    - ``container.read_file`` already truncates at 1MB by default; we
      raise that ceiling explicitly because we need byte-perfect revert
      and a truncation that silently dropped trailing test functions
      would break that contract.
    """
    import ast

    paths = await _list_test_files_in_container(container)
    backups: dict[str, str] = {}
    for path in paths:
        try:
            original = await container.read_file(path, max_bytes=8_000_000)
        except Exception:
            continue
        # Defensive: if the original was truncated (we'd see the marker
        # appended by InstanceContainer.read_file), skip — we can't
        # round-trip what we can't fully see.
        if "[truncated at" in original.splitlines()[-1:][0:1] or False:
            continue
        try:
            tree = ast.parse(original)
        except SyntaxError:
            continue
        rewrote = False
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.startswith("test_"):
                node.body = [ast.Pass()]
                rewrote = True
        if not rewrote:
            continue
        new_src = ast.unparse(tree)
        if original.endswith("\n") and not new_src.endswith("\n"):
            new_src += "\n"
        await container.write_file(path, new_src)
        backups[path] = original

    if not backups:
        raise RuntimeError(
            f"fake_test_success: no test_*.py / *_test.py file found under "
            f"{container.workdir} (or all candidates failed to parse)"
        )
    return {"backups": backups}


async def _revert_fake_test_success_in_container(
    container: Any, token: Any
) -> None:
    assert isinstance(token, dict)
    for path, content in token.get("backups", {}).items():
        await container.write_file(path, content)


FAKE_TEST_SUCCESS = _register(
    InjectedFault(
        name="fake_test_success",
        description=(
            "AST-rewrite every test_*.py / *_test.py: replace each "
            "`def test_*` body with `pass`. After this, pytest reports "
            "all tests as PASSED with no real assertions executed. "
            "Probes C2 Premature Success — does the agent verify the "
            "fix or trust pytest's green exit code?"
        ),
        applies_to_liu="C2",
        apply_fn=_apply_fake_test_success,
        revert_fn=_revert_fake_test_success,
        apply_in_container_fn=_apply_fake_test_success_in_container,
        revert_in_container_fn=_revert_fake_test_success_in_container,
    )
)


_WRONG_IMPORT_EXCLUDED_BASENAMES = frozenset({
    "conftest.py",    # pytest collection bootstrap — breaking this changes
                      # the fault mode from "import error at exec" to
                      # "pytest collection error", not the B2.2 scenario
                      # we advertise.
    "setup.py",       # build entry point; poisoning it breaks `pip install
                      # -e .` before the agent can even read the repo.
    "__init__.py",    # package marker; typically near-empty and breaking
                      # it cascades into *every* submodule import.
})


def _is_non_test_python_source(path: Path, workdir: Path) -> bool:
    """Precise filter for the wrong_import target search.

    Avoids false-positive target modes:
    - Windows tmp dirs named like ``test_xxx`` (host-path contamination)
    - Legit source files whose *parent directory* contains ``test`` as
      a substring but isn't a test directory (e.g. ``contest/``)
    - Repo-bootstrap files (``conftest.py``, ``setup.py``, ``__init__.py``)
      whose failure mode isn't the B2.2 ModuleNotFoundError we want to
      probe — see ``_WRONG_IMPORT_EXCLUDED_BASENAMES``.

    Rules:
    - exclude anything under a ``.git``, ``test``, or ``tests`` directory
      (case-insensitive) **relative to workdir**
    - exclude files whose basename starts with ``test_`` or ends in
      ``_test.py`` (Python test-discovery conventions)
    - exclude basenames in ``_WRONG_IMPORT_EXCLUDED_BASENAMES``
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
    if name in _WRONG_IMPORT_EXCLUDED_BASENAMES:
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


# --- container-mode parallel for wrong_import ------------------------------


def _container_basename(path: str) -> str:
    """POSIX basename without importing posixpath (lighter)."""
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


async def _apply_wrong_import_in_container(
    container: Any,
) -> dict[str, Any]:
    """Container-mode parallel of ``_apply_wrong_import``.

    Same selection rules as host mode: skip ``test_*.py`` / ``*_test.py``,
    skip files under ``test/`` / ``tests/`` / ``.git/``, skip bootstrap
    files (``conftest.py``, ``setup.py``, ``__init__.py``). Pick the
    lexicographically-first non-test ``.py`` and prepend the poisoned
    import line.

    The discovery is done via a single ``find`` call inside the
    container; filtering happens on the result list to keep the rules
    in one place (mirrors host-mode ``_is_non_test_python_source``).
    """
    wd = container.workdir
    r = await container.exec(
        f"find {wd} -type f -name '*.py'",
        timeout=60.0,
    )
    if r.exit_code != 0:
        raise RuntimeError(
            f"wrong_import: find failed in container: {r.stderr.strip()}"
        )

    def _ok(path: str) -> bool:
        if not path:
            return False
        norm = path.replace("\\", "/")
        # Exclude .git / test / tests directory segments (case-insensitive).
        lowered = norm.lower()
        if "/.git/" in lowered or "/test/" in lowered or "/tests/" in lowered:
            return False
        base = _container_basename(path)
        if base.startswith("test_") or base.endswith("_test.py"):
            return False
        if base in _WRONG_IMPORT_EXCLUDED_BASENAMES:
            return False
        return True

    candidates = sorted(p.strip() for p in r.stdout.splitlines() if _ok(p))
    if not candidates:
        raise RuntimeError(
            f"wrong_import: no non-test .py file found under "
            f"{container.workdir}"
        )
    target = candidates[0]
    original = await container.read_file(target, max_bytes=8_000_000)
    poisoned = "import _pare_synthetic_missing_module  # INJECTED\n" + original
    await container.write_file(target, poisoned)
    return {"target": target, "original": original}


async def _revert_wrong_import_in_container(
    container: Any, token: Any
) -> None:
    assert isinstance(token, dict)
    await container.write_file(token["target"], token["original"])


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
        apply_in_container_fn=_apply_wrong_import_in_container,
        revert_in_container_fn=_revert_wrong_import_in_container,
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


async def _apply_empty_edit_baseline_in_container(
    container: Any,
) -> dict[str, Any]:
    return {}


async def _revert_empty_edit_baseline_in_container(
    container: Any, token: Any
) -> None:
    return None


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
        apply_in_container_fn=_apply_empty_edit_baseline_in_container,
        revert_in_container_fn=_revert_empty_edit_baseline_in_container,
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


async def apply_fault_in_container(fault_name: str, container: Any) -> Any:
    """Container-mode parallel to ``apply_fault``.

    ``container`` is duck-typed: must expose
    ``workdir`` / ``exec`` / ``read_file`` / ``write_file``.
    ``InstanceContainer`` and ``HostContainer`` both satisfy this.
    Raises ``KeyError`` for unknown fault names, ``NotImplementedError``
    if the fault hasn't been ported to container mode yet.
    """
    if fault_name not in REGISTRY:
        raise KeyError(
            f"unknown fault {fault_name!r}; known: {sorted(REGISTRY)}"
        )
    return await REGISTRY[fault_name].apply_in_container(container)


async def revert_fault_in_container(
    fault_name: str, container: Any, token: Any
) -> None:
    """Container-mode parallel to ``revert_fault``."""
    await REGISTRY[fault_name].revert_in_container(container, token)


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
    exit_code: int | None = None
    trajectory: dict[str, Any] = {}
    error_msg = ""
    try:
        exit_code, trajectory = agent_runner(instance_id, workdir)
    except Exception as e:  # noqa: BLE001 — we need to revert no matter what
        error_msg = f"{type(e).__name__}: {e}"
        # exit_code stays None — the sentinel for "runner raised".
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


# Container-mode async runner signature. The container instance is
# passed through so the caller's agent_runner can use it (e.g. to
# invoke run_agent with the same container the fault was applied to).
ContainerAgentRunner = Callable[
    [str, Any], Awaitable[tuple[int, dict[str, Any]]]
]
"""``async (instance_id, container) -> (exit_code, trajectory_dict)``"""


async def run_with_fault_in_container(
    *,
    fault_name: str,
    instance_id: str,
    container: Any,
    agent_runner: ContainerAgentRunner,
) -> FaultInjectionResult:
    """Container-mode parallel to ``run_with_fault``.

    Apply the fault inside ``container``, run the agent against the
    same container, revert inside the container. Same revert-always
    contract as the host-mode orchestrator.

    The container's lifecycle (build / start / stop) is the **caller's**
    responsibility — typically the caller does
    ``async with InstanceContainer.build(...) as c`` and then calls
    this with ``container=c``. We don't manage it here because building
    a container per (fault, task, seed) tuple is wasteful when the
    same task gets several faults applied to it.
    """
    if fault_name not in REGISTRY:
        raise KeyError(
            f"unknown fault {fault_name!r}; known: {sorted(REGISTRY)}"
        )

    fault = REGISTRY[fault_name]
    applied_at = time.time()
    token = await fault.apply_in_container(container)

    start = time.time()
    exit_code: int | None = None
    trajectory: dict[str, Any] = {}
    error_msg = ""
    try:
        exit_code, trajectory = await agent_runner(instance_id, container)
    except Exception as e:  # noqa: BLE001 — revert no matter what
        error_msg = f"{type(e).__name__}: {e}"
        # exit_code stays None.
    finally:
        try:
            await fault.revert_in_container(container, token)
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
