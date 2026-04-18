"""Docker-backed Tier-2 verifier using SWE-bench's official harness.

Pare's host-venv Tier-2 cannot cover SWE-bench instances whose base_commit
predates Python 3.10 (sympy, astropy, django <3, ...) while Pare itself
requires Python >=3.12. This module delegates the actual pytest run to
swebench.harness.run_evaluation.run_instance inside a per-instance Docker
image, then maps the harness report back onto Pare's existing
Tier2CheckResult contract so the trajectory/export pipeline is unchanged.

All heavy imports (swebench, datasets, docker) are lazy and guarded so a
default Pare install without the docker-eval extra keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any

from pare.agent.verify import Tier2CheckResult

logger = logging.getLogger(__name__)


_DOCKER_EXTRA_HINT = (
    "Docker tier-2 requires the 'docker-eval' extra. "
    "Install via: pip install -e \".[docker-eval]\""
)


def _strip_pare_internal_paths(diff: str) -> str:
    """Drop hunks touching Pare's own bookkeeping files (``.pare/*``).

    ``GitCheckpoint.get_full_diff()`` records everything the agent modified
    under the workspace, including Pare's own MEMORY.md / history.jsonl which
    live in ``.pare/``. Those files don't exist inside the SWE-bench image, so
    ``git apply`` (and every fallback) fails with "can't find file to patch"
    and no report.json is written. We keep ``final_diff`` in the trajectory
    untouched — Module A/B still see the full picture — and hand only the
    real source hunks to the harness.
    """
    if not diff:
        return diff
    # Split keeping the "diff --git " prefix with each hunk. The first chunk
    # before the first "diff --git " is preamble (usually empty); preserve it.
    parts = diff.split("\ndiff --git ")
    kept: list[str] = []
    for idx, part in enumerate(parts):
        header = part if idx == 0 else "diff --git " + part
        # A unified-diff hunk header looks like:
        #   diff --git a/<path> b/<path>
        first_line = header.split("\n", 1)[0]
        tokens = first_line.split()
        # tokens == ["diff", "--git", "a/<path>", "b/<path>"]
        if len(tokens) >= 4:
            a_path = tokens[2][2:] if tokens[2].startswith("a/") else tokens[2]
            b_path = tokens[3][2:] if tokens[3].startswith("b/") else tokens[3]
            if a_path.startswith(".pare/") or b_path.startswith(".pare/"):
                continue
        kept.append(header)
    if not kept:
        return ""
    # Rejoin: first element already contains its own prefix (or is preamble);
    # subsequent ones already carry "diff --git ". Separator is just "\n".
    out = kept[0]
    for chunk in kept[1:]:
        if not out.endswith("\n"):
            out += "\n"
        out += chunk
    return out


class DockerEvalUnavailable(RuntimeError):
    """Raised when the docker-eval extra is not installed or the daemon is unreachable."""


@dataclass(frozen=True, slots=True)
class DockerEvalConfig:
    """Parameters controlling a DockerEvalSession.

    Defaults match the harness defaults that minimise image churn across a
    batch run (rm_image=False, cache_level="env"). Bump timeout for
    sympy/django which run >20min of tests on cold pytest startup.

    `namespace="swebench"` makes make_test_spec emit registry-prefixed tags
    (swebench/sweb.eval.x86_64.<iid>:latest) so run_instance pulls the
    pre-built instance image from Docker Hub instead of trying to build it
    from a local `sweb.env.*` layer — which is what fails with
    "Environment image ... not found" when the env layer was never built
    locally. Set to None only if you're building images yourself.
    """

    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    model_name: str = "pare_v6"
    run_id: str = "pare-tier2"
    timeout: int = 1800
    cache_level: str = "env"
    rm_image: bool = False
    force_rebuild: bool = False
    namespace: str | None = "swebench"
    instance_image_tag: str = "latest"
    # Where run_instance writes report.json. Harness default is
    # logs/run_evaluation/<run_id>/<model>/<instance>/ under CWD.
    logs_root: Path = Path("logs/run_evaluation")


class DockerEvalSession:
    """Batch-scoped wrapper around swebench.harness.run_evaluation.run_instance.

    A single session caches:
    - the Docker client (one handshake per batch, not per instance),
    - the HuggingFace dataset row map (loaded once, ~5s first time),
    - the TestSpec per instance (via lru_cache).

    Intended usage:

        session = build_session(DockerEvalConfig(...))
        try:
            for task in tasks:
                result = session.verify_diff(task.instance_id, diff_str)
        finally:
            session.close()
    """

    def __init__(self, config: DockerEvalConfig) -> None:
        self.config = config
        self._client: Any | None = None
        self._rows: dict[str, dict[str, Any]] | None = None
        self._initialised = False

    # -- lazy initialisation -------------------------------------------------

    def _require_extra(self) -> tuple[Any, Any, Any, Any]:
        """Import swebench + docker + datasets, raise a clean error if missing."""
        try:
            from swebench.harness.run_evaluation import run_instance  # type: ignore
            from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
            from datasets import load_dataset  # type: ignore
            import docker  # type: ignore
        except ImportError as e:
            raise DockerEvalUnavailable(f"{_DOCKER_EXTRA_HINT} (missing: {e.name})") from e
        return run_instance, make_test_spec, load_dataset, docker

    def _ensure_ready(self) -> None:
        if self._initialised:
            return
        run_instance, make_test_spec, load_dataset, docker = self._require_extra()
        # Cache module refs on the instance so verify_diff doesn't re-import.
        self._run_instance = run_instance
        self._make_test_spec = make_test_spec

        try:
            self._client = docker.from_env()
            # Verify daemon reachability upfront — otherwise the first
            # run_instance call would hang on socket timeout.
            self._client.ping()
        except Exception as e:
            raise DockerEvalUnavailable(
                f"Docker daemon unreachable: {e}. Start Docker Desktop / dockerd."
            ) from e

        logger.info(
            "loading %s split=%s (first call ~5s, cached after)",
            self.config.dataset_name, self.config.split,
        )
        ds = load_dataset(self.config.dataset_name, split=self.config.split)
        self._rows = {r["instance_id"]: dict(r) for r in ds}
        logger.info("dataset rows loaded: %d", len(self._rows))
        self._initialised = True

    # -- TestSpec cache ------------------------------------------------------

    @lru_cache(maxsize=256)
    def _spec(self, instance_id: str) -> Any:
        assert self._rows is not None
        row = self._rows.get(instance_id)
        if row is None:
            raise KeyError(
                f"instance_id {instance_id!r} not in dataset "
                f"{self.config.dataset_name}:{self.config.split}"
            )
        # namespace + instance_image_tag are required by swebench>=3.0 so
        # run_instance pulls the pre-built image from Docker Hub instead
        # of building locally from a missing env layer.
        return self._make_test_spec(
            row,
            namespace=self.config.namespace,
            instance_image_tag=self.config.instance_image_tag,
        )

    # -- main entry point ----------------------------------------------------

    def verify_diff(self, instance_id: str, final_diff: str) -> Tier2CheckResult:
        """Run SWE-bench eval on (instance_id, final_diff) and map to Tier2CheckResult.

        Short-circuits when final_diff is empty — the harness would fail the
        patch-apply step anyway, and spinning up a container for it wastes
        ~30s. Records `tier2_error="empty_diff_skipped_docker"` so Module B
        can tell infra-skip apart from a genuine docker failure.
        """
        if not final_diff or not final_diff.strip():
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error="empty_diff_skipped_docker",
            )

        harness_diff = _strip_pare_internal_paths(final_diff)
        if not harness_diff.strip():
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error="empty_diff_after_pare_strip",
            )

        try:
            self._ensure_ready()
        except DockerEvalUnavailable as e:
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error=str(e),
            )

        try:
            spec = self._spec(instance_id)
        except KeyError as e:
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error=str(e),
            )

        pred = {
            "instance_id": instance_id,
            "model_name_or_path": self.config.model_name,
            "model_patch": harness_diff,
        }

        try:
            self._run_instance(
                test_spec=spec,
                pred=pred,
                rm_image=self.config.rm_image,
                force_rebuild=self.config.force_rebuild,
                client=self._client,
                run_id=self.config.run_id,
                timeout=self.config.timeout,
            )
        except Exception as e:
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error=f"run_instance_failed: {type(e).__name__}: {e}",
            )

        return self._read_report(instance_id)

    # -- report parsing ------------------------------------------------------

    def _report_path(self, instance_id: str) -> Path:
        return (
            self.config.logs_root
            / self.config.run_id
            / self.config.model_name
            / instance_id
            / "report.json"
        )

    def _read_report(self, instance_id: str) -> Tier2CheckResult:
        report_path = self._report_path(instance_id)
        if not report_path.exists():
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error=f"report_missing: {report_path}",
            )

        try:
            raw = report_path.read_text(encoding="utf-8")
            report = json.loads(raw)
        except Exception as e:
            return Tier2CheckResult(
                enabled=True,
                command=f"swebench:{instance_id}",
                passed=False,
                return_code=None,
                output="",
                error=f"report_parse_failed: {e}",
            )

        # Harness wraps per-instance result under {instance_id: {...}}
        # on some versions; unwrap defensively.
        if instance_id in report and isinstance(report[instance_id], dict):
            report = report[instance_id]

        resolved = bool(report.get("resolved", False))
        # Truncate to keep trajectory JSONL lines under a few KB.
        output_blob = json.dumps(report, ensure_ascii=False)[:4000]

        return Tier2CheckResult(
            enabled=True,
            command=f"swebench:{instance_id}",
            passed=resolved,
            return_code=0 if resolved else 1,
            output=output_blob,
            error="",
        )

    # -- teardown ------------------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._initialised = False


def build_session(config: DockerEvalConfig | None = None) -> DockerEvalSession:
    """Construct a DockerEvalSession with default config if none given."""
    return DockerEvalSession(config or DockerEvalConfig())
