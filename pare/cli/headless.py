"""Headless batch mode — run a task with no interactive UI.

Usage:
    pare "fix the bug"
    pare "fix the bug" --output result.json

Headless mode:
- No Rich console output, no prompts, no streaming text
- Logs progress to stderr (structured lines)
- Writes structured JSON result to --output path (if given)
- Exit code: 0 = success, 1 = agent failure, 2 = setup error

R5 state: the legacy ``run_headless`` entry (3-layer orchestrator /
executor / git-checkpoint) has been deleted along with the rest of the
old agent layer. Every invocation goes through the flat ReAct loop
inside a per-instance Docker container. The ``_flat_react_requested``
helper is kept (always returns True) so CLI-level opt-outs fail loudly
rather than silently re-routing to a ghost legacy path.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from pare.llm import create_llm
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    append_trajectory_jsonl,
)

logger = logging.getLogger(__name__)


def _flat_react_requested(cli_value: str | None) -> bool:
    """Resolve loop-mode precedence. Post-R5 the only path is flat ReAct.

    Any explicit ``--loop legacy`` / ``PARE_USE_LEGACY_LOOP=1`` triggers a
    loud setup error in the callers rather than a silent fallback — the
    3-layer loop no longer exists.
    """
    if cli_value in ("legacy", "old"):
        return False
    legacy_env = os.environ.get("PARE_USE_LEGACY_LOOP", "").strip().lower()
    if legacy_env in ("1", "true", "yes", "on"):
        return False
    return True


def _resolve_api_key(provider: str, api_key: str | None) -> str | None:
    """Resolve API key from argument or environment."""
    if api_key:
        return api_key
    env_map = {
        "openai": "OPENAI_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "glm": "GLM_API_KEY",
    }
    env_var = env_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.environ.get(env_var)


def _loop_result_to_record(
    *,
    task: str,
    instance_id: str,
    provider: str,
    model: str,
    seed: int,
    created_at: float,
    elapsed_seconds: float,
    loop_result,  # LoopResult — annotated loosely to avoid import at module load
    system_prompt: str,
) -> TrajectoryRecord:
    """Shape a :class:`LoopResult` into a :class:`TrajectoryRecord` JSONL row.

    Notes on the mapping:

    - ``verification.final_passed`` mirrors ``LoopResult.success`` (already
      computed by ``_finalize`` as ``declared_status=="fixed" AND tier2``).
    - Legacy ``StepAttempt`` records are not emitted — the flat ReAct loop
      is turn-based, not step-based. We synthesise a single placeholder so
      the v1 schema validator accepts the record; downstream consumers
      that care about per-call detail read ``tool_call_events`` instead.
    - ``metadata["declared_status"]`` is always set (empty string on silent
      exits) so Module B can distinguish "agent gave up silently" from
      "agent declared cannot_fix".
    """
    status = "success" if loop_result.success else "failed"
    failure_reason = "" if loop_result.success else (loop_result.stop_reason or "")
    attempt = StepAttempt(
        step_number=1,
        attempt_number=1,
        goal=task[:200],
        status=status,
        target_files=[],
        tool_names=[e.tool_name for e in loop_result.tool_call_events],
        failure_reason=failure_reason,
    )

    verification = VerificationResult(
        final_passed=loop_result.success,
        tier1_pass=loop_result.tier1_pass,
        tier2_pass=loop_result.tier2_pass,
        tier2_command=f"swebench:{instance_id}" if loop_result.tier2_enabled else "",
    )

    metadata: dict[str, str] = {
        "provider": provider,
        "stop_reason": loop_result.stop_reason,
        "elapsed_seconds": str(round(elapsed_seconds, 3)),
        "loop": "flat_react_v1",
        "declared_status": loop_result.declared_status,
    }
    if loop_result.declared_summary:
        metadata["declared_summary"] = loop_result.declared_summary
    if loop_result.tier2_output:
        metadata["tier2_output"] = loop_result.tier2_output
    if loop_result.error:
        metadata["error"] = loop_result.error
    if loop_result.final_diff:
        metadata["final_diff"] = loop_result.final_diff

    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=f"traj-{int(created_at)}-{uuid4().hex[:8]}",
        instance_id=instance_id,
        task=task,
        model=model,
        seed=seed,
        created_at=created_at,
        llm_claimed_success=(loop_result.declared_status == "fixed"),
        verification=verification,
        attempts=[attempt],
        tool_call_events=list(loop_result.tool_call_events),
        token_usage=TokenUsageSummary(
            input_tokens=loop_result.total_usage.input_tokens,
            output_tokens=loop_result.total_usage.output_tokens,
            cache_read_tokens=loop_result.total_usage.cache_read_tokens,
            cache_create_tokens=loop_result.total_usage.cache_create_tokens,
        ),
        metadata=metadata,
    )


async def run_headless_flat_react(
    task: str,
    *,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    output_path: Path | None = None,
    trajectory_path: Path | None = None,
    instance_id: str = "local-run",
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    seed: int = 0,
    max_steps: int = 50,
    system_prompt: str = "",
    verify: bool = False,
    use_orient: bool = False,
    use_planner: bool = False,
    verbose: bool = False,
) -> int:
    """Flat ReAct headless runner — the sole headless entry point post-R5.

    Boots an :class:`InstanceContainer` around a SWE-bench instance, runs
    :func:`pare.agent.loop.run_agent` inside it, then serialises the result
    as a v1/v2 :class:`TrajectoryRecord`. Exit code: ``0`` on
    ``LoopResult.success``, ``1`` otherwise, ``2`` on setup errors.
    """
    # Lazy imports: keep the docker-eval extra optional at module load time.
    from pare.agent.loop import LoopConfig, run_agent
    from pare.sandbox.instance_container import InstanceContainer
    from pare.tools.base import create_default_registry

    resolved_key = _resolve_api_key(provider, api_key)
    if not resolved_key:
        env_var = f"{provider.upper()}_API_KEY"
        print(f"Error: No API key. Set {env_var}.", file=sys.stderr)
        return 2

    try:
        llm = create_llm(
            provider=provider,
            model=model,
            api_key=resolved_key,
            base_url=base_url,
        )
    except Exception as e:
        print(f"Error creating LLM: {e}", file=sys.stderr)
        return 2

    try:
        container_cm = await InstanceContainer.build(
            instance_id,
            dataset_name=dataset_name,
            split=split,
        )
    except Exception as e:
        print(f"Error building container for {instance_id}: {e}", file=sys.stderr)
        return 2

    start = time.time()
    registry = create_default_registry()
    config = LoopConfig(
        system_prompt=system_prompt,
        max_steps=max_steps,
        verify_instance_id=instance_id if verify else None,
        use_orient=use_orient,
        use_planner=use_planner,
    )

    print(
        f"[start] flat-react task={task[:100]} instance={instance_id} "
        f"model={llm.model} steps={max_steps} verify={verify}",
        file=sys.stderr,
    )

    async with container_cm as container:
        try:
            loop_result = await run_agent(
                llm=llm,
                task=task,
                container=container,
                registry=registry,
                config=config,
            )
        except Exception as e:
            print(f"[error] run_agent crashed: {e}", file=sys.stderr)
            return 2

    elapsed = time.time() - start
    created_at = time.time()

    print(
        f"[done] stop_reason={loop_result.stop_reason} "
        f"declared_status={loop_result.declared_status or '<none>'} "
        f"tools={loop_result.tool_call_count} "
        f"tier2_pass={loop_result.tier2_pass if loop_result.tier2_enabled else 'n/a'} "
        f"time={elapsed:.1f}s",
        file=sys.stderr,
    )

    trajectory_record: TrajectoryRecord | None = None
    if trajectory_path:
        try:
            trajectory_record = _loop_result_to_record(
                task=task,
                instance_id=instance_id,
                provider=provider,
                model=llm.model,
                seed=seed,
                created_at=created_at,
                elapsed_seconds=elapsed,
                loop_result=loop_result,
                system_prompt=system_prompt,
            )
            append_trajectory_jsonl(trajectory_path, trajectory_record)
            print(f"[trajectory] {trajectory_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing trajectory JSONL: {e}", file=sys.stderr)
            return 2

    if output_path:
        result_dict = {
            "success": loop_result.success,
            "stop_reason": loop_result.stop_reason,
            "declared_status": loop_result.declared_status,
            "declared_summary": loop_result.declared_summary,
            "tool_call_count": loop_result.tool_call_count,
            "tier1_pass": loop_result.tier1_pass,
            "tier2_enabled": loop_result.tier2_enabled,
            "tier2_pass": loop_result.tier2_pass,
            "tier2_output": loop_result.tier2_output,
            "final_diff": loop_result.final_diff,
            "elapsed_seconds": round(elapsed, 2),
            "usage": {
                "input_tokens": loop_result.total_usage.input_tokens,
                "output_tokens": loop_result.total_usage.output_tokens,
                "total_tokens": loop_result.total_usage.total_tokens,
                "cache_read_tokens": loop_result.total_usage.cache_read_tokens,
                "cache_create_tokens": loop_result.total_usage.cache_create_tokens,
            },
            "error": loop_result.error,
        }
        if trajectory_record is not None:
            result_dict["trajectory_id"] = trajectory_record.trajectory_id
            result_dict["trajectory_path"] = str(trajectory_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False)
        )
        print(f"[output] {output_path}", file=sys.stderr)

    return 0 if loop_result.success else 1
