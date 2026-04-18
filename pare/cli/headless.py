"""Headless batch mode — run a task with no interactive UI.

Usage:
    pare "fix the bug"
    pare "fix the bug" --output result.json

Headless mode:
- No Rich console output, no prompts, no streaming text
- Logs progress to stderr (structured lines)
- Writes structured JSON result to --output path (if given)
- Exit code: 0 = success, 1 = agent failure, 2 = setup error

This is the entry point for CI pipelines, SWE-bench harness,
and any non-interactive automation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from pare.agent.executor import ExecutionResult
from pare.agent.guardrails import GuardrailConfig
from pare.agent.orchestrator import Agent, AgentConfig
from pare.llm import create_llm
from pare.telemetry import EventLog
from pare.trajectory.schema import (
    SCHEMA_VERSION,
    StepAttempt,
    TokenUsageSummary,
    TrajectoryRecord,
    VerificationResult,
    append_trajectory_jsonl,
)

logger = logging.getLogger(__name__)


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


def _result_to_dict(result: ExecutionResult) -> dict:
    """Convert ExecutionResult to a JSON-serializable dict."""
    return {
        "success": result.success,
        "output": result.output,
        "stop_reason": result.stop_reason,
        "tool_call_count": result.tool_call_count,
        "verification": {
            "final_passed": result.success,
            "tier1_pass": result.tier1_pass,
            "tier2_pass": result.tier2_pass,
            "tier2_command": result.tier2_command,
            "tier2_return_code": result.tier2_return_code,
        },
        "usage": {
            "input_tokens": result.total_usage.input_tokens,
            "output_tokens": result.total_usage.output_tokens,
            "total_tokens": result.total_usage.total_tokens,
            "cache_read_tokens": result.total_usage.cache_read_tokens,
            "cache_create_tokens": result.total_usage.cache_create_tokens,
        },
    }


def _build_trajectory_record(
    *,
    task: str,
    instance_id: str,
    provider: str,
    model: str,
    seed: int,
    created_at: float,
    elapsed_seconds: float,
    result: ExecutionResult,
    final_diff: str = "",
) -> TrajectoryRecord:
    attempts: list[StepAttempt] = []
    for raw in result.attempts:
        if isinstance(raw, dict):
            attempts.append(StepAttempt.from_dict(raw))

    if not attempts:
        attempts = [
            StepAttempt(
                step_number=1,
                attempt_number=1,
                goal=task[:200],
                status="success" if result.success else "failed",
                target_files=[],
                tool_names=[],
                failure_reason="" if result.success else result.stop_reason,
            )
        ]

    verification = VerificationResult(
        final_passed=result.success,
        tier1_pass=result.tier1_pass,
        tier2_pass=result.tier2_pass,
        tier2_command=result.tier2_command,
    )

    metadata: dict[str, str] = {
        "provider": provider,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": str(round(elapsed_seconds, 3)),
    }
    if result.tier2_error:
        metadata["tier2_error"] = result.tier2_error
    if result.tier2_output:
        metadata["tier2_output"] = result.tier2_output
    if final_diff:
        metadata["final_diff"] = final_diff

    return TrajectoryRecord(
        schema_version=SCHEMA_VERSION,
        trajectory_id=f"traj-{int(created_at)}-{uuid4().hex[:8]}",
        instance_id=instance_id,
        task=task,
        model=model,
        seed=seed,
        created_at=created_at,
        llm_claimed_success=result.llm_claimed_success,
        verification=verification,
        attempts=attempts,
        tool_call_events=list(result.tool_call_events),
        token_usage=TokenUsageSummary(
            input_tokens=result.total_usage.input_tokens,
            output_tokens=result.total_usage.output_tokens,
            cache_read_tokens=result.total_usage.cache_read_tokens,
            cache_create_tokens=result.total_usage.cache_create_tokens,
        ),
        metadata=metadata,
    )


async def run_headless(
    task: str,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    cwd: Path | None = None,
    output_path: Path | None = None,
    trajectory_path: Path | None = None,
    instance_id: str = "local-run",
    seed: int = 0,
    test_command: str | None = None,
    test_timeout: int = 300,
    use_planning: bool = False,
    max_tool_calls: int = 100,
    max_tool_calls_per_step: int = 15,
    verbose: bool = False,
) -> int:
    """Run a task in headless mode. Returns exit code (0=success).

    Args:
        task: Natural language task description.
        provider: LLM provider name.
        model: Model name (or provider default).
        api_key: API key (or read from env).
        base_url: Custom API base URL.
        cwd: Working directory.
        output_path: Path to write JSON result (optional).
        trajectory_path: Path to append trajectory JSONL (optional).
        instance_id: Dataset/sample instance identifier.
        seed: Run seed for trajectory metadata.
        test_command: Optional Tier-2 verification command.
        test_timeout: Timeout for Tier-2 verification command.
        use_planning: Whether to use Orient -> Plan -> Execute mode.
        max_tool_calls: Total tool-call budget for one task.
        max_tool_calls_per_step: Tool-call budget per step.
        verbose: Enable debug logging.

    Returns:
        0 on success, 1 on agent failure, 2 on setup error.
    """
    cwd = (cwd or Path(".")).resolve()

    # Resolve API key
    resolved_key = _resolve_api_key(provider, api_key)
    if not resolved_key:
        env_var = f"{provider.upper()}_API_KEY"
        print(f"Error: No API key. Set {env_var}.", file=sys.stderr)
        return 2

    # Create LLM
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

    # Telemetry
    # Keep telemetry outside the task repo to avoid interfering with git checkpoints.
    session_dir = (cwd.parent / ".pare_sessions").resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    event_log = EventLog(session_dir / f"{cwd.name}_session_{int(time.time())}.jsonl")

    # Agent — headless=True disables interactive confirmations
    guardrail_config = GuardrailConfig(
        max_tool_calls=max_tool_calls,
        max_tool_calls_per_step=max_tool_calls_per_step,
    )
    agent_config = AgentConfig(
        guardrail_config=guardrail_config,
        use_planning=use_planning,
        tier2_test_command=test_command,
        tier2_timeout_seconds=test_timeout,
    )

    agent = Agent(
        llm=llm,
        cwd=cwd,
        config=agent_config,
        event_log=event_log,
        headless=True,
    )

    # Progress logging to stderr
    def on_tool_call(event):
        if event.blocked_reason:
            print(f"[blocked] {event.tool_name}: {event.blocked_reason}", file=sys.stderr)
        elif event.result:
            status = "ok" if event.result.success else "err"
            print(
                f"[{status}] {event.tool_name} ({event.duration:.1f}s)",
                file=sys.stderr,
            )

    # Run
    print(f"[start] task={task[:100]}", file=sys.stderr)
    print(
        f"[config] provider={provider} model={llm.model} cwd={cwd} "
        f"planning={'on' if use_planning else 'off'} "
        f"budget={max_tool_calls}/{max_tool_calls_per_step} "
        f"tier2={'on' if test_command else 'off'} "
        f"trajectory={'on' if trajectory_path else 'off'}",
        file=sys.stderr,
    )

    start = time.time()
    result = await agent.run(task, on_tool_call=on_tool_call)
    elapsed = time.time() - start
    created_at = time.time()

    # Capture the full diff against the checkpoint's original branch BEFORE
    # finalize squash-merges it away. The diff is the ground truth for
    # trajectory-level "what did the agent actually change," used by the
    # B1.1 (incomplete fix) classifier. Synthesizing from file_edit tool
    # result_content is unreliable — those results are confirmation
    # strings, not unified diffs.
    #
    # Critical: first commit any uncommitted working-tree changes. The
    # orchestrator only runs `checkpoint()` after a step's `result.success`
    # is True — so when every step fails (as in all 20 sympy20 pilot runs),
    # the agent's file_edit changes sit uncommitted and `git diff BRANCH HEAD`
    # returns nothing (or only Pare's own .pare/MEMORY.md noise committed at
    # "before task execution"). Doing one last checkpoint here is a no-op
    # when the tree is clean (see GitCheckpoint.checkpoint) and captures
    # the agent's real edits when it isn't.
    final_diff = ""
    cp = agent.checkpoint
    print(
        f"[diag] checkpoint_present={cp is not None} "
        f"is_active={cp.is_active if cp is not None else 'n/a'} "
        f"original_branch={cp.original_branch if cp is not None else 'n/a'} "
        f"working_branch={cp.working_branch if cp is not None else 'n/a'}",
        file=sys.stderr,
    )
    if cp is not None and cp.is_active:
        try:
            sha = await cp.checkpoint("finalize: capture agent state")
            print(f"[diag] finalize checkpoint sha={sha}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] could not checkpoint before diff: {e}", file=sys.stderr)
        try:
            final_diff = await cp.get_full_diff()
            print(f"[diag] get_full_diff returned {len(final_diff)} bytes", file=sys.stderr)
        except Exception as e:
            print(f"[warn] could not capture final_diff: {e}", file=sys.stderr)

    trajectory_record: TrajectoryRecord | None = None
    if trajectory_path:
        try:
            trajectory_record = _build_trajectory_record(
                task=task,
                instance_id=instance_id,
                provider=provider,
                model=llm.model,
                seed=seed,
                created_at=created_at,
                elapsed_seconds=elapsed,
                result=result,
                final_diff=final_diff,
            )
            append_trajectory_jsonl(trajectory_path, trajectory_record)
            print(f"[trajectory] {trajectory_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing trajectory JSONL: {e}", file=sys.stderr)
            return 2

    # Finalize checkpoint (squash-merge back)
    await agent.finalize_checkpoint()
    event_log.close()

    # Summary to stderr
    print(
        f"[done] success={result.success} tools={result.tool_call_count} "
        f"tokens={result.total_usage.total_tokens} time={elapsed:.1f}s",
        file=sys.stderr,
    )

    # Write JSON result
    if output_path:
        result_dict = _result_to_dict(result)
        result_dict["elapsed_seconds"] = round(elapsed, 2)
        if trajectory_record is not None:
            result_dict["trajectory_id"] = trajectory_record.trajectory_id
            result_dict["trajectory_path"] = str(trajectory_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False))
        print(f"[output] {output_path}", file=sys.stderr)

    # Final text to stdout (for piping). Encode via stdout's encoding with
    # errors='replace' so non-ASCII output (e.g. ✓) doesn't crash on
    # legacy Windows consoles (cp936 / GBK).
    if result.output:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = result.output.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe)

    return 0 if result.success else 1
