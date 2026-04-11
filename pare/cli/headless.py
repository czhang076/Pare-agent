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

from pare.agent.executor import ExecutionResult
from pare.agent.orchestrator import Agent, AgentConfig
from pare.llm import create_llm
from pare.telemetry import EventLog

logger = logging.getLogger(__name__)


def _resolve_api_key(provider: str, api_key: str | None) -> str | None:
    """Resolve API key from argument or environment."""
    if api_key:
        return api_key
    env_map = {
        "openai": "OPENAI_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
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
        "usage": {
            "input_tokens": result.total_usage.input_tokens,
            "output_tokens": result.total_usage.output_tokens,
            "total_tokens": result.total_usage.total_tokens,
            "cache_read_tokens": result.total_usage.cache_read_tokens,
            "cache_create_tokens": result.total_usage.cache_create_tokens,
        },
    }


async def run_headless(
    task: str,
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    cwd: Path | None = None,
    output_path: Path | None = None,
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
    session_dir = cwd / ".pare"
    session_dir.mkdir(exist_ok=True)
    event_log = EventLog(session_dir / f"session_{int(time.time())}.jsonl")

    # Agent — headless=True disables interactive confirmations
    agent = Agent(
        llm=llm,
        cwd=cwd,
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
    print(f"[config] provider={provider} model={llm.model} cwd={cwd}", file=sys.stderr)

    start = time.time()
    result = await agent.run(task, on_tool_call=on_tool_call)
    elapsed = time.time() - start

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False))
        print(f"[output] {output_path}", file=sys.stderr)

    # Final text to stdout (for piping)
    if result.output:
        print(result.output)

    return 0 if result.success else 1
