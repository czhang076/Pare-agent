"""BashTool — execute shell commands with timeout and output capture.

This is the most powerful (and dangerous) tool in the system. It runs
arbitrary shell commands in the sandbox working directory. Safety is
provided by:
- ALWAYS_CONFIRM permission level (user must approve every invocation)
- Configurable timeout (default 30s)
- Output truncation (default 200 lines)
- Headless mode auto-approves but still enforces timeout
"""

from __future__ import annotations

import asyncio
import logging
import os

from pare.tools.base import (
    MutationType,
    PermissionLevel,
    Tool,
    ToolContext,
    ToolResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_OUTPUT_LINES = 200


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command and return its output. "
        "Use this to run tests, install packages, check file contents, "
        "or perform any operation available from the command line. "
        "Commands run in the project working directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Maximum execution time in seconds. Default: 30. "
                    "Increase for long-running commands like test suites."
                ),
            },
        },
        "required": ["command"],
    }
    mutation_type = MutationType.EXECUTE
    permission_level = PermissionLevel.ALWAYS_CONFIRM

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        command = params.get("command", "")
        if not command.strip():
            return ToolResult(success=False, output="", error="Empty command")

        timeout = params.get("timeout", _DEFAULT_TIMEOUT)

        if context.exec_target == "container":
            return await self._execute_in_container(command, timeout, context)

        try:
            env = {**os.environ, **context.env} if context.env else None
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(context.cwd),
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command timed out after {timeout}s: {command}",
                    metadata={"return_code": -1, "timed_out": True},
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            return_code = process.returncode or 0

            # Build output: combine stdout and stderr
            output_parts = []
            if stdout.strip():
                output_parts.append(stdout.rstrip())
            if stderr.strip():
                output_parts.append(f"STDERR:\n{stderr.rstrip()}")

            output = "\n".join(output_parts)

            # Truncate if too long
            lines = output.splitlines()
            if len(lines) > _DEFAULT_MAX_OUTPUT_LINES:
                truncated = "\n".join(lines[:_DEFAULT_MAX_OUTPUT_LINES])
                remaining = len(lines) - _DEFAULT_MAX_OUTPUT_LINES
                output = f"{truncated}\n\n[truncated — {remaining} more lines]"

            result = ToolResult(
                success=return_code == 0,
                output=output,
                error=f"Exit code: {return_code}" if return_code != 0 else "",
                metadata={"return_code": return_code, "timed_out": False},
            )
            return result

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="bash not found. Ensure bash is installed and in PATH.",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute command: {e}",
            )

    async def _execute_in_container(
        self, command: str, timeout: float, context: ToolContext
    ) -> ToolResult:
        """Container-mode execution — routes through InstanceContainer.exec.

        Output formatting mirrors the host path: stdout merged with an
        optional ``STDERR:\n...`` footer, truncated to the same line cap.
        The ``metadata`` dict matches so downstream trajectory consumers
        see a uniform shape regardless of exec_target.
        """
        if context.container is None:
            return ToolResult(
                success=False,
                output="",
                error="bash container mode requires ToolContext.container",
            )
        try:
            r = await context.container.exec(
                command,
                timeout=timeout,
                env=context.env or None,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute command in container: {e}",
            )

        if r.timed_out:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s: {command}",
                metadata={"return_code": r.exit_code, "timed_out": True},
            )

        output_parts: list[str] = []
        if r.stdout.strip():
            output_parts.append(r.stdout.rstrip())
        if r.stderr.strip():
            output_parts.append(f"STDERR:\n{r.stderr.rstrip()}")
        output = "\n".join(output_parts)

        lines = output.splitlines()
        if len(lines) > _DEFAULT_MAX_OUTPUT_LINES:
            truncated = "\n".join(lines[:_DEFAULT_MAX_OUTPUT_LINES])
            remaining = len(lines) - _DEFAULT_MAX_OUTPUT_LINES
            output = f"{truncated}\n\n[truncated — {remaining} more lines]"

        return ToolResult(
            success=r.exit_code == 0,
            output=output,
            error=f"Exit code: {r.exit_code}" if r.exit_code != 0 else "",
            metadata={"return_code": r.exit_code, "timed_out": False},
        )
