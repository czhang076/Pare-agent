"""Tool system core — base class, registry, execution, and result types.

Every tool in pare implements the Tool ABC. The ToolRegistry holds all
registered tools and provides schema export for the LLM. The executor
handles concurrent reads vs serial writes.

Design decisions:
- MutationType controls concurrency: READ tools run in parallel via
  asyncio.gather, WRITE/EXECUTE tools run serially. This is correct
  by construction — no race conditions possible.
- PermissionLevel controls user confirmation: AUTO (reads), CONFIRM_ONCE
  (writes — ask once then auto), ALWAYS_CONFIRM (bash — always ask).
- ToolContext carries per-execution state: working directory, environment,
  permission cache, and the event log.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pare.llm.base import ToolSchema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MutationType(Enum):
    """Classifies a tool's side effects for concurrency control."""

    READ = "read"        # File reads, searches — safe to run concurrently
    WRITE = "write"      # File edits, git operations — must run serially
    EXECUTE = "execute"  # Bash commands — serial, may need confirmation


class PermissionLevel(Enum):
    """Controls when user confirmation is required."""

    AUTO = "auto"                  # No confirmation needed (reads)
    CONFIRM_ONCE = "confirm_once"  # Confirm first time, then auto
    ALWAYS_CONFIRM = "always"      # Always ask (bash, destructive ops)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolResult:
    """Result of a single tool execution.

    Attributes:
        success: Whether the tool completed without error.
        output: The tool's output text (shown to LLM as tool_result).
        error: Error message if success is False.
        metadata: Extra data not shown to LLM (e.g. return code, timing).
    """

    success: bool
    output: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def truncate(self, max_lines: int = 200) -> ToolResult:
        """Return a copy with output truncated to max_lines."""
        lines = self.output.splitlines()
        if len(lines) <= max_lines:
            return self
        truncated = "\n".join(lines[:max_lines])
        remaining = len(lines) - max_lines
        return ToolResult(
            success=self.success,
            output=f"{truncated}\n\n[truncated — {remaining} more lines]",
            error=self.error,
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ToolContext:
    """Per-execution context passed to every tool.

    Attributes:
        cwd: Working directory for file operations and commands.
        env: Extra environment variables for subprocess execution.
        confirmed_tools: Set of tool names the user has already confirmed
            (for CONFIRM_ONCE permission level).
        headless: If True, skip all confirmation prompts (SWE-bench mode).
    """

    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    confirmed_tools: set[str] = field(default_factory=set)
    headless: bool = False


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses must implement execute() and set the class-level attributes.
    The to_schema() method converts the tool definition to the LLM-facing
    format (ToolSchema) used by LLMAdapter.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema for input validation
    mutation_type: MutationType
    permission_level: PermissionLevel

    @abstractmethod
    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """Execute the tool with the given parameters.

        Implementations should:
        - Validate params (raise ValueError for invalid input)
        - Perform the operation
        - Return ToolResult with success/failure and output
        - Never raise unhandled exceptions — catch and return ToolResult
        """

    def to_schema(self) -> ToolSchema:
        """Convert to LLM-facing schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def needs_confirmation(self, context: ToolContext) -> bool:
        """Check if this tool call needs user confirmation."""
        if context.headless:
            return False

        match self.permission_level:
            case PermissionLevel.AUTO:
                return False
            case PermissionLevel.CONFIRM_ONCE:
                return self.name not in context.confirmed_tools
            case PermissionLevel.ALWAYS_CONFIRM:
                return True

    def mark_confirmed(self, context: ToolContext) -> None:
        """Record that the user confirmed this tool."""
        context.confirmed_tools.add(self.name)


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Central registry for all available tools.

    Tools are registered at startup. The registry provides:
    - Schema export for LLM (all tools or filtered by mutation type)
    - Tool lookup by name
    - Execution with read/write concurrency separation
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ValueError if name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (%s)", tool.name, tool.mutation_type.value)

    def get(self, name: str) -> Tool:
        """Look up a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def get_all_schemas(self) -> list[ToolSchema]:
        """Get schemas for all registered tools (sent to LLM)."""
        return [tool.to_schema() for tool in self._tools.values()]

    def get_schemas_by_mutation(self, *types: MutationType) -> list[ToolSchema]:
        """Get schemas filtered by mutation type."""
        return [
            tool.to_schema()
            for tool in self._tools.values()
            if tool.mutation_type in types
        ]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def execute(
        self,
        calls: list[dict],
        context: ToolContext,
    ) -> list[ToolResult]:
        """Execute a batch of tool calls with read/write separation.

        READ tools are dispatched concurrently via asyncio.gather.
        WRITE and EXECUTE tools run sequentially.
        Results are returned in the same order as the input calls.

        Each call dict must have: {"name": str, "arguments": dict}
        """
        results: list[ToolResult | None] = [None] * len(calls)

        # Separate reads from writes
        read_indices: list[int] = []
        serial_indices: list[int] = []

        for i, call in enumerate(calls):
            tool = self.get(call["name"])
            if tool.mutation_type == MutationType.READ:
                read_indices.append(i)
            else:
                serial_indices.append(i)

        # Execute reads concurrently
        if read_indices:
            read_tasks = []
            for i in read_indices:
                call = calls[i]
                tool = self.get(call["name"])
                read_tasks.append(self._execute_one(tool, call["arguments"], context))

            read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
            for idx, result in zip(read_indices, read_results):
                if isinstance(result, Exception):
                    results[idx] = ToolResult(
                        success=False,
                        output="",
                        error=f"Tool execution error: {result}",
                    )
                else:
                    results[idx] = result

        # Execute writes/executes serially
        for i in serial_indices:
            call = calls[i]
            tool = self.get(call["name"])
            try:
                results[i] = await self._execute_one(tool, call["arguments"], context)
            except Exception as e:
                results[i] = ToolResult(
                    success=False,
                    output="",
                    error=f"Tool execution error: {e}",
                )

        return [r for r in results if r is not None]

    async def _execute_one(
        self, tool: Tool, arguments: dict, context: ToolContext
    ) -> ToolResult:
        """Execute a single tool call with error handling."""
        try:
            return await tool.execute(arguments, context)
        except Exception as e:
            logger.exception("Tool '%s' raised an exception", tool.name)
            return ToolResult(
                success=False,
                output="",
                error=f"{type(e).__name__}: {e}",
            )


def create_default_registry() -> ToolRegistry:
    """Create a registry with all P0 (MVP) tools registered.

    Imports are deferred to avoid circular dependencies and to keep
    startup fast when not all tools are needed.
    """
    from pare.tools.bash import BashTool
    from pare.tools.file_edit import FileCreateTool, FileEditTool
    from pare.tools.file_read import FileReadTool
    from pare.tools.search import SearchTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileEditTool())
    registry.register(FileCreateTool())
    registry.register(SearchTool())
    return registry
