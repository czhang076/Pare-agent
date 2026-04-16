"""The Broadcast Eureka Tool for Cross-Universe Communication."""

from __future__ import annotations
from typing import Any

from pare.tools.base import Tool, ToolResult, MutationType, PermissionLevel
from pare.tools.eureka import get_blackboard


class BroadcastEurekaTool(Tool):
    """Shares high-priority context intel across parallel universes."""

    name = "broadcast_eureka"
    description = (
        "Broadcasts a high-confidence discovery to all other parallel universes. "
        "Use this only when you have discovered the absolute crash location or payload. "
        "DO NOT share your thoughts or assumptions. ONLY share stack traces or reproduction payloads."
    )
    mutation_type = MutationType.READ # Treat as harmless context sharing
    permission_level = PermissionLevel.AUTO
    
    parameters = {
        "type": "object",
        "properties": {
            "discovery": {
                "type": "string",
                "description": "The exact traceback or localized bug mechanism and crash line.",
            }
        },
        "required": ["discovery"],
    }

    def __init__(self, universe_id: str) -> None:
        self.universe_id = universe_id

    async def execute(self, params: dict[str, Any], context: dict[str, Any] | None = None) -> ToolResult:
        message = params.get("discovery", "")
        if not message:
            return ToolResult(success=False, output="Error: Empty broadcast payload.")
            
        board = get_blackboard()
        board.broadcast(self.universe_id, message)
        
        return ToolResult(
            success=True, 
            output="Intel successfully broadcasted to the Multiverse."
        )
