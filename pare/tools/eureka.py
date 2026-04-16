"""Async Blackboard / Eureka Broadcasting.
Allows universes to share high-confidence discoveries (subspace intel)
without sharing direct control or breaking persona isolation.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

@dataclass
class EurekaMessage:
    universe_id: str
    message: str


class AsynchronousBlackboard:
    """Central Queue for cross-universe broadcasts."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[EurekaMessage]] = {}

    def register_universe(self, universe_id: str) -> None:
        """Create a dedicated receiving queue for a new universe."""
        if universe_id not in self._queues:
            self._queues[universe_id] = asyncio.Queue()

    def broadcast(self, sender_id: str, message: str) -> None:
        """Broadcast intel from sender to all other universes."""
        intel = EurekaMessage(universe_id=sender_id, message=message)
        logger.info("Eureka Broadcast from Universe %s: %s", sender_id, message)
        
        for uid, queue in self._queues.items():
            if uid != sender_id:
                queue.put_nowait(intel)

    def consume_intel(self, universe_id: str) -> List[str]:
        """Consume all pending intel for this universe."""
        if universe_id not in self._queues:
            return []
            
        q = self._queues[universe_id]
        messages = []
        while not q.empty():
            msg = q.get_nowait()
            # Formatted as a user observation
            alert = (
                f"[System Alert: Universe {msg.universe_id} has shared an external observation. "
                f"You may use this to inform your current strategy, or ignore it if irrelevant.]\n\n"
                f"Payload: {msg.message}"
            )
            messages.append(alert)
        return messages

# Global singleton or context-injected
_global_blackboard = AsynchronousBlackboard()

def get_blackboard() -> AsynchronousBlackboard:
    return _global_blackboard
