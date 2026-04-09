"""Context manager — message assembly, compaction trigger, memory coordination.

This is the central coordinator for the 3-layer memory architecture.  It:
1. Assembles the message list for each LLM call (system + memory + history)
2. Checks if compaction is needed after each turn
3. Runs the 5-stage compression pipeline when triggered
4. Keeps the memory index and session history in sync

The orchestrator/executor interacts only with ContextManager, never with
the lower-level memory/history/compactor modules directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pare.context.compactor import CompactionConfig, CompactionPipeline, CompactionResult
from pare.context.history import SessionHistory
from pare.context.memory import MemoryIndex, TopicStore
from pare.llm.base import LLMAdapter, Message
from pare.llm.token_counter import estimate_tokens_heuristic

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages what the LLM sees at each turn.

    Usage:
        ctx = ContextManager(cwd=Path("."), llm=llm_adapter)
        ctx.set_system_prompt("You are Pare...")
        ctx.set_task("Fix the login bug")

        # Each turn:
        messages = ctx.get_messages()
        response = await llm.chat(messages)
        ctx.add_assistant_message(response_msg)
        ctx.add_tool_results(result_msg)

        # Auto-compaction:
        if ctx.needs_compaction():
            result = await ctx.compact()
    """

    def __init__(
        self,
        cwd: Path,
        llm: LLMAdapter | None = None,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self.cwd = cwd
        self.memory = MemoryIndex(cwd)
        self.topics = TopicStore(cwd)
        self.history = SessionHistory(cwd / ".pare" / "history.jsonl")
        self.compactor = CompactionPipeline(compaction_config, llm=llm)

        # Message state
        self._system_prompt: str = ""
        self._messages: list[Message] = []
        self._topic_context: list[str] = []  # Currently loaded topic names

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt. Called once at session start."""
        self._system_prompt = prompt

    def set_task(self, task: str) -> None:
        """Record the current task in the memory index."""
        self.memory.update_section("Current Task", task)
        self.memory.save()

    def load_topics(self, topic_names: list[str]) -> None:
        """Load specific topic files into the active context."""
        self._topic_context = [
            name for name in topic_names if self.topics.exists(name)
        ]

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def get_messages(self) -> list[Message]:
        """Assemble the full message list for the next LLM call.

        Order (optimized for prompt cache hits):
        1. System prompt (stable — cached)
        2. Memory index (mostly stable — cached)
        3. Topic files (stable within a step — cached)
        4. Conversation history (grows)
        """
        assembled: list[Message] = []

        # 1. System prompt + memory index + topics as system message
        system_parts = [self._system_prompt]

        memory_content = self.memory.get_content()
        if memory_content.strip():
            system_parts.append(f"\n## Memory Index\n{memory_content}")

        for topic_name in self._topic_context:
            topic_content = self.topics.read(topic_name)
            if topic_content:
                system_parts.append(f"\n## Topic: {topic_name}\n{topic_content}")

        assembled.append(Message(role="system", content="\n".join(system_parts)))

        # 2. Conversation messages
        assembled.extend(self._messages)

        return assembled

    def add_user_message(self, text: str) -> None:
        """Add a user message to the conversation."""
        msg = Message(role="user", content=text)
        self._messages.append(msg)
        self.history.append(msg, turn=self.history.next_turn())

    def add_assistant_message(self, message: Message) -> None:
        """Add an assistant message (from LLM response)."""
        self._messages.append(message)
        self.history.append(message)

    def add_tool_results(self, message: Message) -> None:
        """Add a tool_result message."""
        self._messages.append(message)
        self.history.append(message)

    def add_message(self, message: Message) -> None:
        """Add any message to the conversation."""
        self._messages.append(message)
        self.history.append(message)

    @property
    def messages(self) -> list[Message]:
        """Raw conversation messages (without system prompt assembly)."""
        return self._messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._messages = value

    # ------------------------------------------------------------------
    # Token counting & compaction
    # ------------------------------------------------------------------

    def token_count(self) -> int:
        """Estimate token count of the full assembled message list."""
        return estimate_tokens_heuristic(self.get_messages())

    def needs_compaction(self) -> bool:
        """Check if compaction should be triggered."""
        return self.compactor.needs_compaction(self.get_messages())

    async def compact(self) -> CompactionResult:
        """Run the 5-stage compaction pipeline on the message list.

        The system prompt is rebuilt from get_messages(), so we compact
        only the conversation portion (_messages) and then rebuild.
        """
        # Build full list for compaction
        full = self.get_messages()

        result = await self.compactor.compact(full)

        if result.stage_reached > 0:
            # Extract the compacted conversation (everything after system)
            self._messages = full[1:] if len(full) > 1 else []

            logger.info(
                "Compaction complete: stage %d, %d → %d tokens, %d messages dropped",
                result.stage_reached,
                result.tokens_before,
                result.tokens_after,
                result.messages_dropped,
            )

        return result

    # ------------------------------------------------------------------
    # Memory index updates
    # ------------------------------------------------------------------

    def update_memory(self, section: str, content: str) -> None:
        """Update a section of the memory index and persist to disk."""
        self.memory.update_section(section, content)
        self.memory.save()

    # ------------------------------------------------------------------
    # History search
    # ------------------------------------------------------------------

    def search_history(self, query: str, max_results: int = 10) -> list[str]:
        """Search session history. Returns matching text snippets."""
        entries = self.history.search(query, max_results=max_results)
        return [f"[turn {e.turn}, {e.role}] {e.text[:200]}" for e in entries]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        self.history.close()
