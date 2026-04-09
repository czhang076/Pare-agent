"""Five-stage gradient compression pipeline.

When the conversation token count exceeds a threshold, compaction runs
stages in order.  Each stage is progressively more aggressive.  Processing
stops as soon as the token count drops below the threshold.

Stage 1: Trim old tool results      — zero LLM calls, instant
Stage 2: Truncate verbose output    — zero LLM calls, instant
Stage 3: Extract session memory     — 1 LLM call (with failure fallback)
Stage 4: Full history summarization — 1 LLM call (with failure fallback)
Stage 5: Drop oldest messages       — zero LLM calls, lossy last resort

The pipeline never drops the system prompt or memory index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    LLMAdapter,
    Message,
)
from pare.llm.token_counter import estimate_tokens_heuristic

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompactionConfig:
    """Tunables for the compaction pipeline."""

    threshold_ratio: float = 0.70  # Trigger at 70% of max context
    max_context_tokens: int = 128_000  # Model's max context window
    keep_full_results: int = 5  # Stage 1: keep last N tool results in full
    max_result_lines: int = 50  # Stage 2: cap each tool result at N lines
    preserve_last_n_messages: int = 4  # Stage 5: always keep last N messages


@dataclass(slots=True)
class CompactionResult:
    """Outcome of a compaction run."""

    stage_reached: int  # Highest stage that ran (0 = no compaction needed)
    tokens_before: int
    tokens_after: int
    messages_dropped: int = 0
    summary_generated: bool = False


class CompactionPipeline:
    """Runs the 5-stage gradient compression on a message list.

    Usage:
        pipeline = CompactionPipeline(config, llm=llm_adapter)
        result = await pipeline.compact(messages)
        # messages is mutated in-place
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        llm: LLMAdapter | None = None,
    ) -> None:
        self.config = config or CompactionConfig()
        self.llm = llm  # Only needed for stages 3-4

    @property
    def threshold_tokens(self) -> int:
        return int(self.config.max_context_tokens * self.config.threshold_ratio)

    def needs_compaction(self, messages: list[Message]) -> bool:
        """Check if the message list exceeds the compaction threshold."""
        return estimate_tokens_heuristic(messages) > self.threshold_tokens

    async def compact(self, messages: list[Message]) -> CompactionResult:
        """Run the compaction pipeline. Mutates messages in-place.

        Returns CompactionResult describing what happened.
        """
        tokens_before = estimate_tokens_heuristic(messages)

        if tokens_before <= self.threshold_tokens:
            return CompactionResult(stage_reached=0, tokens_before=tokens_before, tokens_after=tokens_before)

        logger.info(
            "Compaction triggered: %d tokens (threshold: %d)",
            tokens_before, self.threshold_tokens,
        )

        result = CompactionResult(
            stage_reached=0,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
        )

        for stage_num, stage_fn in [
            (1, self._stage1_trim_old_results),
            (2, self._stage2_truncate_verbose),
            (3, self._stage3_extract_memory),
            (4, self._stage4_summarize_history),
            (5, self._stage5_drop_oldest),
        ]:
            result.stage_reached = stage_num
            await stage_fn(messages, result)

            current_tokens = estimate_tokens_heuristic(messages)
            result.tokens_after = current_tokens
            logger.info(
                "Stage %d complete: %d → %d tokens",
                stage_num, tokens_before, current_tokens,
            )

            if current_tokens <= self.threshold_tokens:
                break

        return result

    # ------------------------------------------------------------------
    # Stage 1: Trim old tool results
    # ------------------------------------------------------------------

    async def _stage1_trim_old_results(
        self, messages: list[Message], result: CompactionResult
    ) -> None:
        """Replace old tool results with 1-line summaries.

        Keep the last `keep_full_results` tool_result messages in full;
        replace older ones with a brief summary.
        """
        # Find all tool_result message indices
        result_indices: list[int] = []
        for i, msg in enumerate(messages):
            if msg.role == "tool_result":
                result_indices.append(i)

        if len(result_indices) <= self.config.keep_full_results:
            return  # Nothing to trim

        # Trim older results (keep the last N)
        to_trim = result_indices[: -self.config.keep_full_results]
        for idx in to_trim:
            msg = messages[idx]
            if isinstance(msg.content, list):
                new_blocks = []
                for block in msg.content:
                    if block.type == ContentBlockType.TOOL_RESULT and block.text:
                        original_lines = block.text.count("\n") + 1
                        first_line = block.text.split("\n", 1)[0][:100]
                        trimmed_text = f"[trimmed — {original_lines} lines] {first_line}"
                        new_blocks.append(ContentBlock(
                            type=block.type,
                            tool_call_id=block.tool_call_id,
                            text=trimmed_text,
                        ))
                    else:
                        new_blocks.append(block)
                messages[idx] = Message(role=msg.role, content=new_blocks)
            elif isinstance(msg.content, str) and len(msg.content) > 200:
                original_lines = msg.content.count("\n") + 1
                first_line = msg.content.split("\n", 1)[0][:100]
                messages[idx] = Message(
                    role=msg.role,
                    content=f"[trimmed — {original_lines} lines] {first_line}",
                )

    # ------------------------------------------------------------------
    # Stage 2: Truncate verbose tool output
    # ------------------------------------------------------------------

    async def _stage2_truncate_verbose(
        self, messages: list[Message], result: CompactionResult
    ) -> None:
        """Cap each remaining tool result at max_result_lines."""
        max_lines = self.config.max_result_lines

        for i, msg in enumerate(messages):
            if msg.role != "tool_result":
                continue

            if isinstance(msg.content, list):
                new_blocks = []
                changed = False
                for block in msg.content:
                    if block.type == ContentBlockType.TOOL_RESULT and block.text:
                        truncated = self._truncate_text(block.text, max_lines)
                        if truncated != block.text:
                            new_blocks.append(ContentBlock(
                                type=block.type,
                                tool_call_id=block.tool_call_id,
                                text=truncated,
                            ))
                            changed = True
                        else:
                            new_blocks.append(block)
                    else:
                        new_blocks.append(block)
                if changed:
                    messages[i] = Message(role=msg.role, content=new_blocks)
            elif isinstance(msg.content, str):
                truncated = self._truncate_text(msg.content, max_lines)
                if truncated != msg.content:
                    messages[i] = Message(role=msg.role, content=truncated)

    # ------------------------------------------------------------------
    # Stage 3: Extract session memory (LLM-based)
    # ------------------------------------------------------------------

    async def _stage3_extract_memory(
        self, messages: list[Message], result: CompactionResult
    ) -> None:
        """Ask LLM to extract key findings from the conversation.

        On failure: skip to stage 5 (don't retry).
        """
        if not self.llm:
            logger.debug("Stage 3 skipped: no LLM available")
            return

        # Build a condensed text of the conversation for the LLM
        conversation_text = self._messages_to_text(messages)
        if not conversation_text:
            return

        extract_prompt = (
            "You are a memory extraction assistant. Given a conversation "
            "between a coding agent and its tools, extract the key findings "
            "and decisions into a concise summary.\n\n"
            "Focus on:\n"
            "- What files were examined and what was learned\n"
            "- What changes were made and why\n"
            "- What worked and what didn't\n"
            "- Important constraints or edge cases discovered\n\n"
            "Do NOT include raw tool output. Summarize in your own words.\n"
            "Output a markdown summary, max 300 words."
        )

        try:
            response = await self.llm.chat([
                Message(role="system", content=extract_prompt),
                Message(role="user", content=conversation_text),
            ])
            summary = response.content.strip()

            if summary:
                result.summary_generated = True
                # Replace middle messages with the summary
                self._replace_middle_with_summary(messages, summary, "session_extract")
                logger.info("Stage 3: extracted session memory (%d chars)", len(summary))
        except Exception as e:
            logger.warning("Stage 3 LLM call failed, skipping to stage 5: %s", e)

    # ------------------------------------------------------------------
    # Stage 4: Full history summarization (LLM-based)
    # ------------------------------------------------------------------

    async def _stage4_summarize_history(
        self, messages: list[Message], result: CompactionResult
    ) -> None:
        """Ask LLM to produce a full summary of everything done so far.

        On failure: skip to stage 5 (don't retry).
        """
        if not self.llm:
            logger.debug("Stage 4 skipped: no LLM available")
            return

        conversation_text = self._messages_to_text(messages)
        if not conversation_text:
            return

        summarize_prompt = (
            "Summarize the following coding session conversation.\n"
            "Include: what the user asked, what was done, what files were "
            "changed, current status, and any unresolved issues.\n"
            "Be concise — max 200 words."
        )

        try:
            response = await self.llm.chat([
                Message(role="system", content=summarize_prompt),
                Message(role="user", content=conversation_text),
            ])
            summary = response.content.strip()

            if summary:
                result.summary_generated = True
                self._replace_middle_with_summary(messages, summary, "full_summary")
                logger.info("Stage 4: full summary (%d chars)", len(summary))
        except Exception as e:
            logger.warning("Stage 4 LLM call failed, skipping to stage 5: %s", e)

    # ------------------------------------------------------------------
    # Stage 5: Drop oldest messages (last resort)
    # ------------------------------------------------------------------

    async def _stage5_drop_oldest(
        self, messages: list[Message], result: CompactionResult
    ) -> None:
        """Drop oldest messages until under threshold.

        Always preserves:
        - System prompt (index 0)
        - Last N messages (most recent context)
        """
        preserve_tail = self.config.preserve_last_n_messages

        while (
            estimate_tokens_heuristic(messages) > self.threshold_tokens
            and len(messages) > preserve_tail + 1  # +1 for system prompt
        ):
            # Remove the second message (index 1), preserving system prompt at 0
            messages.pop(1)
            result.messages_dropped += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_text(text: str, max_lines: int) -> str:
        """Truncate text to max_lines, adding a truncation notice."""
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        truncated = "\n".join(lines[:max_lines])
        remaining = len(lines) - max_lines
        return f"{truncated}\n[truncated — {remaining} more lines]"

    @staticmethod
    def _messages_to_text(messages: list[Message]) -> str:
        """Flatten messages into a text block for summarization."""
        parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                continue  # Don't include system prompt
            text = msg.text_content()
            if text:
                parts.append(f"[{msg.role}] {text[:1000]}")
        return "\n\n".join(parts)

    @staticmethod
    def _replace_middle_with_summary(
        messages: list[Message], summary: str, label: str
    ) -> None:
        """Replace all messages between system prompt and the last 4 with a summary."""
        if len(messages) <= 5:
            return

        system = messages[0]  # Preserve system prompt
        tail = messages[-4:]  # Preserve recent messages

        summary_msg = Message(
            role="user",
            content=f"[{label}] Previous conversation summary:\n\n{summary}",
        )

        messages.clear()
        messages.append(system)
        messages.append(summary_msg)
        messages.extend(tail)
