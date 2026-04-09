"""Tests for the 5-stage compaction pipeline.

Stages 1-2 and 5 are deterministic (no LLM calls).
Stages 3-4 use a mock LLM adapter.
"""

from __future__ import annotations

import pytest

from pare.context.compactor import CompactionConfig, CompactionPipeline
from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    LLMResponse,
    Message,
    StopReason,
    TokenUsage,
)


def _make_messages(n_tool_results: int, lines_per_result: int = 100) -> list[Message]:
    """Build a message list with system + user + N tool result pairs."""
    msgs = [
        Message(role="system", content="You are a coding agent. " * 50),
        Message(role="user", content="Fix the bug in main.py"),
    ]
    for i in range(n_tool_results):
        # Assistant with tool call
        msgs.append(Message(role="assistant", content=f"Let me check file {i}"))
        # Tool result with verbose output
        output = "\n".join(f"line {j}: content" for j in range(lines_per_result))
        msgs.append(Message(role="tool_result", content=[
            ContentBlock(
                type=ContentBlockType.TOOL_RESULT,
                tool_call_id=f"tc_{i}",
                text=output,
            ),
        ]))
    return msgs


class TestStage1TrimOldResults:
    @pytest.mark.asyncio
    async def test_trims_old_results(self):
        config = CompactionConfig(
            max_context_tokens=1_000_000,  # High threshold so we control when stages run
            keep_full_results=2,
        )
        pipeline = CompactionPipeline(config)
        msgs = _make_messages(5, lines_per_result=50)

        # Manually call stage 1
        from pare.context.compactor import CompactionResult
        result = CompactionResult(stage_reached=1, tokens_before=0, tokens_after=0)
        await pipeline._stage1_trim_old_results(msgs, result)

        # First 3 tool results should be trimmed, last 2 kept
        trimmed_count = 0
        for msg in msgs:
            if msg.role == "tool_result" and isinstance(msg.content, list):
                for block in msg.content:
                    if "[trimmed" in (block.text or ""):
                        trimmed_count += 1
        assert trimmed_count == 3

    @pytest.mark.asyncio
    async def test_no_trim_when_few_results(self):
        config = CompactionConfig(keep_full_results=5)
        pipeline = CompactionPipeline(config)
        msgs = _make_messages(3)

        from pare.context.compactor import CompactionResult
        result = CompactionResult(stage_reached=1, tokens_before=0, tokens_after=0)
        await pipeline._stage1_trim_old_results(msgs, result)

        # Nothing should be trimmed
        for msg in msgs:
            if msg.role == "tool_result" and isinstance(msg.content, list):
                for block in msg.content:
                    assert "[trimmed" not in (block.text or "")


class TestStage2TruncateVerbose:
    @pytest.mark.asyncio
    async def test_truncates_long_output(self):
        config = CompactionConfig(max_result_lines=10)
        pipeline = CompactionPipeline(config)
        msgs = _make_messages(1, lines_per_result=100)

        from pare.context.compactor import CompactionResult
        result = CompactionResult(stage_reached=2, tokens_before=0, tokens_after=0)
        await pipeline._stage2_truncate_verbose(msgs, result)

        # Find the tool result
        for msg in msgs:
            if msg.role == "tool_result" and isinstance(msg.content, list):
                for block in msg.content:
                    if block.type == ContentBlockType.TOOL_RESULT:
                        lines = block.text.splitlines()
                        assert len(lines) <= 12  # 10 + truncation notice
                        assert "truncated" in block.text


class TestStage5DropOldest:
    @pytest.mark.asyncio
    async def test_drops_oldest_messages(self):
        # Set a very low threshold so stage 5 has to drop messages
        config = CompactionConfig(
            max_context_tokens=500,
            threshold_ratio=0.5,  # 250 token threshold
            preserve_last_n_messages=2,
        )
        pipeline = CompactionPipeline(config)
        msgs = _make_messages(10, lines_per_result=20)
        original_count = len(msgs)

        from pare.context.compactor import CompactionResult
        result = CompactionResult(stage_reached=5, tokens_before=0, tokens_after=0)
        await pipeline._stage5_drop_oldest(msgs, result)

        # Should have dropped messages
        assert len(msgs) < original_count
        assert result.messages_dropped > 0
        # System prompt preserved
        assert msgs[0].role == "system"
        # At least preserve_last_n_messages at the end
        assert len(msgs) >= 3  # system + at least 2

    @pytest.mark.asyncio
    async def test_preserves_system_prompt(self):
        config = CompactionConfig(
            max_context_tokens=100,
            threshold_ratio=0.1,
            preserve_last_n_messages=1,
        )
        pipeline = CompactionPipeline(config)
        msgs = _make_messages(5)

        from pare.context.compactor import CompactionResult
        result = CompactionResult(stage_reached=5, tokens_before=0, tokens_after=0)
        await pipeline._stage5_drop_oldest(msgs, result)

        assert msgs[0].role == "system"


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_no_compaction_when_under_threshold(self):
        config = CompactionConfig(max_context_tokens=1_000_000)
        pipeline = CompactionPipeline(config)
        msgs = _make_messages(2, lines_per_result=5)

        result = await pipeline.compact(msgs)
        assert result.stage_reached == 0

    @pytest.mark.asyncio
    async def test_stages_run_in_order(self):
        # Low threshold to force compaction
        config = CompactionConfig(
            max_context_tokens=500,
            threshold_ratio=0.3,
            keep_full_results=2,
            max_result_lines=10,
            preserve_last_n_messages=2,
        )
        pipeline = CompactionPipeline(config)  # No LLM — stages 3/4 skip
        msgs = _make_messages(10, lines_per_result=50)

        result = await pipeline.compact(msgs)
        assert result.stage_reached >= 1
        assert result.tokens_after <= result.tokens_before

    @pytest.mark.asyncio
    async def test_needs_compaction(self):
        config = CompactionConfig(max_context_tokens=100, threshold_ratio=0.5)
        pipeline = CompactionPipeline(config)

        small = [Message(role="system", content="hi")]
        assert not pipeline.needs_compaction(small)

        large = _make_messages(20, lines_per_result=100)
        assert pipeline.needs_compaction(large)


class TestHelpers:
    def test_truncate_text(self):
        text = "\n".join(f"line {i}" for i in range(100))
        result = CompactionPipeline._truncate_text(text, 10)
        assert "truncated" in result
        assert result.count("\n") <= 11  # 10 lines + truncation notice

    def test_truncate_text_short(self):
        text = "short\ntext"
        result = CompactionPipeline._truncate_text(text, 10)
        assert result == text  # Unchanged

    def test_replace_middle_with_summary(self):
        msgs = [
            Message(role="system", content="system"),
            Message(role="user", content="task"),
            Message(role="assistant", content="step 1"),
            Message(role="user", content="continue"),
            Message(role="assistant", content="step 2"),
            Message(role="user", content="continue 2"),
            Message(role="assistant", content="step 3"),
            Message(role="tool_result", content="result"),
            Message(role="assistant", content="done"),
            Message(role="user", content="latest"),
        ]
        CompactionPipeline._replace_middle_with_summary(msgs, "summary text", "test")

        assert msgs[0].role == "system"
        assert "summary" in msgs[1].content
        assert len(msgs) == 6  # system + summary + last 4

    def test_replace_middle_too_few_messages(self):
        msgs = [
            Message(role="system", content="system"),
            Message(role="user", content="task"),
        ]
        CompactionPipeline._replace_middle_with_summary(msgs, "summary", "test")
        assert len(msgs) == 2  # Unchanged
