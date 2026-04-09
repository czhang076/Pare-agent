"""Tests for ContextManager — the top-level coordinator."""

from pathlib import Path

import pytest

from pare.context.manager import ContextManager
from pare.llm.base import Message


@pytest.fixture
def ctx(tmp_path: Path) -> ContextManager:
    return ContextManager(cwd=tmp_path)


class TestContextManager:
    def test_set_system_prompt(self, ctx: ContextManager):
        ctx.set_system_prompt("You are Pare.")
        msgs = ctx.get_messages()
        assert len(msgs) == 1
        assert msgs[0].role == "system"
        assert "Pare" in msgs[0].content

    def test_add_user_message(self, ctx: ContextManager):
        ctx.set_system_prompt("system")
        ctx.add_user_message("Fix the bug")
        msgs = ctx.get_messages()
        assert len(msgs) == 2
        assert msgs[1].role == "user"
        assert msgs[1].content == "Fix the bug"

    def test_set_task_updates_memory(self, ctx: ContextManager):
        ctx.set_system_prompt("system")
        ctx.set_task("Add login feature")
        msgs = ctx.get_messages()
        # Memory index should be injected into system message
        assert "login feature" in msgs[0].content

    def test_memory_index_in_system_message(self, ctx: ContextManager):
        ctx.set_system_prompt("You are Pare.")
        ctx.update_memory("Structure", "src/ (3 files)")
        msgs = ctx.get_messages()
        assert "Structure" in msgs[0].content
        assert "3 files" in msgs[0].content

    def test_topic_loading(self, ctx: ContextManager):
        ctx.set_system_prompt("system")
        ctx.topics.write("auth", "## Auth Module\nUses JWT")
        ctx.load_topics(["auth"])
        msgs = ctx.get_messages()
        assert "JWT" in msgs[0].content

    def test_topic_loading_nonexistent_skipped(self, ctx: ContextManager):
        ctx.set_system_prompt("system")
        ctx.load_topics(["nonexistent"])
        msgs = ctx.get_messages()
        assert len(msgs) == 1  # Just system, no crash

    def test_conversation_ordering(self, ctx: ContextManager):
        ctx.set_system_prompt("system")
        ctx.add_user_message("task")
        ctx.add_assistant_message(Message(role="assistant", content="I'll help"))
        ctx.add_user_message("continue")

        msgs = ctx.get_messages()
        assert len(msgs) == 4
        assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]

    def test_token_count(self, ctx: ContextManager):
        ctx.set_system_prompt("You are a coding agent. " * 100)
        count = ctx.token_count()
        assert count > 0

    def test_search_history(self, ctx: ContextManager):
        ctx.add_user_message("Fix the login bug")
        ctx.add_user_message("Add unit tests")
        results = ctx.search_history("login")
        assert len(results) == 1
        assert "login" in results[0]

    def test_close(self, ctx: ContextManager):
        ctx.add_user_message("hello")
        ctx.close()
        # Should not raise

    @pytest.mark.asyncio
    async def test_needs_compaction_false_for_small(self, ctx: ContextManager):
        ctx.set_system_prompt("short")
        ctx.add_user_message("hello")
        assert not ctx.needs_compaction()

    @pytest.mark.asyncio
    async def test_compact_no_op_when_small(self, ctx: ContextManager):
        ctx.set_system_prompt("short")
        ctx.add_user_message("hello")
        result = await ctx.compact()
        assert result.stage_reached == 0
