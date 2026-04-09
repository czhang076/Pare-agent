"""Tests for MemoryIndex and TopicStore."""

from pathlib import Path

import pytest

from pare.context.memory import MemoryIndex, TopicStore


@pytest.fixture
def mem(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(tmp_path, filename="MEMORY.md")


@pytest.fixture
def topics(tmp_path: Path) -> TopicStore:
    return TopicStore(tmp_path)


class TestMemoryIndex:
    def test_load_empty(self, mem: MemoryIndex):
        content = mem.load()
        assert content == ""

    def test_update_and_save(self, mem: MemoryIndex):
        mem.load()
        mem.update_section("Task", "Fix the bug")
        mem.update_section("Structure", "src/ (3 files)")
        mem.save()

        # Re-read from disk
        mem2 = MemoryIndex(mem.path.parent.parent, filename="MEMORY.md")
        content = mem2.load()
        assert "## Task" in content
        assert "Fix the bug" in content
        assert "## Structure" in content

    def test_get_content_auto_loads(self, mem: MemoryIndex):
        mem.load()
        mem.update_section("Status", "In progress")
        mem.save()

        mem2 = MemoryIndex(mem.path.parent.parent, filename="MEMORY.md")
        content = mem2.get_content()  # Should auto-load
        assert "In progress" in content

    def test_remove_section(self, mem: MemoryIndex):
        mem.load()
        mem.update_section("A", "aaa")
        mem.update_section("B", "bbb")
        mem.remove_section("A")
        mem.save()

        content = mem.get_content()
        assert "## A" not in content
        assert "## B" in content

    def test_get_section(self, mem: MemoryIndex):
        mem.load()
        mem.update_section("Task", "Do something")
        assert mem.get_section("Task") == "Do something"
        assert mem.get_section("Nonexistent") == ""

    def test_round_trip_with_title(self, mem: MemoryIndex):
        mem.load()
        mem._sections["__title__"] = "# Project: test-repo"
        mem.update_section("Task", "Fix bug")
        mem.save()

        mem2 = MemoryIndex(mem.path.parent.parent, filename="MEMORY.md")
        mem2.load()
        assert mem2.get_section("__title__") == "# Project: test-repo"
        assert mem2.get_section("Task") == "Fix bug"

    def test_clear(self, mem: MemoryIndex):
        mem.load()
        mem.update_section("A", "aaa")
        mem.clear()
        assert mem.sections == {}


class TestTopicStore:
    def test_write_and_read(self, topics: TopicStore):
        topics.write("auth_module", "## Auth\nUses JWT tokens")
        content = topics.read("auth_module")
        assert "JWT tokens" in content

    def test_read_nonexistent(self, topics: TopicStore):
        assert topics.read("nonexistent") == ""

    def test_exists(self, topics: TopicStore):
        assert not topics.exists("auth")
        topics.write("auth", "content")
        assert topics.exists("auth")

    def test_delete(self, topics: TopicStore):
        topics.write("temp", "temporary")
        assert topics.delete("temp") is True
        assert not topics.exists("temp")
        assert topics.delete("temp") is False

    def test_list_topics(self, topics: TopicStore):
        topics.write("beta", "b")
        topics.write("alpha", "a")
        names = topics.list_topics()
        assert names == ["alpha", "beta"]  # Sorted

    def test_sanitizes_topic_name(self, topics: TopicStore):
        topics.write("my topic/with spaces!", "content")
        assert topics.exists("my_topic_with_spaces_")
