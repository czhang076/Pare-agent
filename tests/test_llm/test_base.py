"""Tests for pare/llm/base.py — core data types."""

from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    Message,
    ModelProfile,
    StopReason,
    TokenUsage,
    ToolCallRequest,
    get_profile,
)


class TestTokenUsage:
    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_with_cache_tokens(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=30,
            cache_create_tokens=20,
        )
        assert usage.total_tokens == 150  # cache tokens don't count toward total


class TestMessage:
    def test_text_content_from_string(self):
        msg = Message(role="user", content="hello")
        assert msg.text_content() == "hello"

    def test_text_content_from_blocks(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type=ContentBlockType.TEXT, text="I'll help "),
                ContentBlock(type=ContentBlockType.TEXT, text="with that."),
            ],
        )
        assert msg.text_content() == "I'll help with that."

    def test_text_content_ignores_tool_blocks(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type=ContentBlockType.TEXT, text="Let me read the file."),
                ContentBlock(
                    type=ContentBlockType.TOOL_USE,
                    tool_call=ToolCallRequest(
                        id="tc_1",
                        name="file_read",
                        arguments={"file_path": "main.py"},
                    ),
                ),
            ],
        )
        assert msg.text_content() == "Let me read the file."

    def test_tool_calls_extraction(self):
        tc = ToolCallRequest(id="tc_1", name="bash", arguments={"command": "ls"})
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type=ContentBlockType.TEXT, text="Running ls"),
                ContentBlock(type=ContentBlockType.TOOL_USE, tool_call=tc),
            ],
        )
        calls = msg.tool_calls()
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls"}

    def test_tool_calls_from_string_content(self):
        msg = Message(role="user", content="just text")
        assert msg.tool_calls() == []


class TestModelProfile:
    def test_defaults(self):
        profile = ModelProfile()
        assert profile.supports_native_tool_use is True
        assert profile.supports_structured_json is True
        assert profile.max_context_tokens == 128_000

    def test_open_source_profile(self):
        profile = ModelProfile(
            supports_native_tool_use=False,
            supports_structured_json=False,
            tool_call_format="text",
        )
        assert profile.supports_native_tool_use is False
        assert profile.tool_call_format == "text"


class TestGetProfile:
    def test_known_model(self):
        profile = get_profile("claude-sonnet-4-20250514")
        assert profile.supports_cache_control is True
        assert profile.max_context_tokens == 200_000

    def test_unknown_model_returns_defaults(self):
        profile = get_profile("some-unknown-model-v99")
        assert profile.supports_native_tool_use is True  # default
        assert profile.max_context_tokens == 128_000  # default

    def test_deepseek_profile(self):
        profile = get_profile("deepseek/deepseek-chat")
        assert profile.supports_native_tool_use is False
        assert profile.tool_call_format == "text"


class TestStopReason:
    def test_enum_values(self):
        assert StopReason.END_TURN.value == "end_turn"
        assert StopReason.TOOL_USE.value == "tool_use"
        assert StopReason.MAX_TOKENS.value == "max_tokens"
