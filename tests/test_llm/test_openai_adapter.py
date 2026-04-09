"""Tests for pare/llm/openai_adapter.py.

These tests cover the unit-testable parts of the adapter without making
real API calls: message translation, text-based tool call parsing,
temperature clamping, and the factory function.
"""

import json

from pare.llm.base import (
    ContentBlock,
    ContentBlockType,
    Message,
    ModelProfile,
    ToolCallRequest,
    ToolSchema,
    get_profile,
)
from pare.llm.openai_adapter import (
    OpenAIAdapter,
    _build_openai_messages,
    _build_openai_tools,
    _build_tool_system_prompt,
    _parse_text_tool_calls,
    _stop_reason_from_openai,
)
from pare.llm.base import StopReason


# ---------------------------------------------------------------------------
# Stop reason translation
# ---------------------------------------------------------------------------


class TestStopReason:
    def test_stop(self):
        assert _stop_reason_from_openai("stop") == StopReason.END_TURN

    def test_tool_calls(self):
        assert _stop_reason_from_openai("tool_calls") == StopReason.TOOL_USE

    def test_length(self):
        assert _stop_reason_from_openai("length") == StopReason.MAX_TOKENS

    def test_unknown(self):
        assert _stop_reason_from_openai("something_new") == StopReason.END_TURN

    def test_none(self):
        assert _stop_reason_from_openai(None) == StopReason.END_TURN


# ---------------------------------------------------------------------------
# Text-based tool call parsing (for models without native tool_use)
# ---------------------------------------------------------------------------


class TestParseTextToolCalls:
    def test_xml_format(self):
        text = (
            'I need to read the file.\n'
            '<tool_call>{"name": "file_read", "arguments": {"file_path": "main.py"}}</tool_call>'
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "file_read"
        assert calls[0].arguments == {"file_path": "main.py"}

    def test_multiple_xml_calls(self):
        text = (
            '<tool_call>{"name": "file_read", "arguments": {"file_path": "a.py"}}</tool_call>\n'
            'Now let me edit it.\n'
            '<tool_call>{"name": "file_edit", "arguments": {"file_path": "a.py", '
            '"old_str": "x", "new_str": "y"}}</tool_call>'
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].name == "file_read"
        assert calls[1].name == "file_edit"

    def test_code_fence_format(self):
        text = (
            'Let me search for that.\n'
            '```tool_call\n'
            '{"name": "search", "arguments": {"pattern": "def main"}}\n'
            '```'
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_json_fence_format(self):
        text = (
            '```json\n'
            '{"name": "bash", "arguments": {"command": "ls -la"}}\n'
            '```'
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_no_tool_calls(self):
        text = "I think the issue is in the auth module. Let me explain..."
        calls = _parse_text_tool_calls(text)
        assert calls == []

    def test_malformed_json_ignored(self):
        text = '<tool_call>not valid json</tool_call>'
        calls = _parse_text_tool_calls(text)
        assert calls == []

    def test_missing_name_ignored(self):
        text = '<tool_call>{"arguments": {"key": "val"}}</tool_call>'
        calls = _parse_text_tool_calls(text)
        assert calls == []

    def test_parameters_key_as_alias(self):
        """Some models use 'parameters' instead of 'arguments'."""
        text = '<tool_call>{"name": "bash", "parameters": {"command": "pwd"}}</tool_call>'
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {"command": "pwd"}

    def test_ids_are_unique(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls = _parse_text_tool_calls(text)
        assert calls[0].id != calls[1].id


# ---------------------------------------------------------------------------
# Tool system prompt generation
# ---------------------------------------------------------------------------


class TestBuildToolSystemPrompt:
    def test_contains_tool_info(self):
        tools = [
            ToolSchema(
                name="file_read",
                description="Read a file from disk.",
                parameters={"type": "object", "properties": {"file_path": {"type": "string"}}},
            ),
        ]
        prompt = _build_tool_system_prompt(tools)
        assert "file_read" in prompt
        assert "Read a file from disk." in prompt
        assert "<tool_call>" in prompt  # instruction format
        assert "file_path" in prompt

    def test_multiple_tools(self):
        tools = [
            ToolSchema(name="bash", description="Run a command.", parameters={}),
            ToolSchema(name="search", description="Search files.", parameters={}),
        ]
        prompt = _build_tool_system_prompt(tools)
        assert "### bash" in prompt
        assert "### search" in prompt


# ---------------------------------------------------------------------------
# OpenAI tool schema translation
# ---------------------------------------------------------------------------


class TestBuildOpenAITools:
    def test_single_tool(self):
        tools = [
            ToolSchema(
                name="file_read",
                description="Read a file.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]
        result = _build_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "file_read"
        assert result[0]["function"]["description"] == "Read a file."
        assert "path" in result[0]["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


_NATIVE_PROFILE = ModelProfile(supports_native_tool_use=True)
_TEXT_PROFILE = ModelProfile(supports_native_tool_use=False, tool_call_format="text")


class TestBuildOpenAIMessages:
    def test_system_message(self):
        msgs = [Message(role="system", content="You are helpful.")]
        result = _build_openai_messages(msgs, None, _NATIVE_PROFILE)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_user_message(self):
        msgs = [Message(role="user", content="Hello")]
        result = _build_openai_messages(msgs, None, _NATIVE_PROFILE)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_string_content(self):
        msgs = [Message(role="assistant", content="Sure, I'll help.")]
        result = _build_openai_messages(msgs, None, _NATIVE_PROFILE)
        assert result == [{"role": "assistant", "content": "Sure, I'll help."}]

    def test_tool_result_native_mode(self):
        msgs = [
            Message(
                role="tool_result",
                content=[
                    ContentBlock(
                        type=ContentBlockType.TOOL_RESULT,
                        tool_call_id="tc_1",
                        text="file contents here",
                    ),
                ],
            ),
        ]
        result = _build_openai_messages(msgs, None, _NATIVE_PROFILE)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tc_1"
        assert result[0]["content"] == "file contents here"

    def test_tool_result_text_mode(self):
        """In text mode, tool results become user messages."""
        msgs = [
            Message(
                role="tool_result",
                content=[
                    ContentBlock(
                        type=ContentBlockType.TOOL_RESULT,
                        tool_call_id="tc_1",
                        text="result data",
                    ),
                ],
            ),
        ]
        result = _build_openai_messages(msgs, None, _TEXT_PROFILE)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "result data" in result[0]["content"]

    def test_system_message_with_tools_text_mode(self):
        """In text mode, tools are injected into the system prompt."""
        tools = [ToolSchema(name="bash", description="Run cmd.", parameters={})]
        msgs = [Message(role="system", content="You are an agent.")]
        result = _build_openai_messages(msgs, tools, _TEXT_PROFILE)
        assert len(result) == 1
        assert "bash" in result[0]["content"]
        assert "<tool_call>" in result[0]["content"]

    def test_system_message_with_tools_native_mode(self):
        """In native mode, tools are NOT injected into system prompt."""
        tools = [ToolSchema(name="bash", description="Run cmd.", parameters={})]
        msgs = [Message(role="system", content="You are an agent.")]
        result = _build_openai_messages(msgs, tools, _NATIVE_PROFILE)
        assert result[0]["content"] == "You are an agent."

    def test_assistant_with_tool_calls_native(self):
        tc = ToolCallRequest(id="tc_1", name="bash", arguments={"command": "ls"})
        msgs = [
            Message(
                role="assistant",
                content=[
                    ContentBlock(type=ContentBlockType.TEXT, text="Running ls."),
                    ContentBlock(type=ContentBlockType.TOOL_USE, tool_call=tc),
                ],
            ),
        ]
        result = _build_openai_messages(msgs, None, _NATIVE_PROFILE)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Running ls."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "bash"
        # Arguments should be JSON string in OpenAI format
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}


# ---------------------------------------------------------------------------
# Temperature clamping
# ---------------------------------------------------------------------------


class TestTemperatureClamping:
    def test_minimax_zero_clamped(self):
        adapter = OpenAIAdapter(
            model="MiniMax-M2.5",
            base_url="https://api.minimax.io/v1",
            api_key="test",
        )
        assert adapter._clamp_temperature(0.0) == 0.01

    def test_minimax_normal_passthrough(self):
        adapter = OpenAIAdapter(
            model="MiniMax-M2.5",
            base_url="https://api.minimax.io/v1",
            api_key="test",
        )
        assert adapter._clamp_temperature(0.7) == 0.7

    def test_minimax_over_one_clamped(self):
        adapter = OpenAIAdapter(
            model="MiniMax-M2.5",
            base_url="https://api.minimax.io/v1",
            api_key="test",
        )
        assert adapter._clamp_temperature(1.5) == 1.0

    def test_non_minimax_no_clamping(self):
        adapter = OpenAIAdapter(model="gpt-4o", api_key="test")
        assert adapter._clamp_temperature(0.0) == 0.0
        assert adapter._clamp_temperature(2.0) == 2.0


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateLLM:
    def test_create_anthropic(self):
        from pare.llm import create_llm
        from pare.llm.anthropic_adapter import AnthropicAdapter

        llm = create_llm("anthropic", api_key="test-key")
        assert isinstance(llm, AnthropicAdapter)
        assert llm.model == "claude-sonnet-4-20250514"

    def test_create_minimax(self):
        from pare.llm import create_llm

        llm = create_llm("minimax", api_key="test-key")
        assert isinstance(llm, OpenAIAdapter)
        assert llm.model == "MiniMax-M2.7"
        assert llm._base_url == "https://api.minimaxi.com/v1"

    def test_create_openai(self):
        from pare.llm import create_llm

        llm = create_llm("openai", model="gpt-4o", api_key="test-key")
        assert isinstance(llm, OpenAIAdapter)
        assert llm.model == "gpt-4o"

    def test_create_openrouter(self):
        from pare.llm import create_llm

        llm = create_llm("openrouter", api_key="test-key")
        assert isinstance(llm, OpenAIAdapter)
        assert llm._base_url == "https://openrouter.ai/api/v1"

    def test_create_custom_endpoint(self):
        from pare.llm import create_llm

        llm = create_llm(
            "openai",
            model="local-model",
            base_url="http://localhost:8000/v1",
            api_key="dummy",
        )
        assert isinstance(llm, OpenAIAdapter)
        assert llm._base_url == "http://localhost:8000/v1"

    def test_minimax_profile_loaded(self):
        profile = get_profile("MiniMax-M2.5")
        assert profile.supports_native_tool_use is True
        assert profile.max_context_tokens == 204_800
