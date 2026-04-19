"""Tests for pare/llm/output_parser.py.

The output parser is the most critical piece for open-source model support.
These tests cover real-world LLM output patterns observed from DeepSeek,
Qwen, and various models accessed through OpenRouter.
"""

import pytest

from pare.llm.output_parser import ParseError, parse_json_response, try_parse_json_response


class TestDirectParse:
    """Stage 1: raw string is already valid JSON."""

    def test_clean_json(self):
        raw = '{"summary": "fix bug", "steps": [{"goal": "read file"}]}'
        result = parse_json_response(raw)
        assert result["summary"] == "fix bug"
        assert len(result["steps"]) == 1

    def test_clean_json_with_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = parse_json_response(raw)
        assert result["key"] == "value"

    def test_empty_object(self):
        result = parse_json_response("{}")
        assert result == {}


class TestStripCodeFences:
    """Stage 2: JSON wrapped in markdown code fences."""

    def test_json_code_fence(self):
        raw = '```json\n{"status": "completed", "summary": "done"}\n```'
        result = parse_json_response(raw)
        assert result["status"] == "completed"

    def test_plain_code_fence(self):
        raw = '```\n{"status": "completed"}\n```'
        result = parse_json_response(raw)
        assert result["status"] == "completed"

    def test_code_fence_with_preamble(self):
        raw = 'Here is the plan:\n```json\n{"summary": "fix auth"}\n```\nLet me know!'
        result = parse_json_response(raw)
        assert result["summary"] == "fix auth"

    def test_code_fence_with_trailing_comma(self):
        raw = '```json\n{"key": "value",}\n```'
        result = parse_json_response(raw)
        assert result["key"] == "value"

    def test_uppercase_json_fence(self):
        raw = '```JSON\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        assert result["key"] == "value"


class TestBraceExtraction:
    """Stage 3: JSON buried in natural language text."""

    def test_preamble_and_postamble(self):
        raw = 'I think the plan should be: {"summary": "refactor", "steps": []} and that\'s it.'
        result = parse_json_response(raw)
        assert result["summary"] == "refactor"

    def test_nested_braces(self):
        raw = 'Result: {"outer": {"inner": "value"}, "list": [1, 2]}'
        result = parse_json_response(raw)
        assert result["outer"]["inner"] == "value"

    def test_braces_in_strings(self):
        raw = 'Output: {"message": "use {braces} here", "count": 1}'
        result = parse_json_response(raw)
        assert result["message"] == "use {braces} here"
        assert result["count"] == 1


class TestSanitize:
    """Stage 4: JSON with minor formatting issues."""

    def test_trailing_comma_in_object(self):
        raw = '{"key": "value", "key2": "value2",}'
        result = parse_json_response(raw)
        assert result["key"] == "value"

    def test_trailing_comma_in_array(self):
        raw = '{"items": ["a", "b", "c",]}'
        result = parse_json_response(raw)
        assert result["items"] == ["a", "b", "c"]

    def test_trailing_comma_nested(self):
        raw = '{"data": {"nested": true,},}'
        result = parse_json_response(raw)
        assert result["data"]["nested"] is True


class TestParseFailure:
    """Stage 5: all stages fail."""

    def test_no_json_at_all(self):
        raw = "I don't know how to create a plan for this task."
        with pytest.raises(ParseError) as exc_info:
            parse_json_response(raw)
        assert "direct" in exc_info.value.stages_tried
        assert raw in exc_info.value.raw

    def test_pure_prose_raises_parse_error(self):
        """Input with no JSON-like structure at all still fails loudly."""
        raw = "I don't know how to create a plan for this task."
        with pytest.raises(ParseError):
            parse_json_response(raw)

    def test_array_extracts_first_object(self):
        """An array input triggers brace extraction, finding the first dict."""
        raw = '[{"item": 1}, {"item": 2}]'
        result = parse_json_response(raw)
        # Brace extraction finds the first { ... } which is {"item": 1}
        assert result == {"item": 1}

    def test_just_a_string(self):
        raw = '"hello world"'
        with pytest.raises(ParseError):
            parse_json_response(raw)


class TestTryParseJsonResponse:
    """Non-throwing variant."""

    def test_success(self):
        result = try_parse_json_response('{"ok": true}')
        assert isinstance(result, dict)
        assert result["ok"] is True

    def test_failure_returns_parse_error(self):
        result = try_parse_json_response("not json")
        assert isinstance(result, ParseError)


class TestRealWorldPatterns:
    """Patterns observed from actual open-source model outputs."""

    def test_deepseek_style_preamble(self):
        """DeepSeek often adds explanation before the JSON."""
        raw = (
            "Based on the repository structure, here is my plan:\n\n"
            '```json\n'
            '{\n'
            '  "summary": "Add authentication middleware",\n'
            '  "steps": [\n'
            '    {"step_number": 1, "goal": "Read existing middleware"}\n'
            '  ]\n'
            '}\n'
            '```\n\n'
            "This plan covers the essential changes needed."
        )
        result = parse_json_response(raw)
        assert result["summary"] == "Add authentication middleware"
        assert len(result["steps"]) == 1

    def test_qwen_style_no_fence(self):
        """Qwen sometimes outputs JSON directly with surrounding text."""
        raw = (
            'Sure! Here\'s the plan:\n'
            '{"summary": "Fix the login bug", '
            '"steps": [{"goal": "investigate"}]}\n'
            'Let me know if you want changes.'
        )
        result = parse_json_response(raw)
        assert result["summary"] == "Fix the login bug"

    def test_status_completion_signal(self):
        """The execute phase uses JSON to signal step completion."""
        raw = '{"status": "completed", "summary": "Modified token validation to support refresh tokens"}'
        result = parse_json_response(raw)
        assert result["status"] == "completed"

    def test_status_with_preamble(self):
        raw = (
            "I've completed all the changes. "
            '{"status": "completed", "summary": "All tests pass"}'
        )
        result = parse_json_response(raw)
        assert result["status"] == "completed"


class TestJsonRepairFallback:
    """Stage 5 — hand off to json_repair when brace extraction fails.

    These inputs intentionally break earlier stages: truncated JSON, unquoted
    keys, prose preamble with braces inside, single-quoted strings. json_repair
    is the last line of defence before we raise ParseError.
    """

    def test_truncated_json_gets_repaired(self):
        # Real-world Minimax failure: output cut off by max_tokens mid-object.
        raw = '{"status": "in_progress", "summary": "Found the bug in '
        result = parse_json_response(raw)
        assert result["status"] == "in_progress"

    def test_unquoted_keys_get_repaired(self):
        raw = "{status: 'ok', count: 3}"
        result = parse_json_response(raw)
        assert result["status"] == "ok"
        assert result["count"] == 3

    def test_single_quoted_strings_get_repaired(self):
        # A common Minimax / DeepSeek failure: Python-style single quotes
        # instead of JSON double quotes.
        raw = "{'status': 'in_progress', 'plan': ['step1', 'step2']}"
        result = parse_json_response(raw)
        assert result["status"] == "in_progress"
        assert result["plan"] == ["step1", "step2"]
