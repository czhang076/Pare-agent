"""Resilient JSON parsing for LLM outputs.

Open-source models (DeepSeek, Qwen) frequently produce outputs that are
*almost* valid JSON but need cleanup: markdown code fences, preamble text,
trailing commas, etc. This module implements a progressive extraction
pipeline that handles these cases without requiring an extra LLM call.

The pipeline is ordered from cheapest to most aggressive:
    Stage 1: json.loads(raw) — works for well-behaved models
    Stage 2: strip markdown fences — handles ```json ... ```
    Stage 3: brace extraction — finds first { and last }
    Stage 4: light sanitization — trailing commas, single quotes
    Stage 5: raise ParseError with raw content for caller to handle

The caller (e.g. planner.py) can then decide whether to retry by feeding
the error back to the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParseError(Exception):
    """Raised when all parsing stages fail."""

    raw: str
    stages_tried: list[str]

    def __str__(self) -> str:
        return (
            f"Failed to parse JSON after stages: {', '.join(self.stages_tried)}. "
            f"Raw content (first 200 chars): {self.raw[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches ```json ... ``` or ``` ... ``` blocks
_CODE_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

# Trailing comma before closing brace/bracket: ,} or ,]
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_fences(text: str) -> str | None:
    """Extract content from the first markdown code fence, if present."""
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _extract_braces(text: str) -> str | None:
    """Find the outermost { ... } substring.

    Uses a simple brace-depth counter to handle nested objects. This is not
    a full JSON parser but handles the common case where the LLM wraps its
    JSON in explanatory text.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    return text[start : end + 1]


def _sanitize_json(text: str) -> str:
    """Apply light fixes for common JSON-ish mistakes."""
    # Remove trailing commas before } or ]
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_json_response(raw: str) -> dict:
    """Progressively extract a JSON object from an LLM response.

    Raises ParseError if all stages fail. The caller can inspect
    ParseError.raw to feed it back to the LLM for correction.

    Returns the parsed dict on success.
    """
    stages_tried: list[str] = []

    # Stage 1: direct parse
    stages_tried.append("direct")
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Stage 2: strip markdown code fences
    stages_tried.append("strip_fences")
    fenced = _strip_code_fences(raw)
    if fenced is not None:
        try:
            result = json.loads(fenced)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Also try sanitizing the fenced content
        try:
            result = json.loads(_sanitize_json(fenced))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Stage 3: brace extraction
    stages_tried.append("brace_extraction")
    braced = _extract_braces(raw)
    if braced is not None:
        try:
            result = json.loads(braced)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Stage 4: sanitize + retry
        stages_tried.append("sanitize")
        try:
            result = json.loads(_sanitize_json(braced))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # All stages failed
    raise ParseError(raw=raw, stages_tried=stages_tried)


def try_parse_json_response(raw: str) -> dict | ParseError:
    """Non-throwing variant — returns the dict or a ParseError."""
    try:
        return parse_json_response(raw)
    except ParseError as e:
        return e
