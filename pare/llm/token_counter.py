"""Token estimation utilities.

Provides fast, approximate token counting for context management decisions.
The context manager needs to check "are we over 70% of the limit?" on every
turn — this must be cheap (no API calls, no heavy imports on the hot path).

Strategy:
- Character heuristic (~3.5 chars/token) for quick threshold checks.
- tiktoken for accurate OpenAI model counting when available.
- Anthropic exact counting deferred to an explicit API call by the caller.

The heuristic is intentionally conservative (slightly over-estimates token
count) so that compaction triggers a bit early rather than a bit late.
"""

from __future__ import annotations

from pare.llm.base import Message

# Average characters per token. This is a rough heuristic that works
# reasonably for English text mixed with code. Different tokenizers give
# different ratios, but for a threshold check (not a billing calculation)
# this is sufficient.
_CHARS_PER_TOKEN = 3.5


def estimate_tokens_heuristic(messages: list[Message]) -> int:
    """Fast character-based token estimate. No external dependencies."""
    total_chars = sum(len(msg.text_content()) for msg in messages)
    return int(total_chars / _CHARS_PER_TOKEN)


def estimate_tokens_tiktoken(messages: list[Message], model: str = "gpt-4o") -> int:
    """Accurate token count using tiktoken (OpenAI models only).

    Lazy-imports tiktoken to avoid paying the import cost when not needed.
    Falls back to heuristic if tiktoken is not installed or the model
    encoding is unknown.
    """
    try:
        import tiktoken
    except ImportError:
        return estimate_tokens_heuristic(messages)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Unknown model — fall back to cl100k_base (GPT-4 family)
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        text = msg.text_content()
        total += len(encoding.encode(text))
        # Overhead per message (role, separators) — approximate
        total += 4

    return total
