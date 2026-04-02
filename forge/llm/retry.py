"""Self-managed retry policy with exponential backoff.

Replaces SDK-level retry to give us control over which errors are retried,
how long we wait, and when to give up. Only network and rate-limit errors
are retried — client errors (400, 401, 403, 404) fail immediately.

Key design decisions:
- Exponential backoff with jitter to avoid thundering herd.
- Respects Retry-After header from rate-limit responses.
- Configurable max retries, base delay, and max delay.
- Classifies errors into retryable/non-retryable by exception type,
  not by introspecting HTTP status codes (avoids coupling to SDK internals).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Exception class names that are safe to retry. We match by name rather than
# importing from each SDK to stay provider-agnostic. Both Anthropic and OpenAI
# SDKs use similar names for these error classes.
_RETRYABLE_ERROR_NAMES: frozenset[str] = frozenset(
    {
        # Anthropic SDK
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        "OverloadedError",
        # OpenAI SDK
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
        # Generic
        "ConnectError",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutError",
    }
)


def is_retryable(error: Exception) -> bool:
    """Decide whether an error is worth retrying.

    We classify by exception class name so this module doesn't need to
    import any provider SDK. This is intentionally loose — if a new SDK
    introduces a retryable error with an unexpected name, we'll fail
    fast rather than retry blindly.
    """
    cls_name = type(error).__name__

    # Direct name match
    if cls_name in _RETRYABLE_ERROR_NAMES:
        return True

    # Some SDKs wrap errors — check the cause chain
    if error.__cause__ is not None:
        return is_retryable(error.__cause__)

    return False


def _get_retry_after(error: Exception) -> float | None:
    """Extract Retry-After seconds from the error, if available.

    Both Anthropic and OpenAI SDKs expose response headers on their
    exception objects, but the attribute paths differ.
    """
    # Try Anthropic-style: error.response.headers
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    return None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap on the backoff delay.
        jitter: If True, add random jitter to prevent synchronized retries.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True

    def compute_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Compute the delay before retry attempt `attempt` (0-indexed).

        If the server provided a Retry-After value, we use at least that.
        Otherwise, exponential backoff: base_delay * 2^attempt, capped at
        max_delay, with optional jitter.
        """
        exponential = min(self.base_delay * (2**attempt), self.max_delay)

        if retry_after is not None:
            delay = max(exponential, retry_after)
        else:
            delay = exponential

        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311

        return delay


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
) -> T:
    """Execute an async function with retry on retryable errors.

    Usage:
        response = await with_retry(
            lambda: client.messages.create(...),
            policy=RetryPolicy(max_retries=3),
        )

    Non-retryable errors propagate immediately.
    """
    if policy is None:
        policy = RetryPolicy()

    last_error: Exception | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e

            if not is_retryable(e):
                raise

            if attempt >= policy.max_retries:
                logger.warning(
                    "All %d retries exhausted for %s: %s",
                    policy.max_retries,
                    type(e).__name__,
                    e,
                )
                raise

            retry_after = _get_retry_after(e)
            delay = policy.compute_delay(attempt, retry_after)

            logger.info(
                "Retry %d/%d after %.1fs for %s: %s",
                attempt + 1,
                policy.max_retries,
                delay,
                type(e).__name__,
                e,
            )

            await asyncio.sleep(delay)

    # Should never reach here, but satisfy type checker
    assert last_error is not None  # noqa: S101
    raise last_error
