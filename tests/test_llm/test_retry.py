"""Tests for forge/llm/retry.py."""

import asyncio

import pytest

from forge.llm.retry import RetryPolicy, is_retryable, with_retry


# ---------------------------------------------------------------------------
# Fixtures — fake exceptions that mimic SDK error class names
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    pass


class InternalServerError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class BadRequestError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class NotFoundError(Exception):
    pass


class RateLimitWithRetryAfter(Exception):
    """Simulates an error with a Retry-After header."""

    def __init__(self, retry_after: float):
        super().__init__("rate limited")

        class FakeResponse:
            headers = {"retry-after": str(retry_after)}

        self.response = FakeResponse()


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_rate_limit_is_retryable(self):
        assert is_retryable(RateLimitError()) is True

    def test_internal_server_error_is_retryable(self):
        assert is_retryable(InternalServerError()) is True

    def test_connection_error_is_retryable(self):
        assert is_retryable(APIConnectionError()) is True

    def test_bad_request_is_not_retryable(self):
        assert is_retryable(BadRequestError()) is False

    def test_auth_error_is_not_retryable(self):
        assert is_retryable(AuthenticationError()) is False

    def test_not_found_is_not_retryable(self):
        assert is_retryable(NotFoundError()) is False

    def test_generic_exception_is_not_retryable(self):
        assert is_retryable(ValueError("oops")) is False

    def test_wrapped_retryable_error(self):
        """Retryable error wrapped in a generic exception via __cause__."""
        outer = RuntimeError("wrapper")
        outer.__cause__ = RateLimitError("inner")
        assert is_retryable(outer) is True


# ---------------------------------------------------------------------------
# RetryPolicy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_default_values(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.jitter is True

    def test_exponential_backoff_no_jitter(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=60.0, jitter=False)
        assert policy.compute_delay(0) == 1.0   # 1 * 2^0
        assert policy.compute_delay(1) == 2.0   # 1 * 2^1
        assert policy.compute_delay(2) == 4.0   # 1 * 2^2
        assert policy.compute_delay(3) == 8.0   # 1 * 2^3

    def test_max_delay_cap(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=False)
        assert policy.compute_delay(10) == 10.0  # 1 * 2^10 = 1024, capped at 10

    def test_retry_after_respected(self):
        policy = RetryPolicy(base_delay=1.0, jitter=False)
        # retry_after of 30s should override the exponential 1s
        delay = policy.compute_delay(0, retry_after=30.0)
        assert delay == 30.0

    def test_jitter_within_bounds(self):
        policy = RetryPolicy(base_delay=2.0, jitter=True)
        for _ in range(50):
            delay = policy.compute_delay(0)
            # Jitter: delay * [0.5, 1.0), so for base=2: [1.0, 2.0)
            assert 1.0 <= delay < 2.0


# ---------------------------------------------------------------------------
# with_retry integration tests
# ---------------------------------------------------------------------------


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        call_count = 0

        async def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await with_retry(succeed, RetryPolicy(max_retries=3))
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self):
        call_count = 0

        async def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RateLimitError("rate limited")
            return "ok"

        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
        result = await with_retry(fail_twice_then_succeed, policy)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("rate limited")

        policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)
        with pytest.raises(RateLimitError):
            await with_retry(always_fail, policy)
        assert call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self):
        call_count = 0

        async def bad_request():
            nonlocal call_count
            call_count += 1
            raise BadRequestError("invalid input")

        policy = RetryPolicy(max_retries=3, base_delay=0.01)
        with pytest.raises(BadRequestError):
            await with_retry(bad_request, policy)
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_no_retries_policy(self):
        call_count = 0

        async def fail():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("rate limited")

        policy = RetryPolicy(max_retries=0)
        with pytest.raises(RateLimitError):
            await with_retry(fail, policy)
        assert call_count == 1
