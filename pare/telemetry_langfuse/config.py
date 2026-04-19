"""Langfuse client configuration — env vars and connection setup.

R0 scaffold. Real values populated in W3 Day 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LangfuseConfig:
    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"


def from_env() -> LangfuseConfig:
    """Build a config from ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
    ``LANGFUSE_HOST``. Raises ``RuntimeError`` if required vars are missing."""
    raise NotImplementedError("W3 Day 1")
