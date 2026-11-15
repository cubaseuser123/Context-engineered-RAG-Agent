"""
tracing - Langfuse observability bootstrap.
Call init_tracing() once at startup or use get_client() to access Langfuse.
"""
from __future__ import annotations

import logging
import os
from typing import Any
from langfuse import Langfuse, observe

logger = logging.getLogger(__name__)

_client: Langfuse | None = None


def init_tracing() -> Langfuse | None:
    """Initialize Langfuse client from environment variables."""
    global _client
    if _client is not None:
        return _client

    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://us.cloud.langfuse.com"
    os.environ["LANGFUSE_HOST"] = host

    try:
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=host,
        )
        logger.info("Langfuse tracing initialized (host=%s)", host)
    except Exception as e:
        logger.warning("Could not initialize Langfuse: %s", e)

    return _client


def get_client() -> Langfuse | None:
    global _client
    if _client is None:
        return init_tracing()
    return _client


def flush() -> None:
    """Flush pending telemetry before process exit."""
    if _client is not None:
        try:
            _client.flush()
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)
