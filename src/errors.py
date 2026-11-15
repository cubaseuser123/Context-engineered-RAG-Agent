"""errors - a typed failure taxonomy for run records.

A failed call is data: it counts against completion rate and API error rate.
What it must never do is become an empty row that silently reads as a wrong
answer.
"""
from __future__ import annotations

from src import llm


class AgentError(RuntimeError):
    """Base class; ``error_type`` is what the run record stores."""

    error_type = "agent_error"


class RouterParseError(AgentError):
    error_type = "router_json_parse"


class MemoryExtractionParseError(AgentError):
    error_type = "memory_json_parse"


class RetrievalError(AgentError):
    error_type = "retrieval_error"


def classify(exc: BaseException) -> tuple[str, str]:
    """Return ``(error_type, message)`` for the record."""
    if isinstance(exc, llm.ProviderNotConfigured):
        return "provider_not_configured", str(exc)
    if isinstance(exc, llm.ProviderRateLimited):
        return "provider_rate_limit", str(exc)
    if isinstance(exc, llm.ProviderError):
        return "provider_error", str(exc)
    if isinstance(exc, AgentError):
        return exc.error_type, str(exc)
    if isinstance(exc, TimeoutError):
        return "timeout", str(exc) or "timed out"
    return "unexpected", f"{type(exc).__name__}: {exc}"
