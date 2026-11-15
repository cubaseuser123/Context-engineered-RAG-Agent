"""observability - Langfuse trace and span helpers for the study.

Traces map to individual evaluation turns / agent runs, with nested spans
for retrieval, synthesis, router, and context compaction.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from src.tracing import get_client

logger = logging.getLogger(__name__)


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Span context manager compatible with Langfuse tracing.

    Provides an ``annotate(**more)`` callback for late-bound attributes.
    """
    client = get_client()
    context = client.start_as_current_observation(name=name, as_type="span", metadata=attributes) if client else nullcontext(None)
    with context as span_obj:
        def annotate(**more: Any) -> None:
            if span_obj:
                span_obj.update(metadata=more)
        yield annotate


@contextmanager
def turn_span(**attributes: Any) -> Iterator[None]:
    """Root span for one agent invocation, carrying run identity."""
    with span("agent.turn", **attributes):
        yield
