"""out_of_scope - fixed refusal handler, no LLM call.

Engineered-only: the naive arm has no refusal path and answers every query,
which is exactly the stock-RAG behaviour under study.
"""
from __future__ import annotations

import logging

from src.models.state import AgentState
from src.prompts import REFUSAL_RESPONSE

logger = logging.getLogger(__name__)


def out_of_scope_node(state: AgentState) -> dict:
    """Return a fixed polite refusal without any LLM call."""
    logger.info("Out of scope handler: returning fixed refusal")
    return {"response": REFUSAL_RESPONSE, "cited_sources": [], "cited_unknown": []}
