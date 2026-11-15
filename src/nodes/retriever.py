"""retriever - semantic search against Pinecone.

The naive arm reaches this node without a router, so there is no filter and no
intent; the filter it *would* have used is recorded as ``None`` rather than
invented.
"""
from __future__ import annotations

import logging
import time
import re

from src.errors import RetrievalError
from src.models.state import AgentState
from src.observability import span
from src.run_config import RunConfig
from src.stores import vector_store

logger = logging.getLogger(__name__)


def retriever_node(state: AgentState) -> dict:
    """Query the vector store for the configured number of chunks."""
    config = RunConfig.from_dict(state.get("run_config"))
    query = (state.get("search_query") if config.routing else None) or state.get("query", "")
    # Department is a preference, not a hard exclusion of cross-policy evidence.
    preferred_department = state.get("retrieval_metadata_filter")
    metadata_filter = None if config.routing else state.get("retrieval_metadata_filter")

    started = time.perf_counter()
    with span(
        "node.retriever",
        **{
            "node.name": "retriever",
            "arm": config.arm,
            "retrieval.k": config.retrieval_k,
            "retrieval.filter": metadata_filter,
            "retrieval.query": query,
        },
    ) as annotate:
        try:
            chunks = vector_store.query(
                text=query, k=config.retrieval_k, metadata_filter=metadata_filter
            )
        except Exception as exc:  # noqa: BLE001 - wrapped into the record
            raise RetrievalError(str(exc)) from exc
        if config.enforce_budget:
            terms = set(re.findall(r"[a-z0-9]{4,}", query.lower())) - {"what", "does", "that", "this", "with", "have", "from", "when", "which", "their"}
            selected = {}
            for original in chunks:
                chunk = dict(original)
                text = " ".join(chunk.get("content", "").lower().split())
                if not text:
                    continue
                words = set(re.findall(r"[a-z0-9]{4,}", text))
                relevance = len(terms & words) / max(1, len(terms))
                chunk["vector_score"] = chunk["score"]
                chunk["score"] += 0.15 * relevance
                if chunk.get("department") == preferred_department:
                    chunk["score"] += 0.02
                if terms & {"exception", "exceptions", "tenure", "tenured"} and words & {"exception", "exceptions", "transitional", "grace"}:
                    chunk["score"] += 0.1
                key = (chunk.get("source_doc"), text)
                if key not in selected or chunk["score"] > selected[key]["score"]:
                    selected[key] = chunk
            chunks = sorted(selected.values(), key=lambda c: c["score"], reverse=True)
        annotate(
            **{
                "retrieval.chunk_ids": [c["chunk_id"] for c in chunks],
                "retrieval.scores": [round(c["score"], 4) for c in chunks],
                "retrieval.vector_scores": [c.get("vector_score", c["score"]) for c in chunks],
            }
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "Retriever: %d chunks (k=%d, filter=%s, top=%.3f)",
        len(chunks),
        config.retrieval_k,
        metadata_filter,
        chunks[0]["score"] if chunks else 0.0,
    )

    return {
        "retrieved_chunks": chunks,
        "retrieval_query": query,
        "retrieval_k": config.retrieval_k,
        "retrieval_filter_applied": metadata_filter if metadata_filter else None,
        "searches": 1,
        "retrieval_latency_ms": latency_ms,
    }
