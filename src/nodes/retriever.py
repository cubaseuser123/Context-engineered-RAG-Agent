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
from src.config import CORPUS_DIR
from src.documents import normalize_doc_id, parse_document, chunk_document
from src.nodes.context_enforcer import zone_caps
from src.prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.token_utils import count_tokens

logger = logging.getLogger(__name__)


def _expand_policies(chunks: list[dict], config: RunConfig) -> list[dict]:
    """Keep short matching policies whole, so budgeting cannot split exceptions.

    The vector hit IDs remain attached as provenance. A full-policy block has
    its own ID rather than pretending it was a single indexed chunk. Unknown,
    stale or oversized documents retain their original search fragments.
    """
    cap = zone_caps(config, count_tokens(SYNTHESIZER_SYSTEM_PROMPT))["retrieval"]
    expanded = set()
    result = []
    for chunk in chunks:
        doc_id = normalize_doc_id(chunk.get("source_doc", ""))
        if doc_id in expanded:
            continue
        path = CORPUS_DIR / f"{doc_id}.txt"
        if path.is_file():
            doc = parse_document(path)
            body = doc["body"]
            # Do not replace remotely indexed evidence with a different local version.
            matches = " ".join(chunk["content"].split()) in " ".join(body.split())
            if matches and (cap is None or count_tokens(body) <= cap):
                hits = [c["chunk_id"] for c in chunks if normalize_doc_id(c.get("source_doc", "")) == doc_id]
                result.append({**chunk, "chunk_id": f"{doc_id}::full",
                               "content": body, "tokens": count_tokens(body),
                               "section": "Complete policy", "evidence_scope": "full_document",
                               "seed_chunk_ids": hits,
                               "expanded_chunk_ids": [c["chunk_id"] for c in chunk_document(doc_id, body)]})
                expanded.add(doc_id)
                continue
        result.append(chunk)
    return result


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
            chunks = _expand_policies(chunks, config)
        annotate(
            **{
                "retrieval.chunk_ids": [c["chunk_id"] for c in chunks],
                "retrieval.scores": [round(c["score"], 4) for c in chunks],
                "retrieval.vector_scores": [c.get("vector_score", c["score"]) for c in chunks],
                "retrieval.expansions": [
                    {"source_doc": c["source_doc"], "seed_chunk_ids": c["seed_chunk_ids"],
                     "expanded_chunk_ids": c["expanded_chunk_ids"]}
                    for c in chunks if c.get("evidence_scope") == "full_document"
                ],
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
