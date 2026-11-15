"""vector_store - Pinecone ingestion and retrieval.

Chunk IDs are the vector IDs and come back from ``query()``, so retrieval is
scored on chunk identity. Ingestion is a full replace: a stale namespace shares
every query with the new corpus and silently corrupts retrieval numbers.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from pinecone import NotFoundException, Pinecone, ServerlessSpec

from src import llm
from src.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
    RETRIEVAL_TOP_K,
)
from src.documents import chunk_document, load_documents

logger = logging.getLogger(__name__)

UPSERT_BATCH = 100

_pc: Pinecone | None = None
_index = None


def _get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        if not PINECONE_API_KEY:
            raise llm.ProviderNotConfigured("PINECONE_API_KEY is not set in .env")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def _get_index():
    global _index
    if _index is not None:
        return _index
    pc = _get_pc()
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s'", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def namespace_stats(namespace: str = PINECONE_NAMESPACE) -> dict:
    """Vector count and dimension for one namespace."""
    index = _get_index()
    described = index.describe_index_stats()
    namespaces = getattr(described, "namespaces", None) or {}
    entry = namespaces.get(namespace) or {}
    if hasattr(entry, "vector_count"):
        count = entry.vector_count
    elif isinstance(entry, dict):
        count = entry.get("vector_count", 0)
    else:
        count = 0
    return {
        "namespace": namespace,
        "vector_count": count,
        "total_vector_count": getattr(described, "total_vector_count", None),
        "dimension": getattr(described, "dimension", None),
    }


def delete_namespace(namespace: str = PINECONE_NAMESPACE) -> dict:
    """Empty a namespace while keeping the index definition (768-d, cosine).

    Pinecone answers 404 "Namespace not found" when there is nothing to delete;
    that is the desired end state, not a failure.
    """
    index = _get_index()
    try:
        index.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        logger.info("Namespace '%s' was already empty", namespace)
    after = namespace_stats(namespace)
    logger.info("Namespace '%s' emptied: %s", namespace, after)
    return after


def ingest(
    docs_dir: Path, *, wipe: bool = True, namespace: str = PINECONE_NAMESPACE
) -> dict:
    """Full-replace ingestion. Returns counts used by the P1 acceptance check."""
    index = _get_index()
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No .txt documents found in {docs_dir}")

    if wipe:
        delete_namespace(namespace)

    total_chunks = 0
    per_department: Counter[str] = Counter()
    vectors: list[tuple] = []

    for doc in documents:
        chunks = chunk_document(doc["doc_id"], doc["body"])
        if not chunks:
            logger.warning("Document '%s' produced no chunks", doc["doc_id"])
            continue
        embeddings = llm.embed(
            [c["content"] for c in chunks], task_type="RETRIEVAL_DOCUMENT"
        )
        for chunk, vector in zip(chunks, embeddings):
            vectors.append(
                (
                    chunk["chunk_id"],
                    vector,
                    {
                        "text": chunk["content"],
                        "doc_id": doc["doc_id"],
                        "source_doc": doc["doc_id"],
                        "department": doc["department"],
                        "section": chunk["section"],
                        "chunk_index": chunk["chunk_index"],
                        "version": doc["version"],
                    },
                )
            )
        per_department[doc["department"]] += 1
        total_chunks += len(chunks)
        logger.info("%s -> %d chunks", doc["doc_id"], len(chunks))

    for start in range(0, len(vectors), UPSERT_BATCH):
        index.upsert(vectors=vectors[start : start + UPSERT_BATCH], namespace=namespace)

    stats = namespace_stats(namespace)
    logger.info("Ingested %d chunks across %d docs", total_chunks, len(documents))
    return {
        "documents": len(documents),
        "chunks": total_chunks,
        "by_department": dict(sorted(per_department.items())),
        "namespace": stats,
        "wiped_first": wipe,
    }


def query(
    text: str,
    k: int = RETRIEVAL_TOP_K,
    metadata_filter: str | None = None,
    *,
    namespace: str = PINECONE_NAMESPACE,
) -> list[dict]:
    """Semantic search returning chunk identity, not just prose."""
    index = _get_index()
    vector = llm.embed_query(text)
    pinecone_filter = (
        {"department": {"$eq": metadata_filter}} if metadata_filter else None
    )
    results = index.query(
        vector=vector,
        top_k=k,
        namespace=namespace,
        include_metadata=True,
        filter=pinecone_filter,
    )
    chunks = []
    for match in results.matches:
        meta = match.metadata or {}
        chunks.append(
            {
                "chunk_id": match.id,
                "content": meta.get("text", ""),
                "source_doc": meta.get("doc_id", meta.get("source_doc", "")),
                "section": meta.get("section", ""),
                "department": meta.get("department", ""),
                "score": float(match.score),
            }
        )
    logger.info(
        "Query returned %d chunks (k=%d, filter=%s, top_score=%.3f)",
        len(chunks),
        k,
        metadata_filter,
        chunks[0]["score"] if chunks else 0.0,
    )
    return chunks
