"""
retriever - Semantic search node against Pincone.
Takes : query, retrieval_metadata_filter. Returns:
retrieved_chunks.
"""
import logging 

from src.config import RETRIEVAL_TOP_K
from src.models.state import AgentState
from src.stores import vector_store

logger = logging.getLogger(__name__)

def retriever_node(state: AgentState) -> dict:
    """Query vector store for relevant chunks."""
    query = state.get("query", "")
    metadata_filter = state.get("retrieval_metadata_filter")

    chunks = vector_store.query(
        text = query,
        k = RETRIEVAL_TOP_K,
        metadata_filter = metadata_filter,
    )

    logger.info(
        "Retriever: %d chunks returned (filter=%s, top_score=%.3f)",
        len(chunks),
        metadata_filter,
        chunks[0]["score"] if chunks else 0.0,
    )

    return {"retrieved_chunks" : chunks}
