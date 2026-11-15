from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ---- input -----------------------------------------------------------
    user_id: str
    query: str
    conversation_history: list[dict[str, str]]
    turn_number: int
    run_config: dict[str, Any]

    # ---- router output ---------------------------------------------------
    intent: str
    needs_retrieval: bool
    search_query: str
    user_facts: list[str]
    router_reasoning: str
    retrieval_metadata_filter: str | None
    department_filter_requested: str | None
    router_usage: dict[str, Any]
    router_latency_ms: float

    # ---- retrieval output ------------------------------------------------
    retrieved_chunks: list[dict[str, Any]]
    retrieval_query: str
    retrieval_k: int
    retrieval_filter_applied: str | None
    searches: int
    retrieval_latency_ms: float

    # ---- memory reader ---------------------------------------------------
    memory_entries: list[dict[str, Any]]

    # ---- context enforcer output -----------------------------------------
    trimmed_chunks: list[dict[str, Any]]
    trimmed_history: list[dict[str, str]]
    trimmed_memory: list[dict[str, Any]]
    budget_log: dict[str, Any]

    # ---- synthesis output ------------------------------------------------
    prompt_text: str
    response: str
    cited_sources: list[str]
    cited_unknown: list[str]
    synthesizer_usage: dict[str, Any]
    synthesizer_latency_ms: float

    # ---- memory writer output --------------------------------------------
    new_memory_entries: list[dict[str, Any]]
    memory_write_error: str | None
    memory_writer_usage: dict[str, Any]
    memory_writer_latency_ms: float
